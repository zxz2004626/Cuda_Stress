"""predicate_pressure strategy.

Stress targets: predicate rematerialization, MachineCSE, if-conversion, predicate
propagation, select lowering, predicate->GPR materialization, branch folding.

Structure per thread:
  * NCMP runtime comparisons -> predicates p[i], some duplicated far apart (CSE vs recompute)
  * a long GPR-heavy arithmetic region after the predicates are formed (register pressure,
    so surviving predicates must be rematerialized or spilled)
  * small diamonds (pa && pb) with an else arm
  * predicates used as branch conditions (forward) then as selects (reverse order)
  * all predicates folded into the output signature so they cannot be DCE'd
Variants may add __launch_bounds__.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..base import GeneratedCase, Strategy
from ..rng import mix64
from .helpers import MASK64, make_input_words, pack_words, unpack_words

MIX = 0x9E3779B97F4A7C15
CHAIN_MUL = 0x5851F42D4C957F2D


def _consts(seed: int, ncmp: int) -> "tuple[List[int], List[int]]":
    """Constant arrays K/A as a pure function of the kernel seed (shared by kernel+oracle)."""
    k = [((mix64(seed) ^ (i * 0x100000001B3)) & MASK64) | 1 for i in range(ncmp)]
    a = [((mix64(seed ^ MIX) ^ (i * MIX)) & MASK64) for i in range(ncmp)]
    return k, a


class PredicatePressure(Strategy):
    name = "predicate_pressure"
    target_opcodes = ["isetp", "psetp", "@p", "selp", "bra"]

    def _fmt_params(self, p: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "num_comparisons": int(p["num_comparisons"]),
            "chain_len": int(p["chain_len"]),
            "num_diamonds": int(p["num_diamonds"]),
            "num_folds": int(p["num_folds"]),
            "dup_groups": int(p["dup_groups"]),
            "use_launch_bounds": bool(p["use_launch_bounds"]),
        }

    def _emit_cuda(self, p: Dict[str, Any], k_consts: List[int], a_consts: List[int]) -> str:
        ncmp = p["num_comparisons"]
        chain = p["chain_len"]
        ndiam = p["num_diamonds"]
        nfold = p["num_folds"]
        dupg = p["dup_groups"]
        lb = "__launch_bounds__(64, 1) " if p["use_launch_bounds"] else ""

        k_lit = ", ".join(f"{x}ULL" for x in k_consts)
        a_lit = ", ".join(f"{x}ULL" for x in a_consts)

        return f"""\
// strategy=predicate_pressure  (generated; do not edit)
extern "C" __global__ {lb}void test_kernel(
    const unsigned long long* __restrict__ input,
    unsigned long long* __restrict__ output,
    unsigned int input_words,
    unsigned long long seed)
{{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    static const unsigned long long K[{ncmp}] = {{ {k_lit} }};
    static const unsigned long long A[{ncmp}] = {{ {a_lit} }};

    // -- runtime values ----------------------------------------------------
    unsigned long long x[{ncmp}];
    #pragma unroll 4
    for (unsigned int i = 0u; i < {ncmp}u; ++i)
    {{
        unsigned int idx = (tid + i * 7u) % input_words;
        x[i] = input[idx] ^ (seed + i * {MIX}ULL);
        x[i] = x[i] * K[i] + A[i];
    }}

    // -- predicates (runtime comparisons), with distant duplicates ---------
    int p[{ncmp}];
    #pragma unroll 2
    for (unsigned int i = 0u; i < {ncmp}u; ++i)
    {{
        unsigned int amt = (unsigned int)((seed + i * 5u) & 63u);
        p[i] = (int)(x[i] < (seed >> amt)) | (int)((x[i] >> 1u) >= (seed ^ i));
    }}
    // duplicated comparisons at distance: identical operands as p[src] far away.
    #pragma unroll 2
    for (unsigned int i = {ncmp - dupg}u; i < {ncmp}u; ++i)
    {{
        unsigned int src = (i * 7u) % {ncmp}u;
        unsigned int amt = (unsigned int)((seed + src * 5u) & 63u);
        p[i] = (int)(x[src] < (seed >> amt)) | (int)((x[src] >> 1u) >= (seed ^ src));
    }}

    // -- long GPR-heavy region (register pressure between predicate uses) --
    unsigned long long acc = seed ^ 0x123456789ABCDEF0ULL;
    for (unsigned int j = 0u; j < {chain}u; ++j)
    {{
        unsigned int idx = (tid + j * 3u + 1u) % input_words;
        acc = acc * {CHAIN_MUL}ULL + (input[idx] ^ x[j % {ncmp}u]);
        acc ^= acc >> 29u;
    }}

    // -- small diamonds ----------------------------------------------------
    for (unsigned int d = 0u; d < {ndiam}u; ++d)
    {{
        int pa = p[d % {ncmp}u], pb = p[(d + {ndiam // 2}u) % {ncmp}u];
        unsigned long long t = x[d % {ncmp}u] * K[d % {ncmp}u] + seed;
        if (pa && pb) acc += t;
        else          acc ^= (t >> 1u);
    }}

    // -- forward branch use ------------------------------------------------
    for (unsigned int i = 0u; i < {nfold}u; ++i)
    {{
        if (p[i]) acc ^= x[i];
        else      acc += (x[i] >> 1u);
    }}

    // -- reverse select use -------------------------------------------------
    for (unsigned int i = {nfold}u; i < {ncmp}u; ++i)
    {{
        unsigned long long t = (x[i] * K[i]) ^ seed;
        acc = p[i] ? (acc + t) : (acc ^ t);
    }}

    // -- fold all predicates into the signature (predicate -> GPR) ---------
    unsigned long long mask = 0ULL;
    for (unsigned int i = 0u; i < {ncmp}u; ++i)
        mask |= ((unsigned long long)(unsigned int)p[i]) << (i % 64u);

    output[tid * 2u] = acc ^ mask;
    output[tid * 2u + 1u] = acc ^ seed ^ mask;
}}
"""

    def cpu_oracle(self, input_words: int, seed: int, params: Dict[str, Any],
                   input_bytes: bytes, grid: List[int], block: List[int],
                   output_words_per_thread: int) -> bytes:
        p = self._fmt_params(params)
        words = unpack_words(input_bytes)
        ncmp, chain, ndiam, nfold, dupg = (
            p["num_comparisons"], p["chain_len"], p["num_diamonds"],
            p["num_folds"], p["dup_groups"])
        k_consts, a_consts = _consts(seed, ncmp)
        total = grid[0] * block[0]
        out: List[int] = []
        for tid in range(total):
            x = [0] * ncmp
            for i in range(ncmp):
                idx = (tid + i * 7) % input_words
                v = (words[idx] ^ (seed + i * MIX)) & MASK64
                x[i] = (v * k_consts[i] + a_consts[i]) & MASK64
            pt = [0] * ncmp
            for i in range(ncmp):
                amt = (seed + i * 5) & 63
                pt[i] = int((x[i] < ((seed >> amt) & MASK64)) or ((x[i] >> 1) >= (seed ^ i)))
            for i in range(ncmp - dupg, ncmp):
                src = (i * 7) % ncmp
                amt = (seed + src * 5) & 63
                pt[i] = int((x[src] < ((seed >> amt) & MASK64)) or ((x[src] >> 1) >= (seed ^ src)))
            acc = (seed ^ 0x123456789ABCDEF0) & MASK64
            for j in range(chain):
                idx = (tid + j * 3 + 1) % input_words
                acc = (acc * CHAIN_MUL + (words[idx] ^ x[j % ncmp])) & MASK64
                acc ^= acc >> 29
            for d in range(ndiam):
                pa, pb = pt[d % ncmp], pt[(d + ndiam // 2) % ncmp]
                t = (x[d % ncmp] * k_consts[d % ncmp] + seed) & MASK64
                if pa and pb:
                    acc = (acc + t) & MASK64
                else:
                    acc ^= t >> 1
            for i in range(nfold):
                if pt[i]:
                    acc ^= x[i]
                else:
                    acc = (acc + (x[i] >> 1)) & MASK64
            for i in range(nfold, ncmp):
                t = (x[i] * k_consts[i] ^ seed) & MASK64
                acc = (acc + t) & MASK64 if pt[i] else (acc ^ t) & MASK64
            mask = 0
            for i in range(ncmp):
                mask |= pt[i] << (i % 64)
            out.append((acc ^ mask) & MASK64)
            out.append((acc ^ seed ^ mask) & MASK64)
        return pack_words(out)

    def generate(self, config_seed: int, test_seed: int, params: Dict[str, Any],
                 gen_version: str, target_sm: str) -> GeneratedCase:
        p = self._fmt_params(params)
        rng = self._new_rng(config_seed, test_seed)
        input_words = 256
        words = make_input_words(rng, input_words, 8)
        ncmp = p["num_comparisons"]
        seed = rng.getrandbits(64)
        k_consts, a_consts = _consts(seed, ncmp)
        source = self._emit_cuda(p, k_consts, a_consts)
        grid, block = [1], [64]
        owpt = 2
        output_words = grid[0] * block[0] * owpt
        input_bytes = pack_words(words)
        expected = self.cpu_oracle(input_words, seed, p, input_bytes, grid, block, owpt)
        return GeneratedCase(
            test_id=f"predicate_pressure__{config_seed:x}__{test_seed:x}",
            config_seed=config_seed, test_seed=test_seed, strategy=self.name, params=p,
            kernel_seed=seed,
            generator_version=gen_version, source=source,
            input_bytes=input_bytes, expected_bytes=expected,
            grid=grid, block=block, input_words=input_words, output_words=output_words,
            output_words_per_thread=owpt, target_sm=target_sm,
            cpu_oracle=self.name, comparison_policy="exact_u64",
            cuda_features=[target_sm],
        )

    def describe_pressure(self, p: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "predicates": p["num_comparisons"],
            "diamonds": p["num_diamonds"],
            "dup_comparisons": p["dup_groups"],
            "gpr_chain_len": p["chain_len"],
        }
