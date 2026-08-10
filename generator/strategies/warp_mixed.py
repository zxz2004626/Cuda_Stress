"""warp_mixed strategy.

Stress targets: __ballot / __any / __all / __activemask / __shfl(_up/_down/_xor) /
__match_any / __match_all, full and legal partial member masks, followed by a long
integer chain, predicates and stores.

Legal-mask rule: for partial masks every warp op is guarded by `if (lane < 16u)` so the
calling lane is always a member of the mask it uses (no UB).  __activemask is exercised
both converged (full mask) and inside that divergent region (partial active mask).

The CPU oracle simulates warp semantics lane-by-lane and byte-identically reproduces the
GPU output.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..base import GeneratedCase, Strategy
from .helpers import MASK32, MASK64, make_input_words, pack_words, unpack_words

GOLDEN = 0x9E3779B97F4A7C15
FULL = 0xFFFFFFFF
PARTIAL_MASK = 0xFFFF


class WarpMixed(Strategy):
    name = "warp_mixed"
    target_opcodes = ["ballot", "shfl", "match", "vote", "activemask"]

    OPS = ["ballot", "any", "all", "activemask", "shfl_idx", "shfl_up", "shfl_down",
           "shfl_xor", "match_any", "match_all"]

    def _fmt_params(self, p: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "warp_ops": [o for o in p["warp_ops"] if o in self.OPS],
            "chain_len": int(p["chain_len"]),
            "partial_masks": bool(p["partial_masks"]),
            "boundary_words": int(p["boundary_words"]),
        }

    # ------------------------------------------------------------------ CUDA
    def _emit_cuda(self, p: Dict[str, Any], sh: int) -> str:
        ops = p["warp_ops"]
        chain = p["chain_len"]
        partial = p["partial_masks"]

        # Legal-member-mask rule: with a partial mask (lanes 0..15) shuffles use
        # width=16 so every source lane stays inside the mask (no UB).  Full mask
        # uses width=32.
        width = "16u" if partial else "32u"
        srcmask = "15u" if partial else "31u"

        # Build per-op lines (all use member mask m, legal per the guard above).
        op_lines = []
        if "ballot" in ops:
            op_lines.append(("b = __ballot_sync(m, pred);").rstrip())
        if "any" in ops:
            op_lines.append(("an = __any_sync(m, pred);").rstrip())
        if "all" in ops:
            op_lines.append(("al = __all_sync(m, pred);").rstrip())
        if "shfl_idx" in ops:
            op_lines.append((f"s = __shfl_sync(m, v, (lane + {sh}u) & {srcmask}, {width});").rstrip())
        if "shfl_up" in ops:
            op_lines.append((f"s = __shfl_up_sync(m, v, {sh}u, {width});").rstrip())
        if "shfl_down" in ops:
            op_lines.append((f"s = __shfl_down_sync(m, v, {sh}u, {width});").rstrip())
        if "shfl_xor" in ops:
            op_lines.append((f"s = __shfl_xor_sync(m, v, {sh}u, {width});").rstrip())
        if "match_any" in ops:
            op_lines.append(("ma = __match_any_sync(m, vu);").rstrip())
        if "match_all" in ops:
            op_lines.append(("mall = __match_all_sync(m, vu, &mallp);").rstrip())

        warp_block = "\n".join(op_lines)
        if partial and warp_block:
            warp_block = f"if (lane < 16u)\n    {{\n    {warp_block}\n    }}"

        ops_joined = ", ".join(ops) if ops else "none"
        return f"""\
// strategy=warp_mixed  (generated; do not edit)
extern "C" __global__ void test_kernel(
    const unsigned long long* __restrict__ input,
    unsigned long long* __restrict__ output,
    unsigned int input_words,
    unsigned long long seed)
{{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int lane = threadIdx.x & 31u;
    unsigned int warpid = (blockIdx.x * (blockDim.x >> 5u)) + (threadIdx.x >> 5u);
    unsigned int base = warpid * 32u;

    unsigned long long v = input[(base + lane) % input_words] ^ seed;
    unsigned int vu = (unsigned int)(v) ^ (unsigned int)(v >> 32u);
    int pred = (int)((vu & 0x80000000u) != 0u);

    const unsigned int SH = {sh}u;
    unsigned int m = 0xFFFFFFFFu;
    if ({1 if partial else 0}u) m = (lane < 16u) ? 0xFFFFu : 0xFFFFFFFFu;

    // -- warp ops (ops={ops_joined}) -----------------------------------------
    unsigned int b = 0u, an = 0u, al = 0u, am = 0xFFFFFFFFu, ma = 0u, mall = 0u;
    int mallp = 0;
    unsigned long long s = v;
{warp_block}
    // activemask in a partially-divergent region exercises partial active masks.
    am = __activemask();
    if (lane < 16u) {{ unsigned int am2 = __activemask(); am ^= am2; }}

    // -- long integer chain, predicates, store --------------------------------
    unsigned long long acc = seed ^ v;
    acc = acc * {GOLDEN}ULL + (unsigned long long)b;
    acc ^= s;
    acc = acc * {GOLDEN}ULL + (unsigned long long)ma;
    acc = acc * {GOLDEN}ULL + (unsigned long long)mall;
    for (unsigned int j = 0u; j < {chain}u; ++j)
        acc = acc * {GOLDEN}ULL + (input[(base + lane + j * 3u) % input_words] ^ (seed >> (j % 63u)));
    acc = pred ? (acc ^ s) : (acc + (s >> 1u));

    output[tid * 2u] = acc ^ (unsigned long long)am;
    output[tid * 2u + 1u] = acc ^ (unsigned long long)mallp ^ (unsigned long long)al
                            ^ (unsigned long long)an ^ v;
}}
"""

    # ------------------------------------------------------------------ oracle
    def cpu_oracle(self, input_words: int, seed: int, params: Dict[str, Any],
                   input_bytes: bytes, grid: List[int], block: List[int],
                   output_words_per_thread: int) -> bytes:
        p = self._fmt_params(params)
        words = unpack_words(input_bytes)
        ops = p["warp_ops"]
        chain = p["chain_len"]
        partial = p["partial_masks"]
        sh = (seed % 15) + 1

        nwarps = (block[0] // 32) * grid[0] if len(grid) else 1
        out: List[int] = []

        for warp in range(nwarps):
            base = warp * 32
            v = [(words[(base + lane) % input_words] ^ seed) & MASK64 for lane in range(32)]
            vu = [((vv & MASK32) ^ ((vv >> 32) & MASK32)) & MASK32 for vv in v]
            pred = [1 if (x & 0x80000000) else 0 for x in vu]

            # Per-lane results of each warp op (initial values for non-participating lanes).
            b = [0] * 32
            an = [0] * 32
            al = [0] * 32
            am = [FULL] * 32
            ma = [0] * 32
            mall = [0] * 32
            mallp = [0] * 32
            s = list(v)

            part_lanes = range(32)
            m = FULL
            seg = 32
            if partial:
                part_lanes = range(16)
                m = PARTIAL_MASK
                seg = 16

            for lane in part_lanes:
                if "ballot" in ops:
                    b[lane] = sum((1 << L) for L in part_lanes if pred[L]) & MASK32
                if "any" in ops:
                    an[lane] = int(any(pred[L] for L in part_lanes))
                if "all" in ops:
                    al[lane] = int(all(pred[L] for L in part_lanes))
                if "shfl_idx" in ops:
                    src = (lane + sh) & (seg - 1)
                    s[lane] = v[src]
                if "shfl_up" in ops:
                    s[lane] = v[lane - sh] if lane >= sh else v[lane]
                if "shfl_down" in ops:
                    s[lane] = v[lane + sh] if lane + sh < seg else v[lane]
                if "shfl_xor" in ops:
                    s[lane] = v[lane ^ sh]
                if "match_any" in ops:
                    ma[lane] = sum((1 << L) for L in part_lanes if vu[L] == vu[lane]) & MASK32
                if "match_all" in ops:
                    allsame = all(vu[L] == vu[lane] for L in part_lanes)
                    mallp[lane] = int(allsame)
                    mall[lane] = m if allsame else 0   # match_all returns the member mask

            # activemask: converged point -> full; divergent region -> partial active mask
            for lane in range(32):
                am[lane] = FULL
                if lane < 16:
                    am2 = PARTIAL_MASK
                    am[lane] ^= am2

            for lane in range(32):
                tid = warp * 32 + lane
                acc = (seed ^ v[lane]) & MASK64
                acc = (acc * GOLDEN + b[lane]) & MASK64
                acc ^= s[lane]
                acc = (acc * GOLDEN + ma[lane]) & MASK64
                acc = (acc * GOLDEN + mall[lane]) & MASK64
                for j in range(chain):
                    acc = (acc * GOLDEN + (words[(base + lane + j * 3) % input_words] ^ (seed >> (j % 63)))) & MASK64
                if pred[lane]:
                    acc ^= s[lane]
                else:
                    acc = (acc + (s[lane] >> 1)) & MASK64
                out.append((acc ^ am[lane]) & MASK64)
                out.append((acc ^ mallp[lane] ^ al[lane] ^ an[lane] ^ v[lane]) & MASK64)
        return pack_words(out)

    def generate(self, config_seed: int, test_seed: int, params: Dict[str, Any],
                 gen_version: str, target_sm: str) -> GeneratedCase:
        p = self._fmt_params(params)
        rng = self._new_rng(config_seed, test_seed)
        input_words = 256
        words = make_input_words(rng, input_words, p["boundary_words"])
        seed = rng.getrandbits(64)
        sh = (seed % 15) + 1
        source = self._emit_cuda(p, sh)
        grid, block = [1], [64]
        owpt = 2
        output_words = grid[0] * block[0] * owpt
        input_bytes = pack_words(words)
        expected = self.cpu_oracle(input_words, seed, p, input_bytes, grid, block, owpt)
        return GeneratedCase(
            test_id=f"warp_mixed__{config_seed:x}__{test_seed:x}",
            config_seed=config_seed, test_seed=test_seed, strategy=self.name, params=p,
            kernel_seed=seed,
            generator_version=gen_version, source=source,
            input_bytes=input_bytes, expected_bytes=expected,
            grid=grid, block=block, input_words=input_words, output_words=output_words,
            output_words_per_thread=owpt, target_sm=target_sm,
            cpu_oracle=self.name, comparison_policy="exact_u64",
            cuda_features=[target_sm, "warp_shfl_match", "sm_70_intrinsics"],
        )

    def describe_pressure(self, p: Dict[str, Any]) -> Dict[str, Any]:
        return {"ops": p["warp_ops"], "partial_masks": p["partial_masks"]}
