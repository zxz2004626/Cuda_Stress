"""register_pressure strategy.

Stress targets: 32-128 simultaneously-live u32/u64 values, multiple independent chains
that merge before use, __noinline__/__forceinline__ combinations, local spill, and
explicit launch_bounds / maxrregcount variants.

`maxrregcount` and `launch_bounds` variants change the ptxas constraints so register
allocation / spill decisions differ across profiles.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..base import GeneratedCase, Strategy
from ..rng import mix64
from .helpers import MASK64, make_input_words, pack_words, unpack_words

GOLDEN = 0x9E3779B97F4A7C15


class RegisterPressure(Strategy):
    name = "register_pressure"
    target_opcodes = ["LDL", "STL", "CALL", "IMAD"]

    def _fmt_params(self, p: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "live_vars": int(p["live_vars"]),
            "chains": int(p["chains"]),
            "merge_len": int(p["merge_len"]),
            "call_style": str(p["call_style"]),
            "variant": str(p["variant"]),
        }

    def _emit_cuda(self, p: Dict[str, Any], k_consts: List[int]) -> str:
        lv = p["live_vars"]
        chains = p["chains"]
        call = "noinline" if p["call_style"] == "noinline" else "forceinline"
        k_lit = ", ".join(f"{x}ULL" for x in k_consts)
        variant = p["variant"]

        lb = ""
        if variant == "launch_bounds":
            lb = "__launch_bounds__(64, 1) "

        # One chain function per chain id.  Each keeps LV live u64 values in flight.
        funcs = []
        for c in range(chains):
            attr = "__device__ __noinline__" if call == "noinline" else "__device__ __forceinline__"
            funcs.append(f"""\
{attr} unsigned long long chain_{c}(
    const unsigned long long* __restrict__ in, unsigned int n,
    unsigned long long seed, unsigned int tid)
{{
    unsigned long long v[{lv}];
    for (unsigned int i = 0u; i < {lv}u; ++i)
    {{
        unsigned int idx = (tid + i * 5u + {c}u * 7u) % n;
        v[i] = (in[idx] ^ (seed >> (i % 63u))) * K[i];
    }}
    unsigned long long acc = seed ^ {c}u;
    for (unsigned int i = 0u; i < {lv}u; ++i) acc = acc * {GOLDEN}ULL + v[i];
    for (unsigned int i = 0u; i < {lv}u; ++i)
        acc ^= v[(i * 17u) % {lv}u] >> ((i * 3u) & 31u);
    return acc;
}}
""")
        func_body = "\n".join(funcs)

        calls = "    ".join(f"acc = acc * {GOLDEN}ULL ^ chain_{c}(input, input_words, seed, tid);\n" for c in range(chains))

        return f"""\
// strategy=register_pressure  (generated; do not edit)
__constant__ unsigned long long K[{lv}] = {{ {k_lit} }};
{func_body}
extern "C" __global__ {lb}void test_kernel(
    const unsigned long long* __restrict__ input,
    unsigned long long* __restrict__ output,
    unsigned int input_words,
    unsigned long long seed)
{{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long long acc = seed;
{calls}
    for (unsigned int j = 0u; j < {p['merge_len']}u; ++j)
    {{
        unsigned int idx = (tid + j * 13u) % input_words;
        acc = acc * {GOLDEN}ULL + (input[idx] ^ (acc >> 33u));
    }}
    output[tid * 2u] = acc;
    output[tid * 2u + 1u] = acc ^ seed ^ input[tid % input_words];
}}
"""

    def cpu_oracle(self, input_words: int, seed: int, params: Dict[str, Any],
                   input_bytes: bytes, grid: List[int], block: List[int],
                   output_words_per_thread: int) -> bytes:
        p = self._fmt_params(params)
        words = unpack_words(input_bytes)
        lv, chains, merge_len = p["live_vars"], p["chains"], p["merge_len"]
        k_consts = [((mix64(seed ^ i)) & MASK64) | 1 for i in range(lv)]
        total = grid[0] * block[0]
        out: List[int] = []
        for tid in range(total):
            acc = seed
            for c in range(chains):
                v = [0] * lv
                for i in range(lv):
                    idx = (tid + i * 5 + c * 7) % input_words
                    v[i] = ((words[idx] ^ (seed >> (i % 63))) * k_consts[i]) & MASK64
                ca = (seed ^ c) & MASK64
                for i in range(lv):
                    ca = (ca * GOLDEN + v[i]) & MASK64
                for i in range(lv):
                    ca ^= v[(i * 17) % lv] >> ((i * 3) & 31)
                acc = (acc * GOLDEN ^ ca) & MASK64
            for j in range(merge_len):
                idx = (tid + j * 13) % input_words
                acc = (acc * GOLDEN + (words[idx] ^ (acc >> 33))) & MASK64
            out.append(acc & MASK64)
            out.append((acc ^ seed ^ words[tid % input_words]) & MASK64)
        return pack_words(out)

    def generate(self, config_seed: int, test_seed: int, params: Dict[str, Any],
                 gen_version: str, target_sm: str) -> GeneratedCase:
        p = self._fmt_params(params)
        rng = self._new_rng(config_seed, test_seed)
        input_words = 256
        words = make_input_words(rng, input_words, 8)
        seed = rng.getrandbits(64)
        lv = p["live_vars"]
        k_consts = [((mix64(seed ^ i)) & MASK64) | 1 for i in range(lv)]
        source = self._emit_cuda(p, k_consts)
        grid, block = [1], [64]
        owpt = 2
        output_words = grid[0] * block[0] * owpt
        input_bytes = pack_words(words)
        expected = self.cpu_oracle(input_words, seed, p, input_bytes, grid, block, owpt)
        extra_flags = []
        if p["variant"] == "maxrregcount":
            extra_flags.append("--maxrregcount=40")
        params_out = dict(p)
        if extra_flags:
            params_out["ptxas_extra"] = extra_flags
        return GeneratedCase(
            test_id=f"register_pressure__{config_seed:x}__{test_seed:x}",
            config_seed=config_seed, test_seed=test_seed, strategy=self.name, params=params_out,
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
            "live_vars": p["live_vars"],
            "chains": p["chains"],
            "call_style": p["call_style"],
            "variant": p["variant"],
        }
