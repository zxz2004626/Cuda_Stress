"""memory_address strategy.

Stress targets: u8/u16/u32/u64 accesses, v2/v4-aligned load/store, ld.global cache
operators (default / ca / cg / cs), multi-level runtime address computation,
struct/array indexing, local arrays (spill pressure), load merging/widening/reordering.

Alignment safety: every access is relative to a u64 slot `base` < input_words; narrower
types read at sub-offsets within that slot (natural alignment), vectors read at slots
that are a multiple of 2 (16-byte aligned).
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..base import GeneratedCase, Strategy
from .helpers import MASK64, make_input_words, pack_words, unpack_words

GOLDEN = 0x9E3779B97F4A7C15

# ld.global cache operator string fragments, keyed by config name.
_LDOP = {"default": "", "ca": "ca.", "cg": "cg.", "cs": "cs."}


class MemoryAddress(Strategy):
    name = "memory_address"
    target_opcodes = ["ld.global", "LDG", "LDG.E", "LDG.E.CG", "LDG.E.CS"]

    def _fmt_params(self, p: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "reads": int(p["reads"]),
            "widths": [int(w) for w in p["widths"]],
            "vectors": bool(p["vectors"]),
            "cache_policy": str(p["cache_policy"]),
            "structs": bool(p["structs"]),
        }

    def _load_helper(self, p: Dict[str, Any]) -> str:
        """File-scope device load function specialized for this testcase's widths+policy."""
        widths = p["widths"]
        op = _LDOP[p["cache_policy"]]
        cases = []
        if 8 in widths:
            cases.append(f'        case 8: {{ unsigned int v; asm("ld.global.{op}u8 %0, [%1];" : "=r"(v) : "l"(p)); return (unsigned long long)v; }}')
        if 16 in widths:
            cases.append(f'        case 16: {{ unsigned int v; asm("ld.global.{op}u16 %0, [%1];" : "=r"(v) : "l"(p)); return (unsigned long long)v; }}')
        if 32 in widths:
            cases.append(f'        case 32: {{ unsigned int v; asm("ld.global.{op}u32 %0, [%1];" : "=r"(v) : "l"(p)); return (unsigned long long)v; }}')
        if 64 in widths:
            cases.append(f'        case 64: {{ unsigned long long v; asm("ld.global.{op}b64 %0, [%1];" : "=l"(v) : "l"(p)); return v; }}')
        body = "\n".join(cases)
        return f"""\
__device__ __forceinline__ unsigned long long load_by_width(const unsigned char* p, unsigned int w)
{{
    switch (w)
    {{
{body}
    }}
    return 0ULL;
}}
"""

    def _emit_cuda(self, p: Dict[str, Any]) -> str:
        reads = p["reads"]
        widths = p["widths"]
        usevec = p["vectors"]
        usestruct = p["structs"]
        nvec = max(1, reads // 8)
        nstruct = max(1, reads // 12)
        wlit = ", ".join(f"{w}u" for w in widths)
        nw = len(widths)
        helper = self._load_helper(p)

        vec_block = ""
        if usevec:
            vec_block = f"""\
    // -- aligned ulonglong2 loads (LDG.E.128) --------------------------------
    for (unsigned int k = 0u; k < {nvec}u; ++k)
    {{
        unsigned int base2 = (((tid * 5u + k * 7u + (unsigned int)(seed >> (k % 64u))) % (input_words / 2u)) * 2u);
        ulonglong2 v = *(const ulonglong2*)(by + (size_t)base2 * 8u);
        acc = acc * {GOLDEN}ULL + (v.x ^ v.y);
    }}
"""
        struct_block = ""
        if usestruct:
            struct_block = f"""\
    // -- struct Pair{{lo,hi}} loads (LDG.E.64) --------------------------------
    for (unsigned int k = 0u; k < {nstruct}u; ++k)
    {{
        unsigned int base3 = (tid * 3u + k * 11u + (unsigned int)(seed >> (32u + k % 32u))) % input_words;
        struct Pair {{ unsigned int lo, hi; }};
        Pair pr = *(const Pair*)(by + (size_t)base3 * 8u);
        acc = acc * {GOLDEN}ULL + (unsigned long long)pr.lo ^ (unsigned long long)pr.hi;
    }}
"""
        return f"""\
// strategy=memory_address  (generated; do not edit)
{helper}
extern "C" __global__ void test_kernel(
    const unsigned long long* __restrict__ input,
    unsigned long long* __restrict__ output,
    unsigned int input_words,
    unsigned long long seed)
{{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned char* by = (const unsigned char*)input;

    static const unsigned int W[{nw}] = {{ {wlit} }};
    unsigned long long acc = seed ^ 0xDEADBEEFCAFEBABEULL;

    // -- scalar multi-level runtime-addressed reads ---------------------------
    for (unsigned int k = 0u; k < {reads}u; ++k)
    {{
        unsigned int w = W[k % {nw}u];
        unsigned int base = (tid * 17u + k * 31u + (unsigned int)(seed >> (k % 64u))) % input_words;
        unsigned int off = base * 8u;
        unsigned int sub;
        if (w == 8u)       sub = (unsigned int)((seed >> ((k + 1u) % 64u)) & 7u);
        else if (w == 16u) sub = 2u * (unsigned int)((seed >> ((k + 2u) % 64u)) & 3u);
        else if (w == 32u) sub = 4u * (unsigned int)((seed >> ((k + 3u) % 64u)) & 1u);
        else               sub = 0u;
        unsigned long long v = load_by_width(by + off + sub, w);
        acc = acc * {GOLDEN}ULL + v;
        acc ^= acc >> 27u;
    }}
{vec_block}
{struct_block}
    // -- local array (spill stress) -------------------------------------------
    unsigned long long local[4];
    for (unsigned int i = 0u; i < 4u; ++i)
    {{
        unsigned int base4 = (tid + i * 3u) % input_words;
        local[i] = ((const unsigned long long*)by)[base4] ^ (seed >> (i * 7u % 64u));
    }}
    unsigned long long lacc = 0ULL;
    for (unsigned int i = 0u; i < 4u; ++i) lacc = lacc * {GOLDEN}ULL + local[i];

    output[tid * 2u] = acc ^ lacc;
    output[tid * 2u + 1u] = (acc * 3ULL) ^ seed ^ (lacc >> 1u);
}}
"""

    def cpu_oracle(self, input_words: int, seed: int, params: Dict[str, Any],
                   input_bytes: bytes, grid: List[int], block: List[int],
                   output_words_per_thread: int) -> bytes:
        p = self._fmt_params(params)
        data = input_bytes
        total = grid[0] * block[0]
        reads = p["reads"]
        widths = p["widths"]
        usevec = p["vectors"]
        usestruct = p["structs"]
        nvec = max(1, reads // 8)
        nstruct = max(1, reads // 12)
        out: List[int] = []
        for tid in range(total):
            acc = (seed ^ 0xDEADBEEFCAFEBABE) & MASK64
            for k in range(reads):
                w = widths[k % len(widths)]
                base = (tid * 17 + k * 31 + ((seed >> (k % 64)) & 0xFFFFFFFF)) % input_words
                off = base * 8
                if w == 8:
                    sub = (seed >> ((k + 1) % 64)) & 7
                    v = data[off + sub]
                elif w == 16:
                    sub = 2 * ((seed >> ((k + 2) % 64)) & 3)
                    v = int.from_bytes(data[off + sub: off + sub + 2], "little")
                elif w == 32:
                    sub = 4 * ((seed >> ((k + 3) % 64)) & 1)
                    v = int.from_bytes(data[off + sub: off + sub + 4], "little")
                else:
                    sub = 0
                    v = int.from_bytes(data[off + sub: off + sub + 8], "little")
                acc = (acc * GOLDEN + v) & MASK64
                acc ^= acc >> 27
            if usevec:
                for k in range(nvec):
                    base2 = (((tid * 5 + k * 7 + ((seed >> (k % 64)) & 0xFFFFFFFF)) % (input_words // 2)) * 2)
                    vx = int.from_bytes(data[base2 * 8: base2 * 8 + 8], "little")
                    vy = int.from_bytes(data[base2 * 8 + 8: base2 * 8 + 16], "little")
                    acc = (acc * GOLDEN + (vx ^ vy)) & MASK64
            if usestruct:
                for k in range(nstruct):
                    base3 = (tid * 3 + k * 11 + (seed >> (32 + k % 32))) % input_words
                    lo = int.from_bytes(data[base3 * 8: base3 * 8 + 4], "little")
                    hi = int.from_bytes(data[base3 * 8 + 4: base3 * 8 + 8], "little")
                    acc = (acc * GOLDEN + lo ^ hi) & MASK64
            local = [0] * 4
            for i in range(4):
                base4 = (tid + i * 3) % input_words
                local[i] = int.from_bytes(data[base4 * 8: base4 * 8 + 8], "little") ^ (seed >> (i * 7 % 64))
            lacc = 0
            for i in range(4):
                lacc = (lacc * GOLDEN + local[i]) & MASK64
            out.append((acc ^ lacc) & MASK64)
            out.append(((acc * 3) ^ seed ^ (lacc >> 1)) & MASK64)
        return pack_words(out)

    def generate(self, config_seed: int, test_seed: int, params: Dict[str, Any],
                 gen_version: str, target_sm: str) -> GeneratedCase:
        p = self._fmt_params(params)
        rng = self._new_rng(config_seed, test_seed)
        input_words = 256
        words = make_input_words(rng, input_words, 8)
        source = self._emit_cuda(p)
        grid, block = [1], [64]
        owpt = 2
        output_words = grid[0] * block[0] * owpt
        seed = rng.getrandbits(64)
        input_bytes = pack_words(words)
        expected = self.cpu_oracle(input_words, seed, p, input_bytes, grid, block, owpt)
        return GeneratedCase(
            test_id=f"memory_address__{config_seed:x}__{test_seed:x}",
            config_seed=config_seed, test_seed=test_seed, strategy=self.name, params=p,
            kernel_seed=seed,
            generator_version=gen_version, source=source,
            input_bytes=input_bytes, expected_bytes=expected,
            grid=grid, block=block, input_words=input_words, output_words=output_words,
            output_words_per_thread=owpt, target_sm=target_sm,
            cpu_oracle=self.name, comparison_policy="exact_u64",
            cuda_features=[target_sm, "ldg_ca_cg_cs"],
        )

    def describe_pressure(self, p: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "widths": p["widths"],
            "reads": p["reads"],
            "cache_policy": p["cache_policy"],
            "vectors": p["vectors"],
        }
