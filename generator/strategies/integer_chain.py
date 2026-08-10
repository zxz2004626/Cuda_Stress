"""integer_chain strategy.

A single long loop-carried u32/u64 dependency chain exercising PTX integer ops:
  add/sub/mul, mul.hi.u32/u64, shf (funnel shift), bfe/bfi, prmt, lop3,
  popc/clz/brev, predicates and selects, rotations, runtime input and boundary values.

Everything is unsigned modular arithmetic => fully defined.  All shift amounts, bfe/bfi
positions+widths and byte-permute selectors are construction-constrained to be legal.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..base import GeneratedCase, Strategy
from .helpers import MASK32, MASK64, make_input_words, pack_words, unpack_words

GOLDEN = 0x5851F42D4C957F2D  # nvrandom, odd
MULH = 0x9E3779B97F4A7C15


class IntegerChain(Strategy):
    name = "integer_chain"
    target_opcodes = ["mul.hi", "shf", "bfe", "bfi", "prmt", "lop3", "popc", "clz", "brev"]

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _fmt_params(p: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "chain_len": int(p["chain_len"]),
            "use_mulhi": bool(p["use_mulhi"]),
            "use_funnelshift": bool(p["use_funnelshift"]),
            "use_bfe_bfi": bool(p["use_bfe_bfi"]),
            "use_prmt": bool(p["use_prmt"]),
            "use_lop3": bool(p["use_lop3"]),
            "boundary_words": int(p["boundary_words"]),
        }

    # ------------------------------------------------------------------ CUDA
    def _emit_cuda(self, p: Dict[str, Any]) -> str:
        n = p["chain_len"]
        use_mulhi = p["use_mulhi"]
        use_funnel = p["use_funnelshift"]
        use_bfe = p["use_bfe_bfi"]
        use_prmt = p["use_prmt"]
        use_lop3 = p["use_lop3"]

        # Op mix per iteration index, chosen so the full op set appears.
        lines = []
        if use_mulhi:
            lines.append(
                "if ((j & 1u) == 0u) { a = __umulhi(a, b); }"
                " else { unsigned long long x1 = (unsigned long long)a ^ ((acc * 3u) & 0xFFFFFFFFULL);"
                " unsigned long long x2 = (unsigned long long)b ^ (seed & 0xFFFFFFFFULL);"
                " unsigned long long hi;"
                " asm(\"mul.hi.u64 %0, %1, %2;\" : \"=l\"(hi) : \"l\"(x1), \"l\"(x2));"
                " a = (unsigned int)(hi ^ (hi >> 32u)); }")
        else:
            lines.append("a = a * b + (a ^ b);")
        if use_funnel:
            lines.append("a = __funnelshift_r(a, b, c & 31u);")
        if use_bfe:
            lines.append("{ unsigned int bpos = c & 15u, bwid = 1u + ((c >> 4u) & 15u);")
            lines.append('  asm("bfe.u32 %0, %1, %2, %3;" : "=r"(a) : "r"(a), "r"(bpos), "r"(bwid)); }')
            lines.append("{ unsigned int bpos = (c >> 5u) & 15u, bwid = 1u + ((c >> 9u) & 15u);")
            lines.append('  asm("bfi.b32 %0, %1, %2, %3, %4;" : "=r"(a) : "r"(a), "r"(b), "r"(bpos), "r"(bwid)); }')
        if use_prmt:
            lines.append("a = __byte_perm(a, b, c & 0x77777777u);")
        if use_lop3:
            lines.append('asm volatile("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(a) : "r"(a), "r"(b), "r"(c));')
        lines.append("a = __popc(a ^ b);")
        lines.append("a = __clz(a | 1u);")
        lines.append("a = __brev(a);")
        lines.append("acc = acc * %dULL + (unsigned long long)a;" % GOLDEN)
        lines.append("acc = acc ^ (acc >> 29u);")
        lines.append("if ((j & 3u) == 0u) acc = acc + input[(tid + j * 13u) % input_words] ^ (seed >> (j % 63u));")
        lines.append("{ unsigned int r = j % 64u; if (r != 0u) acc = (acc << r) | (acc >> (64u - r)); }")

        body = "\n    ".join(lines)
        return f"""\
// strategy=integer_chain  (generated; do not edit)
extern "C" __global__ void test_kernel(
    const unsigned long long* __restrict__ input,
    unsigned long long* __restrict__ output,
    unsigned int input_words,
    unsigned long long seed)
{{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long long acc = seed ^ ((unsigned long long)tid * {MULH}ULL) ^ {GOLDEN}ULL;
    for (unsigned int j = 0u; j < {n}u; ++j)
    {{
        unsigned int idx = (tid * 3u + j * 5u) % input_words;
        unsigned long long v = input[idx] ^ (seed + j);
        unsigned int a = (unsigned int)(v >> 32u), b = (unsigned int)(v ^ acc);
        unsigned int c = (unsigned int)(acc >> 32u) ^ (unsigned int)(seed >> (j % 63u));
        {body}
    }}
    output[tid * 2u] = acc;
    output[tid * 2u + 1u] = acc ^ seed ^ input[tid % input_words];
}}
"""

    # ------------------------------------------------------------------ oracle
    def cpu_oracle(self, input_words: int, seed: int, params: Dict[str, Any],
                   input_bytes: bytes, grid: List[int], block: List[int],
                   output_words_per_thread: int) -> bytes:
        p = self._fmt_params(params)
        words = unpack_words(input_bytes)
        n = p["chain_len"]
        out: List[int] = []
        total = (grid[0] * block[0]) if len(grid) >= 1 else 1
        for tid in range(total):
            acc = (seed ^ (tid * MULH) ^ GOLDEN) & MASK64
            for j in range(n):
                idx = (tid * 3 + j * 5) % input_words
                v = (words[idx] ^ (seed + j)) & MASK64
                a = (v >> 32) & MASK32
                b = (v ^ acc) & MASK32
                c = (((acc >> 32) & MASK32) ^ ((seed >> (j % 63)) & MASK32)) & MASK32
                if p["use_mulhi"]:
                    if j % 2 == 0:
                        a = ((a * b) >> 32) & MASK32          # __umulhi
                    else:
                        x1 = (a ^ ((acc * 3) & 0xFFFFFFFF)) & MASK64
                        x2 = (b ^ (seed & 0xFFFFFFFF)) & MASK64
                        hi = (x1 * x2) >> 64                  # mul.hi.u64
                        a = (hi ^ (hi >> 32)) & MASK32
                else:
                    a = (a * b + (a ^ b)) & MASK32
                if p["use_funnelshift"]:
                    # __funnelshift_r => shf.r (wrap): sh==0 yields a (verified on GPU)
                    a = ((a | (b << 32)) >> (c & 31)) & MASK32
                if p["use_bfe_bfi"]:
                    pos, wid = c & 15, 1 + ((c >> 4) & 15)
                    a = (a >> pos) & ((1 << wid) - 1)          # __bfe_u
                    pos, wid = (c >> 5) & 15, 1 + ((c >> 9) & 15)
                    mask = ((1 << wid) - 1) << pos
                    # PTX bfi.b32 d,a,b,pos,wid => ((a << pos) & mask) | (b & ~mask)
                    a = (((a << pos) & mask) | (b & ~mask)) & MASK32
                if p["use_prmt"]:
                    a = self._byte_perm(a, b, c & 0x77777777)
                if p["use_lop3"]:
                    a ^= b ^ c                                  # lop3 0x96 = a^b^c
                    a &= MASK32
                a = self._popc(a ^ b)
                a = self._clz(a | 1)
                a = self._brev(a)
                acc = (acc * GOLDEN + a) & MASK64
                acc = acc ^ (acc >> 29)
                if j % 4 == 0:
                    acc = (acc + words[(tid + j * 13) % input_words] ^ (seed >> (j % 63))) & MASK64
                r = j % 64
                if r:
                    acc = ((acc << r) | (acc >> (64 - r))) & MASK64
            out.append(acc & MASK64)
            out.append((acc ^ seed ^ words[tid % input_words]) & MASK64)
        return pack_words(out)

    @staticmethod
    def _popc(x: int) -> int:
        return bin(x & MASK32).count("1")

    @staticmethod
    def _clz(x: int) -> int:
        x &= MASK32
        if x == 0:
            return 32
        return 32 - x.bit_length()

    @staticmethod
    def _brev(x: int) -> int:
        x &= MASK32
        return int(f"{x:032b}"[::-1], 2)

    @staticmethod
    def _byte_perm(a: int, b: int, c: int) -> int:
        src = [(a >> (8 * k)) & 0xFF for k in range(4)] + [(b >> (8 * k)) & 0xFF for k in range(4)]
        res = 0
        for k in range(4):
            sel = (c >> (4 * k)) & 7
            res |= src[sel] << (8 * k)
        return res & MASK32

    # ------------------------------------------------------------------ entry
    def generate(self, config_seed: int, test_seed: int, params: Dict[str, Any],
                 gen_version: str, target_sm: str) -> GeneratedCase:
        p = self._fmt_params(params)
        rng = self._new_rng(config_seed, test_seed)
        input_words = 256
        words = make_input_words(rng, input_words, p["boundary_words"])
        source = self._emit_cuda(p)
        grid, block = [1], [64]
        owpt = 2
        output_words = grid[0] * block[0] * owpt
        seed = rng.getrandbits(64)
        input_bytes = pack_words(words)
        expected = self.cpu_oracle(input_words, seed, p, input_bytes, grid, block, owpt)
        return GeneratedCase(
            test_id=f"integer_chain__{config_seed:x}__{test_seed:x}",
            config_seed=config_seed, test_seed=test_seed, strategy=self.name, params=p,
            kernel_seed=seed,
            generator_version=gen_version, source=source,
            input_bytes=input_bytes, expected_bytes=expected,
            grid=grid, block=block, input_words=input_words, output_words=output_words,
            output_words_per_thread=owpt, target_sm=target_sm,
            cpu_oracle=self.name, comparison_policy="exact_u64",
            cuda_features=[target_sm, "sm_80_intrinsics"],
        )

    def describe_pressure(self, p: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "ops": ["mul.hi", "shf", "bfe", "bfi", "prmt", "lop3", "popc", "clz", "brev",
                    "rotate", "modular_add_mul"],
            "loop_carried_len": p["chain_len"],
        }
