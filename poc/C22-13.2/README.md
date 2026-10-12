# C22: cicc misfold — the modulo in `(ull)__mul24(x,x) % C` is deleted (CUDA 11.5–13.2 wrong, 13.3 conforms to source semantics; the archived correct/wrong labels are reversed, see "Verdict correction" below)

Self-contained, stable, UB-free, self-judging reduced reproducer for
trigger-matrix class **C22** of the Cuda_Stress differential run.

> Note: `test.cu` in this directory was rewritten on 2026-09-21 as a
> statement-level minimization of the confirmed root cause (see below). It is
> self-judging: it prints both the observed value and a CORRECT/BUG verdict.
> The archived `checksums:` values in `metadata.yaml` (7347577240743941561 /
> 8398921093249958953) refer to the pre-reduction, checksum-based `test.cu`
> and do not match the current file.

## What the reproducer does

```cuda
__global__ void kernel_main(unsigned int* out, int x) {
    out[0] = (unsigned int)(((unsigned long long)__mul24(x, x)) % 4986325391324574397ull);
}
// launch: kernel_main<<<1, 1>>>(d, 6428759);
```

With `x = 6428759`, `__mul24(x, x)` returns the `int` `0xA4EC6D91`
(`-1528009327` — the low 32 bits of the exact product). ISO C++ requires
`(unsigned long long)(int)` to sign-extend, so the divisor exceeds the
dividend and the modulo is significant:

1. sign-extend: `(unsigned long long)(-1528009327)` = `18446744072181542289`
2. `% 4986325391324574397` = `3487767898207819098` = `0x30670A6CA7B0155A`
3. truncate to `unsigned int` → **`0xA7B0155A`** (source-semantics answer)

The buggy frontends (CUDA 11.5–13.2) model the `__mul24` result as the exact
mathematical product (non-negative, smaller than the divisor), conclude the
modulo is an identity, delete it, and store the raw low 32 bits:
**`0xA4EC6D91`**.

## Build & run

```bash
cd /export/ssd1/xzhanglb/llm-template/Cuda_Stress/poc/C22
make            # builds test.cu with CUDA 13.3 ptxas -O0/-O1/-O2, prints each result
make clean

# show the miscompiled side (any CUDA 11.5–13.2 frontend):
make NVCC=/export/ssd1/xzhanglb/nvcc-toolkit/cuda-13.2/bin/nvcc
```

Each build prints two lines; the `checksum=` line is the one the Makefile
greps:

```text
checksum=a7b0155a
gpu=a7b0155a  correct=a7b0155a  buggy=a4ec6d91  -> CORRECT
```

## Expected results (measured on RTX 3090, sm_86, all local toolchains × ptxas -O0/-O1/-O2/-O3)

| CUDA toolkit | result | verdict |
| --- | --- | --- |
| 11.5 / 11.8 / 12.2 / 12.5 / 12.8 / 13.0 / 13.1 / 13.2 | `0xa4ec6d91` | **BUG** (all -O levels) |
| 13.3 | `0xa7b0155a` | CORRECT (all -O levels) |
| any toolkit with `-G` | `0xa7b0155a` | CORRECT |

(CUDA 11.1 cannot target sm_86 and was not testable.)

## Trigger conditions (all required, verified by statement-level ablation)

1. `x` must be a **runtime** kernel parameter — a literal `6428759` in the
   kernel body is constant-folded correctly (13.2 does not trigger either);
2. `x` must make `__mul24(x, x)` negative after truncation to `int`
   (`(x*x) & 0x80000000 != 0`; for `x = 6428759` the product's low 32 bits are
   `0xA4EC6D91`);
3. the result must be truncated through `(unsigned int)` on store (a variant
   that stores the `unsigned long long` directly looks correct even on 13.2).

## PTX evidence (12.6 vs 13.3, `-O3 -ptx`)

12.6 (buggy) — both the `(u64)(int)` sign extension and the entire modulo are
gone:

```text
mul.lo.s32  %r4, %r3, %r3;
st.global.u32 ...
```

13.3 (correct) — sign extension plus magic-number modulo are present:

```text
mul.lo.s32  %r4, %r3, %r3;
cvt.s64.s32 %rd3, %r4;
mul.hi.u64  %rd4, %rd3, -1385965834438579005;
mul.lo.s64  %rd6, %rd5, 4986325391324574397;
...
```

The PTX is identical across `-Xptxas` levels, and `-G` is always correct, so
the defect is in the cicc/NVVM optimizer, not in ptxas.

## Root cause (mechanism)

cicc/NVVM models the `int` result of `__mul24` as the exact mathematical
product (hence non-negative and `< 2^46`, far below the divisor), which makes
`(unsigned long long)__mul24(x,x) % C` an identity that gets eliminated. ISO
C++ instead requires the `int` value to be sign-extended to `unsigned long
long` first, after which the modulo is meaningful. The correct value is
`0xa7b0155a`; the miscompiled value is `0xa4ec6d91`. An explicit
`(unsigned long long)(long long)__mul24(...)` cast does not prevent the fold.

## Verdict correction (important)

The archived trigger matrix lists `13.3/O0-O1-O2` as the deviating column and
`metadata.yaml` records `correct: 7347577240743941561` /
`wrong: 8398921093249958953` — labels derived from a majority-vote oracle
over the pre-reduction checksum-based `test.cu`. Under ISO C++ semantics (and
per nvcc's own `-G` path) the orientation is reversed: the 13.3 frontend
emits correct PTX, while the 11.5–13.2 frontends miscompile the expression.
The current self-judging `test.cu` therefore prints CORRECT on 13.3.

## Files

| file | description |
| --- | --- |
| `test.cu` | Self-judging minimal reproducer; prints `checksum=<hex>` plus a human-readable CORRECT/BUG verdict line. |
| `Makefile` | Builds `test.cu` with CUDA 13.3 ptxas -O0/-O1/-O2 and prints each level's checksum. |
| `metadata.yaml` | Archived machine-readable summary (labels follow the majority-vote oracle; see "Verdict correction"). |

## Status

Unfixed for CUDA 11.5–13.2 frontends as of CUDA 13.3 (13.3's frontend emits
the correct lowering).
