# CUDA 12.5/12.6 cicc front-end miscompile: multiply moved across char narrowing

Front-end (`cicc`/NVVM optimizer) differential originally found at:

`/export/ssd1/xzhanglb/Cuda_Stress/runs/run-20260829-011020-1bbc/differentials/generator_ir__350e41ea3b05302b/test.cu`

This is **not** a ptxas bug: the buggy PTX alone, JITed by the host driver
(`nvcc -arch=compute_86`), already produces the wrong output, and `nvcc -G`
(optimized passes off) is correct on the affected toolchains.

## Reproduce

```sh
cd /export/ssd1/xzhanglb/Cuda_Stress/poc/miscompile-all-cicc-mulnarrow0
./run.sh
```

Observed output (RTX 3090, sm_86):

```text
compiler opt checksum
cuda-13.3 O0  0
cuda-13.3 O1  0
cuda-13.3 O2  0
cuda-13.2 O0  0
cuda-13.2 O1  0
cuda-13.2 O2  0
cuda-13.1 O0  0
cuda-13.1 O1  0
cuda-13.1 O2  0
cuda-13.0 O0  0
cuda-13.0 O1  0
cuda-13.0 O2  0
cuda-12.6 O0  1
cuda-12.6 O1  1
cuda-12.6 O2  1
cuda-12.5 O0  1
cuda-12.5 O1  1
cuda-12.5 O2  1
```

12.6 is the **last bad** version; the bug is gone in 12.8 and later (CUDA
12.7 does not exist — NVIDIA skipped it). The bug is independent of the
ptxas optimization level (it is already present in the PTX).

Docker verification of the gap versions (images pulled then removed):

```text
cuda-12.8.2 O0  0    (nvidia/cuda:12.8.2-devel-ubuntu22.04)
cuda-12.8.2 O2  0
cuda-12.9.2 O0  0    (nvidia/cuda:12.9.2-devel-ubuntu22.04)
cuda-12.9.2 O2  0
```

(static-cudart binaries built inside the containers, executed on the host
under driver 590.48.01).

## Reduced program (12 lines, hand-golfed)

```cuda
#include<cstdio>
__managed__ int m=1;
__global__ void k(int*i){
    auto f=threadIdx.y,t=blockIdx.x,a=t|1,b=t|3,c=t+5,d=t+7,e=t|5;
    char x=*i-129;__ldg(i);
    *i?f^=3*x:f^=__umul24(t,x>-2?1:x);
    printf("checksum=%u",e/c^a/d^f%b);
}

main(){
    k<<<1,1>>>(&m);
    cudaDeviceReset();
}
```

Notes:

- `f` is `unsigned int` (deduced from `threadIdx.y`).
- The dead `__ldg(i);` is required: deleting it makes 12.5/12.6 correct.
- With `*i == 1`, the true branch runs: `x = (char)(1-129) = -128`,
  `f ^= 3*x = -384`, i.e. `f = 0xFFFFFE80u`.
- Correct value: `e/c ^ a/d ^ f%b = 5/5 ^ 1/7 ^ (0xFFFFFE80u % 3u) = 1^0^1 = 0`.

## PTX evidence

12.5/12.6 lower the true branch as (`asm/cuda-12.5.ptx`, `asm/cuda-12.6.ptx`):

```ptx
ld.global.s8   %rs1, [%rd2];        // loads *i (=1), not x
cvt.s32.s16    %r6, %rs1;
mad.lo.s32     %r26, %r6, 3, 381;   // 3*(*i) + 381 = 384
```

13.0+ lower it with the truncation intact, producing `-384`.

cicc rewrote `3*(char)(c-129)` into `3*c + 381` — it distributed the
multiply across the narrowing char conversion (`3*(char)v -> 3*v`), skipping
the intermediate truncation. The value changes by exactly `768 = 3*256`, so
the rewrite is congruence-preserving modulo 3; the pass evidently assumes
the value's only consumer is `% 3`.

## Why that assumption is wrong

The consumer is `rem.u32` on `f` (unsigned). Signed congruence mod 3 does not
imply equal unsigned remainders of the two's-complement bit patterns:

- correct `f = 0xFFFFFE80u` → `rem.u32 3` = **1**
- buggy   `f = 0x00000180u` → `rem.u32 3` = **0**

Hence the buggy checksum `1 ^ 0 ^ 0 = 1` instead of `1 ^ 0 ^ 1 = 0`.

## Root cause

A cicc/NVVM optimizer strength-reduction pass in CUDA 12.5/12.6 moves a
multiply across a narrowing `char` conversion under a modulo-congruence
justification that is only valid for signed (or non-negative) semantics; the
actual consumer is an unsigned remainder. Fixed in 13.0. Versions before
12.5 are not installed on this host and remain unverified.
