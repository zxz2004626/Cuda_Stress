# 10.8_misconpile_0

Self-contained CUDA-C reproducer reduced from:

`poc/cuda-11.8_miscompile/generator_ir__1df8dd7d23ccc883/test.cu`

It does not use inline PTX and does not depend on the `cuda_stress` Python package or other repository code. The scripts only assume local CUDA toolkit installs.

## Files

- `runtime_minimal_no_inline.cu`: reduced CUDA source.
- `build_all.sh`: builds CUDA 10.1 / 11.4 / 11.8 across ptxas `-O0..-O3`, and dumps SASS beside each binary.
- `run_all.sh`: builds by default, then runs the 12 binaries with a timeout and prints a matrix.

Default toolchain paths:

- `CUDA_10_1=/export/d3/xzhanglb/nvcc-toolkit/cuda-10-1`
- `CUDA_11_4=/export/d3/xzhanglb/nvcc-toolkit/cuda-11-4`
- `CUDA_11_8=/export/d3/xzhanglb/nvcc-toolkit/cuda-11-8`
- `GXX8=/export/d3/xzhanglb/nvcc-toolkit/gcc-8/usr/bin/g++-8`

Override them through environment variables if needed. The default target architecture is `sm_75`; override with `ARCH=...`.

## Usage

```bash
cd /export/d3/xzhanglb/Cuda_Stress/poc/10.8_misconpile_0
./build_all.sh
SKIP_BUILD=1 ./run_all.sh
```

or simply:

```bash
./run_all.sh
```

## Reduction notes

The minimized trigger is CUDA source only:

- one block and one thread;
- one 32-bit output word, no input buffer;
- two `helper_2` calls, where the second return value feeds the final value;
- two `helper_1` calls, one conditional and one unconditional;
- fence-heavy small loops using CUDA C intrinsics only;
- final result `v_3 ^ (v_24 << 6) ^ v_10`.

The source avoids inline PTX. Arithmetic that may wrap is performed in unsigned types.

## Observed result on this host

Timeout is set to 5 seconds.

| compiler | O0 | O1 | O2 | O3 |
| --- | --- | --- | --- | --- |
| cuda-10.1 | OK checksum=5850102576 | OK checksum=5850102576 | OK checksum=5850102576 | OK checksum=5850102576 |
| cuda-11.4 | OK checksum=5850102576 | OK checksum=5850102576 | OK checksum=5850102576 | OK checksum=5850102576 |
| cuda-11.8 | OK checksum=5850102576 | OK checksum=5850102547 | OK checksum=5850102576 | OK checksum=5850102576 |

The reduced case preserves the original divergence shape: only CUDA 11.8 ptxas `-O1` produces a different runtime result.

## Root cause from SASS

This is a CUDA 11.8 ptxas `-O1` backend miscompile. The CUDA front-end PTX is not the cause: `cuobjdump --dump-ptx` shows that the CUDA 11.8 `-O1` and `-O2` binaries contain identical PTX except for the recorded `ptxasOptions` string.

The wrong value is in `v_10`, not in the final store or host-side checksum. Source-level expected values for the one-thread launch are:

```text
v3        = 0x56565655
v24 << 6  = 0xe82fa080
v10       = ffsll(first) ^ ffsll(second)
          = 1 ^ 3
          = 0x2
output    = 0xbe79f6d7
```

CUDA 11.8 `-O1` writes `0xbe79f6f6`. If `v3` and `v24` are correct, this implies that `v_10` was computed as `0x23` instead of `0x2`.

The faulty subexpression is the second `__ffsll` input:

```cuda
((unsigned long long)(-173619356)) *
(((unsigned long long)gridDim.x) + 2398593718343027946ull)
```

PTX lowers `__ffsll(x)` through the standard `x & -x`, `clz.b64`, `64 - clz` sequence. The correct intermediate for the second `__ffsll` is:

```text
second_arg       = 0xdc6a5ef8f5a0b8cc
second_arg & -x  = 0x0000000000000004
ffsll(second)    = 3
```

In the correct CUDA 11.8 `-O2` SASS, ptxas computes the high 32-bit correction for the negative 64-bit multiply using `-gridDim.x`:

```sass
/*03c0*/ IMAD.MOV R7, RZ, RZ, -c[0x0][0xc] ;
/*03e0*/ IMAD.WIDE.U32 R2, R0, -0xa59389c, RZ ;
/*0490*/ IADD3.X R5, R3, -0x2395a108, R7, P0, !PT ;
```

In the faulty CUDA 11.8 `-O1` SASS, ptxas uniformizes the computation into `UR` registers but uses positive `gridDim.x` for that same high-half correction:

```sass
/*0400*/ ULDC UR8, c[0x0][0xc] ;
/*0430*/ UIMAD.WIDE.U32 UR4, UR8, -0xa59389c, URZ ;
/*04c0*/ UIADD3.X UR4, UR5, -0x2395a108, UR8, UP0, !UPT ;
```

Here `UR8` is `gridDim.x`; it should be `-gridDim.x`, as in the `-O2` code. This changes the high 32 bits of the `x & -x` input from:

```text
correct high32: 0xdc6a5ef8
faulty high32:  0xdc6a5efa
```

The faulty high word is non-zero after the following AND:

```text
correct: 0xdc6a5ef8 & 0x2395a107 = 0x00000000
faulty:  0xdc6a5efa & 0x2395a107 = 0x00000002
```

Therefore `clz.b64` sees a high 32-bit bit set, so `ffsll(second)` becomes `34` instead of `3`:

```text
correct v10 = 3  ^ 1 = 0x02
faulty v10  = 34 ^ 1 = 0x23
```

That exactly explains the observed output difference:

```text
0xbe79f6d7 ^ 0x02 ^ 0x23 = 0xbe79f6f6
```

Root cause summary: CUDA 11.8 ptxas `-O1` miscompiles a negative-constant 64-bit multiply when lowering/optimizing it through the uniform-register path. The high-half sign correction uses `+gridDim.x` where it should use `-gridDim.x`; this corrupts the `x & -x` value used to implement `__ffsll`, producing the wrong final result.
