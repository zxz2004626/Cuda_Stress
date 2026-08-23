# miscompile-O1plus-csel-u16

Self-contained CUDA-C reproducer for a ptxas miscompile in CUDA 13.1+ on `sm_86`:
an `unsigned short` conditional-select loses the `(unsigned short)` truncation on
the selected arm when compiled with `-Xptxas=-O1` or higher.

## Files

| file | description |
| --- | --- |
| `test.cu` | Minimal reproducer (37 lines). |

## Usage

```bash
cd /export/ssd1/xzhanglb/Cuda_Stress/poc/miscompile-O1plus-csel-u16

make clean && make

# or directly
NVCC=/export/ssd1/xzhanglb/nvcc-toolkit/cuda-13.3/bin/nvcc
$NVCC -std=c++17 -arch=sm_86 -w -Xptxas=-O0 test.cu -o tiny_O0
$NVCC -std=c++17 -arch=sm_86 -w -Xptxas=-O1 test.cu -o tiny_O1
```

## Expected Result

`test.cu` writes one checksum (`output[2]`) from thread 2.

For `tid == 2`:

- `warp_0 = tid / 32 == 0`
- the condition `f < warp_0` is always false (`f` is unsigned)
- therefore `c` must be `(unsigned short)s`
- `s = input[2] = 2 * 2654435761u + 12345u == 1013916571`
- `(unsigned short)1013916571 == 9115`

So the source-correct checksum is `9115`.

| ptxas | status | checksum |
| --- | --- | --- |
| O0 | CORRECT | 9115 |
| O1 | WRONG | 1013916571 |
| O2 | WRONG | 1013916571 |
| O3 | WRONG | 1013916571 |

## Version Matrix (sm_86, same test)

| CUDA | nvcc build | O0 | O1/O2/O3 |
| --- | --- | --- | --- |
| 12.5 | 12.5.r12.5/compiler.34177558_0 | 9115 | 9115 |
| 12.6 | 12.6.r12.6/compiler.35059454_0 | 9115 | 9115 |
| 13.0 | 13.0.r13.0/compiler.36260728_0 | 9115 | 9115 |
| 13.0 update | 13.0.r13.0/compiler.36424714_0 | 9115 | 9115 |
| 13.1 | 13.1.r13.1/compiler.36836380_0 | 9115 | **1013916571** |
| 13.2 | 13.2.r13.2/compiler.37434383_0 | 9115 | **1013916571** |
| 13.3 | 13.3.r13.3/compiler.37862127_0 | 9115 | **1013916571** |

The miscompile is introduced in CUDA 13.1 and persists through 13.3.

## Root Cause: ptxas drops the u16 truncation of the false arm

### Source

```cuda
unsigned short c =
    ((f < (t / 32u))
         ? ((unsigned short)((signed char)s))
         : ((unsigned short)s));
```

### PTX is correct

The front-end keeps the select in 16-bit form (`selp.b16`) and then
zero-extends it (`cvt.u32.u16`):

```text
cvt.u16.u32  %rs1, %r63;       // false arm: (unsigned short)s
cvt.s16.s8   %rs2, %rs1;       // true arm
setp.lt.u32  %p21, %r125, %r6;
selp.b16     %rs3, %rs2, %rs1, %p21;
cvt.u32.u16  %r125, %rs3;
```

`cvt.u16.u32` is the truncation that the source requires. The bug is therefore
introduced by ptxas, not by the CUDA-C/C++ front-end.

### O0 SASS: truncation is present

`-Xptxas=-O0` keeps the 32-bit select followed by a final `PRMT` that extracts
the low 16 bits before the store:

```text
ISETP.LT.U32.AND P1, PT, R28, R10, PT
SEL R24, R24, R30, P1
PRMT R24, R24, 0x7710, RZ      // final u16 truncation
MOV R28, R24
STG.E [R24.64], R28
```

### O1 SASS: false-arm truncation is predicated away

`-Xptxas=-O1` if-converts the select into predicated `PRMT`s. The truncation
`PRMT` only executes on the true arm (`@!P0`), while the false arm falls through
with the full 32-bit load value:

```text
ISETP.GE.U32.AND P0, PT, R17, R4, PT    // P0 = (f >= warp_0), i.e. false arm
ISETP.NE.AND P1, PT, R0, 0x2, PT        // P1 = (tid != 2)
@!P0 PRMT R16, R20, 0x8880, RZ          // true-arm conversion only
@!P0 PRMT R20, R16, 0x7710, RZ
IMAD.MOV.U32 R17, RZ, RZ, R20           // unpredicated: takes full s on false arm
@!P1 STG.E [R18.64+-0x4], R17
```

For `tid == 2`, `warp_0 == 0` and `f >= 0`, so `P0` is true and both `PRMT`s
are skipped. `R20` still holds the full 32-bit `s = 1013916571`, the unpredicated
`IMAD.MOV.U32` passes it to `R17`, and the store writes `1013916571` instead of
the truncated `9115`.

This is the smoking gun: ptxas O1 widened a 16-bit `selp` to a 32-bit predicated
move and only attached the narrowing conversion to the true-arm side.

## Minimal Trigger Conditions

All of the following are required to reproduce the O0/O1 difference on CUDA
13.3 / sm_86:

- The guarded modulo macro must have this shape:
  `#define S(a,b) (((a) < 0x80000000u) ? (a) % (b) : 0)`.
  Plain `%`, `& (b-1)`, or `(a) != 0x80000000u` forms make O1 correct again.
- The outer loop must run 2 iterations. A single iteration does not trigger.
- The 12-load dead XOR chain (`p`, `rr[12]`, `o[3]`) is semantically irrelevant
  but must be present; deleting it or shortening it below 12 loads makes O1
  correct. It perturbs ptxas register allocation/scheduling.
- The guard must be the data-dependent
  `if (i[S(t*4u+2u,n)] < i[S(t*4u,n)])`; replacing it with `if (t == 2u)`
  makes O1 correct.

These constraints suggest the bug only shows up in a particular ptxas
if-conversion / register-allocation state.

## UB Check

In `test.cu`, `grid = 1`, `block = 256`, so `t` is in `[0, 255]`.
All `S()` index expressions are provably less than `0x80000000u` and are
wrapped by `% n`; the input allocation is 256 words. Only `tid == 2` writes
`o[2]`; the `o[3]` store is guarded by `t == 9999u` and is never executed.

CUDA 13.3 `compute-sanitizer` reports 0 errors on both binaries:

| binary | memcheck | initcheck |
| --- | --- | --- |
| O0 | 0 errors | 0 errors |
| O1 | 0 errors | 0 errors |
