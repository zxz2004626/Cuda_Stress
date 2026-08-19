# CUDA 13.0/12.5 optimized ptxas differential reduced testcase

This directory contains a reduced, self-contained CUDA testcase for the differential originally found at:

`/export/ssd1/xzhanglb/Cuda_Stress/runs/run-20260818-105706-0f72/differentials/generator_ir__e957aeccec983129/test.cu`

Files:

- `test.cu`: reduced testcase, no inline PTX.
- `run.sh`: builds and runs the testcase with CUDA 13.3, 13.0, and 12.5 at `-Xptxas -O0/-O1/-O2`.
- `asm/`: PTX/SASS artifacts generated with `nvcc -keep` and `cuobjdump -sass`.

## Reproduce

```sh
cd /export/ssd1/xzhanglb/Cuda_Stress/poc/13.0-O0_12.5-O12_miscompile_0
./run.sh
```

Observed output:

```text
compiler opt checksum
cuda-13.3 O0  65407
cuda-13.3 O1  65407
cuda-13.3 O2  65407
cuda-13.0 O0  65407
cuda-13.0 O1  127
cuda-13.0 O2  127
cuda-12.5 O0  65407
cuda-12.5 O1  127
cuda-12.5 O2  127
```

## Reduced Program

```cuda
#include <stdio.h>

__global__ void k(unsigned *o) {
    if (!blockIdx.x) {
        *o = (unsigned short)(signed char)-(int)gridDim.x - 1;
    }
}

int main() {
    unsigned *o;

    cudaMallocManaged(&o, 4);
    k<<<128, 1>>>(o);
    cudaDeviceSynchronize();

    printf("checksum=%u\n", *o);
    return 0;
}
```

Only block 0 writes to `*o`, so the testcase does not rely on an inter-block write race. With `k<<<128, 1>>>`, `gridDim.x == 128`.

The source-level value is:

```text
-(int)gridDim.x              = -128
(signed char)-128            = -128
(unsigned short)-128         = 65408
(unsigned short)-128 - 1     = 65407
```

So the correct checksum is `65407`. CUDA 13.3 is the correct compiler version in this comparison. More precisely, CUDA 13.3 `-Xptxas -O0/-O1/-O2` all match the source semantics; CUDA 13.0 and CUDA 12.5 also match at `-Xptxas -O0`, but miscompile at `-Xptxas -O1/-O2`.

## PTX Evidence

The key PTX sequence is the same across the checked compiler versions, apart from metadata such as PTX version and compiler build ID. For example, `asm/cuda-13.3_O1/test.ptx` contains:

```ptx
mov.u32        %r2, %nctaid.x;
cvt.u16.u32    %rs1, %r2;
shl.b16        %rs2, %rs1, 8;
neg.s16        %rs3, %rs2;
shr.s16        %rs4, %rs3, 8;
cvt.u32.u16    %r3, %rs4;
add.s32        %r4, %r3, -1;
st.global.u32  [%rd2], %r4;
```

This PTX implements the intended narrowing/sign-extension pattern. For `gridDim.x == 128`, it produces `0xff7f == 65407`. Therefore the differential is not caused by CUDA front-end source-to-PTX lowering; it appears after PTX, during `ptxas` lowering/optimization to SASS.

## SASS Evidence

The optimized SASS differs at the point where the 16-bit value must be re-normalized after negation and before the arithmetic right shift.

Correct CUDA 13.3 `-Xptxas -O1`, from `asm/cuda-13.3_O1/kernel.sass`:

```sass
LDC.U16 R0, c[0x0][0xc]
IMAD.SHL.U32 R0, R0, 0x100, RZ
PRMT R0, R0, 0x9910, RZ
IMAD.MOV R0, RZ, RZ, -R0
PRMT R0, R0, 0x9910, RZ
SHF.R.S32.HI R0, RZ, 0x8, R0
LOP3.LUT R0, R0, 0xffff, RZ, 0xc0, !PT
IADD3 R5, R0, -0x1, RZ
STG.E [R2.64], R5
```

Wrong CUDA 13.0 `-Xptxas -O1`, from `asm/cuda-13.0_O1/kernel.sass`:

```sass
LDC.U16 R0, c[0x0][0xc]
IMAD.SHL.U32 R0, R0, 0x100, RZ
PRMT R0, R0, 0x9910, RZ
IMAD.MOV R0, RZ, RZ, -R0
SHF.R.S32.HI R0, RZ, 0x8, R0
LOP3.LUT R0, R0, 0xffff, RZ, 0xc0, !PT
IADD3 R5, R0, -0x1, RZ
STG.E [R2.64], R5
```

The second `PRMT R0, R0, 0x9910, RZ` is present in CUDA 13.3 but missing in CUDA 13.0/12.5 optimized SASS. In this generated sequence, that `PRMT` re-establishes the signed 16-bit value in the 32-bit register before `SHF.R.S32.HI` performs the arithmetic right shift.

For this input:

- After the first shift/sign-extension sequence, the value being negated corresponds to `0xffff8000`.
- After negation, the low 16-bit value is `0x8000`.
- Correct lowering sign-extends that low 16-bit value again to `0xffff8000` before shifting right by 8, giving `0xffffff80`, which becomes `0xff80`, then `0xff7f == 65407` after subtracting 1.
- The buggy optimized lowering skips that re-sign-extension step, shifts `0x00008000` right by 8, gets `0x00000080`, then subtracts 1 and stores `127`.

## Root Cause

The root cause is an optimized `ptxas` SASS-lowering bug in CUDA 12.5/13.0 at `-Xptxas -O1/-O2` for this `cvt.u16.u32; shl.b16; neg.s16; shr.s16; cvt.u32.u16` pattern. The buggy lowering drops the post-negation 16-bit sign-extension/truncation operation before the arithmetic shift, changing the signed 16-bit semantics encoded in PTX.

CUDA 13.3 preserves the required operation in optimized SASS, matches `-Xptxas -O0`, and produces the source-correct result `65407`.
