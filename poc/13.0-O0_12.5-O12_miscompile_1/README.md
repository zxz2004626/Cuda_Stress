# CUDA 13.3 differential reduced testcase

Original case:

`/export/ssd1/xzhanglb/Cuda_Stress/runs/run-20260819-000123-2433/differentials/generator_ir__37afd2c4c9ac8dd2/test.cu`

This directory contains a reduced, self-contained testcase. It has no inline PTX and does not require any input file.

## Reproduce

```sh
cd /export/ssd1/xzhanglb/Cuda_Stress/poc/13.3_miscompile_1
./run.sh
```

Observed result:

```text
compiler opt checksum
cuda-13.3 O0  4294967168
cuda-13.3 O1  4294967168
cuda-13.3 O2  4294967168
cuda-13.0 O0  4294967168
cuda-13.0 O1  128
cuda-13.0 O2  128
cuda-12.5 O0  4294967168
cuda-12.5 O1  128
cuda-12.5 O2  128
```

## Reduced Program

```cuda
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void kernel_main(unsigned int *output) {
    int x = -(int)blockIdx.x;
    signed char c = (signed char)x;
    short s = (short)c;

    output[blockIdx.x] = (unsigned int)s;
}

int main() {
    unsigned int *output;

    cudaMallocManaged(&output, 129 * sizeof(*output));
    kernel_main<<<129, 1>>>(output);
    cudaDeviceSynchronize();

    printf("checksum=%u\n", output[128]);
    cudaFree(output);
    return 0;
}
```

For `blockIdx.x == 128`, the source semantics are:

```text
x                         = -128
(signed char)x            = -128
(short)(signed char)x     = -128
(unsigned int)-128        = 4294967168
```

So the correct result is `4294967168`. CUDA 13.3 `-Xptxas -O0/-O1/-O2`, CUDA 13.0 `-O0`, and CUDA 12.5 `-O0` agree with this value. CUDA 13.0/12.5 optimized `ptxas` at `-O1/-O2` produces `128`.

## Notes

The original differential's full checksum split was preserved by first reducing the output to the original `v_1` value:

```cuda
unsigned int v_1 = (unsigned int)(short)(signed char)(-tid);
```

The final testcase then reduced the launch to the first boundary where the bad value appears: `129` single-thread blocks and `output[128]`. Launching only up to index `127` does not trigger a difference; index `128` is the first value where the sign bit of the narrowed `signed char` is set.

The `asm/` directory contains PTX/SASS generated with `nvcc -keep` and `cuobjdump -sass` for the checked compiler/optimization combinations.
