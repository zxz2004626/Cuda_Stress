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
