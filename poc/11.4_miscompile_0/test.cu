#include <stdio.h>
#include <cuda_runtime.h>

#define GPU_SSHL(a, b) ((b) < 32 ? ((unsigned long long)(a) << (b)) : 0ULL)

__global__ void kernel_main(unsigned int* output, int var_1) {
    unsigned int v_5 = (-((GPU_SSHL(0, (blockIdx.x + 10)) ^ (1 >> ((blockIdx.x + 10) & 31)))));
    if (threadIdx.x == 0) output[0] = ((((unsigned long long)(var_1) << 31) ^ (v_5 >> 31)) >> ((blockIdx.x + 1) & 63));
}

int main(void) {
    unsigned int out = 0, *d_out = NULL;
    cudaMalloc((void**)&d_out, sizeof(out));
    kernel_main<<<1, 1>>>(d_out, 1);
    cudaDeviceSynchronize();
    cudaMemcpy(&out, d_out, sizeof(out), cudaMemcpyDeviceToHost);
    printf("checksum=%u\n", out);
    cudaFree(d_out);
    return 0;
}
