#include <stdio.h>
#include <cuda_runtime.h>

#define GPU_CUDA_CHECK(call) do { \
    cudaError_t _gpu_err = (call); \
    if (_gpu_err != cudaSuccess) { \
        fprintf(stderr, "CUDA_ERROR %s:%d: %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(_gpu_err)); \
        return 2; \
    } \
} while (0)

__forceinline__ __device__ void helper_1() {
    unsigned int p0 = 0x61c88647u;
    for (int i_3 = 0; i_3 < ((((int)(~(blockDim.x | p0))) & 7) + 1); i_3 += 1) {
        for (int i_4 = 0; i_4 < ((((int)(__umulhi(p0, (unsigned int)__ffs(p0)))) & 7) + 1); i_4 += 1) {
            __threadfence_block();
        }
    }
}

__device__ unsigned int helper_2(unsigned int p1, unsigned int p3) {
#pragma unroll
    for (int i_3 = 0; i_3 < ((((int)(((unsigned int)__brevll(0ull)) ^ ((unsigned int)__clz(p1)))) & 7) + 1); i_3 += 1) {
        __threadfence();
    }
    return (~p3) & 0x2fa0be82u;
}

extern "C" __global__ void kernel_main(unsigned int* __restrict__ output, int var_1, unsigned int var_2) {
    unsigned int v_3 = __byte_perm(
        blockDim.x - (2863311530u ^ 1u),
        (unsigned int)(((unsigned long long)((unsigned int)var_1)) * ((unsigned long long)blockDim.x)),
        2u);

    unsigned int v_10 =
        ((unsigned int)__ffsll((unsigned long long)
            ((10965215011232021246ull - 6547211081039364433ull) +
             ((unsigned long long)__byte_perm(gridDim.x, var_2, 2u))))) ^
        ((unsigned int)__ffsll((unsigned long long)
            (((unsigned long long)(-173619356)) *
             (((unsigned long long)gridDim.x) + 2398593718343027946ull))));

    (void)helper_2(var_2, blockIdx.x);
    if (v_3 != threadIdx.x) {
        helper_1();
    }

    unsigned int v_24 = helper_2(1431655765u, threadIdx.x);
    helper_1();
    __threadfence_block();

    v_3 ^= v_24 << 6;
    output[0] = v_3 ^ v_10;
}

int main(void) {
    unsigned int h_output = 0;
    unsigned int *d_output = NULL;
    GPU_CUDA_CHECK(cudaMalloc((void**)&d_output, sizeof(unsigned int)));

    kernel_main<<<dim3(1), dim3(1)>>>(d_output, 1218333469, 2619514586u);
    GPU_CUDA_CHECK(cudaGetLastError());
    GPU_CUDA_CHECK(cudaDeviceSynchronize());
    GPU_CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    unsigned long long hash = 42;
    unsigned long long val = h_output;
    hash ^= val + 0x9e3779b9ULL + (hash << 6) + (hash >> 2);
    printf("checksum=%llu\n", hash);

    GPU_CUDA_CHECK(cudaFree(d_output));
    return 0;
}
