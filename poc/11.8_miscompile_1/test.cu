#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>

// Same UB-guarded shift macros as the original generated test.  They do not use
// inline PTX.  All wrapping arithmetic is performed in unsigned types.
#define GPU_SSHL(a, b) ((b) >= 0 && (b) < (long long)(sizeof(a) * 8) ? ((unsigned long long)(a) << (b)) : 0ULL)
#define GPU_SSHR(a, b) ((b) >= 0 && (b) < (long long)(sizeof(a) * 8) ? (a) >> (b) : 0)
#define GPU_CUDA_CHECK(call) do { \
    cudaError_t _gpu_err = (call); \
    if (_gpu_err != cudaSuccess) { \
        fprintf(stderr, "CUDA_ERROR %s:%d: %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(_gpu_err)); \
        return 2; \
    } \
} while (0)

__global__ void kernel_main(unsigned int* output, int n) {
    // v_19 and v_25 are the launch-time values from the original kernel path:
    // original v_19 evaluated to 0 and original v_25 was the unsigned constant
    // below.  This is the original v_28 expression (the first diverging value).
    unsigned int v_19 = 0u;
    unsigned int v_25 = ((unsigned int)(-1299614030));
    unsigned int v_28 = ((unsigned int)(__popcll(((unsigned long long)(((unsigned long long)((GPU_SSHL((n), ((~(v_19)) & 31u)) ^ GPU_SSHR((-(v_25)), ((~(v_19)) & 31u))))))))));

    // The final XOR with (unsigned)n is both the original reduction of the
    // surrounding 32-bit uses of n and the value-anchor that keeps n alive as
    // a 32-bit SSA value.  Without a 32-bit use of n, NVVM emits the correct
    // 64-bit shift on all toolchains and the divergence disappears.
    output[0] = v_28 ^ ((unsigned int)(n));
}

int main(void) {
    unsigned int h_output = 0u;
    unsigned int *d_output = NULL;

    GPU_CUDA_CHECK(cudaMalloc((void**)&d_output, sizeof(unsigned int)));
    kernel_main<<<1, 1>>>(d_output, 1);
    GPU_CUDA_CHECK(cudaGetLastError());
    GPU_CUDA_CHECK(cudaDeviceSynchronize());
    GPU_CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    printf("checksum=%u\n", h_output);

    GPU_CUDA_CHECK(cudaFree(d_output));
    return 0;
}
