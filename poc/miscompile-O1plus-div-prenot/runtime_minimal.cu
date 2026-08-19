// Reduced from:
//   poc/cuda-11.8_miscompile/generator_ir__38c3f9cb2f485ed7/test.cu
//
// Cross-toolchain result:
//   cuda-10.1 / cuda-11.4, ptxas -O0..-O3 : checksum=65520
//   cuda-11.8,               ptxas -O0     : checksum=65520
//   cuda-11.8,               ptxas -O1..-O3: checksum=3
//
// Source semantics:
//   ~0ULL / ((unsigned long long)__clzll((unsigned long long)blockIdx.x) | 1ULL)
// = 0xffffffffffffffff / 65
// = 0x3f03f03f03f03f0, low byte cast to signed char -> 0xf0 (-16),
// then widened back to unsigned short -> 0xfff0 = 65520.
//
// ptxas 11.8 O1+ selects the 32-bit division fast path even though the
// numerator has nonzero high 32 bits, yielding ~0U / 65 = 0x03f03f03,
// whose low byte is 0x03 -> output 3.
//
#include <stdio.h>
#include <cuda_runtime.h>
#define GPU_CUDA_CHECK(call) do { cudaError_t _gpu_err = (call); if (_gpu_err != cudaSuccess) { fprintf(stderr, "CUDA_ERROR %s:%d: %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(_gpu_err)); return 2; } } while (0)

__global__ void kernel_main(unsigned int* output, int x) {
    output[0] = ((unsigned int)((unsigned short)((signed char)(((~(unsigned long long)x)) / (((unsigned long long)__clzll(((unsigned long long)blockIdx.x))) | 1ull)))));
}

int main(void) {
    unsigned int h_output = 0xdeadbeefu;
    unsigned int *d_output = NULL;
    GPU_CUDA_CHECK(cudaMalloc((void**)&d_output, sizeof(h_output)));
    kernel_main<<<1, 1>>>(d_output, 0);
    GPU_CUDA_CHECK(cudaGetLastError());
    GPU_CUDA_CHECK(cudaDeviceSynchronize());
    GPU_CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(h_output), cudaMemcpyDeviceToHost));
    printf("checksum=%u\n", h_output);
    GPU_CUDA_CHECK(cudaFree(d_output));
    return 0;
}
