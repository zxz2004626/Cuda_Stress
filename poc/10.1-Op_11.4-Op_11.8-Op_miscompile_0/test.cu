#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) return 2; } while (0)

__global__ void kernel_main(unsigned int* out, unsigned int v_4) {
    out[0] = (unsigned int)__umul64hi(1027807175u,
              (unsigned long long)__umulhi(gridDim.x, v_4) - threadIdx.y);
}

int main(void) {
    unsigned int* d_out;
    CHECK(cudaMalloc((void**)&d_out, sizeof(unsigned int)));
    kernel_main<<<1, 1>>>(d_out, 4042322160u);
    CHECK(cudaDeviceSynchronize());
    unsigned int h_out = 0;
    CHECK(cudaMemcpy(&h_out, d_out, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    printf("checksum=%u\n", h_out);
    CHECK(cudaFree(d_out));
    return 0;
}
