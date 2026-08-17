#include <stdio.h>
#include <cuda_runtime.h>

#define GPU_SMOD(a, b) ((b) ? (a) % (b) : 0)

__global__ void kernel_main(unsigned* output, const unsigned* input, int n) {
    bool v_22 = ((unsigned char)__byte_perm(0u, input[GPU_SMOD(threadIdx.x, ((unsigned)(n)))], 6u));
    unsigned _fv = input[GPU_SMOD(threadIdx.x, ((unsigned)(n)))];
    _fv ^= ((unsigned)(v_22));
    if (threadIdx.x == 1u) output[0] = _fv;
}

int main(void) {
    const int n = 2;
    unsigned h_input[2] = {12345u, 2654448106u};
    unsigned out = 0u, *d_out = NULL, *d_in = NULL;
    cudaMalloc((void**)&d_in, sizeof(h_input));
    cudaMalloc((void**)&d_out, sizeof(out));
    cudaMemcpy(d_in, h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    kernel_main<<<1, 2>>>(d_out, d_in, n);
    cudaMemcpy(&out, d_out, sizeof(out), cudaMemcpyDeviceToHost);
    printf("checksum=%u\n", out);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
