#include <stdio.h>
#include <cuda_runtime.h>

#define GPU_SSHR(a, b) ((a) >> (b))

__global__ void kernel_main(unsigned int* output, signed char var_2) {
    unsigned int tid = (threadIdx.x + (blockIdx.x * blockDim.x));
    long long v_11 = (37u ^ GPU_SSHR(tid, (tid & 31u)));
    unsigned int v_16 = GPU_SSHR(((0 < ((threadIdx.x))) ? (((var_2 > ((signed char)(blockDim.x))) ? v_11 : 0)) : 0), 0);
    signed char v_24 = ((0 != ((unsigned char)(__byte_perm(1431655765u, threadIdx.x, 2u)))) ? 0 : 49);
    unsigned char v_49 = ((v_16 ^ GPU_SSHR(2739585995u, (blockDim.y & 31u))) ^ GPU_SSHR(__funnelshift_rc(0, (((blockDim.y ^ v_24))), ((15u))), 12u));
    if (tid == 256u) output[0] = v_49;
}

int main(void) {
    unsigned int out = 0u, *d_out = NULL;
    cudaMalloc(&d_out, sizeof(out));
    kernel_main<<<2, 256>>>(d_out, ((signed char)0));
    cudaMemcpy(&out, d_out, sizeof(out), cudaMemcpyDeviceToHost);
    printf("checksum=%u\n", out);
    cudaFree(d_out);
    return 0;
}
