__device__ unsigned int helper(unsigned int p0) {
#pragma unroll 8
    for (int i = 0; i < ((int)(gridDim.x) & 7) + 1; i++) {
        if (blockIdx.x == 0) {
#pragma unroll 2
            for (int j = 0; j < 4; j++) {
                p0 += 1u;
                if (gridDim.y <= gridDim.x) break;
            }
            __threadfence();
        }
    }
    return p0;
}

extern "C" __global__ void kernel_main(unsigned int* out) {
    unsigned int r = 0;
    for (int i = 0; i < 2; i++) r = helper(42u);
    out[threadIdx.x] = r;
}
