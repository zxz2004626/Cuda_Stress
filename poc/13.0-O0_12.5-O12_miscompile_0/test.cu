#include <stdio.h>
__global__ void k(unsigned *o) {
    if (!blockIdx.x) {
        *o = (unsigned short)(signed char)-(int)gridDim.x - 1;
    }
}
int main() {
    unsigned *o;
    cudaMallocManaged(&o, 4);
    k<<<128, 1>>>(o);
    cudaDeviceSynchronize();
    printf("checksum=%u\n", *o);
    return 0;
}
