#include <cstdio>
#include <cuda_runtime.h>

// C32 最小触发候选: (short)(-x) 在 x = -32768 (INT16_MIN) 时的回绕截断
// 正确: v = (short)(-(-32768)) = (short)32768 = -32768 → (-32768 > 13569) = false
// ptxas 错误: 32 位取负不回绕 → v = +32768 → (32768 > 13569) = true
__global__ void k(short* out, short x) {
    short v = (short)(-x);
    out[0] = (short)((v > (short)13569) ? 1 : 0);
}
int main(){
    short* d; cudaMalloc(&d, 2);
    k<<<1,1>>>(d, (short)-32768); cudaDeviceSynchronize();
    short v; cudaMemcpy(&v, d, 2, cudaMemcpyDeviceToHost);
    printf("gpu=%d  correct=0  buggy=1  -> %s\n", (int)v,
        v == 0 ? "CORRECT" : (v == 1 ? "BUG" : "???"));
    return 0;
}
