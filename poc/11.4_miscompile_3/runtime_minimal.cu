#include <cuda_runtime.h>
#include <stdio.h>
__device__ unsigned long long arr_0[11];
__global__ void kernel_main(unsigned int *output, const unsigned int *input, int n) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int in0 = input[n ? tid % n : 0];
  unsigned int ov = in0 & 0xffffu ^ in0 & 1u;
  bool v_1 = (unsigned char)__byte_perm(arr_0[0u], __brevll(gridDim.x), 6u);
  unsigned long long v_3 = 0x930acfeu ^ arr_0[0u];
  unsigned long long v_35 = arr_0[(n == gridDim.x) ? 0 : arr_0[0] % 11];
  long long v_38 = gridDim.x <= blockDim.x ? __funnelshift_r(2u, 0x5b79443cu, 8u) : v_3;
  unsigned int dsw_0 = tid ^ 0xdeadbeefu;
  unsigned int _fv = in0;
  for (int _rep = 0; _rep < 3; _rep++) {
    dsw_0 ^= dsw_0 * 0x9e3779b9u ^ (dsw_0 >> 16u);
    if (tid < n) {
      atomicMin(output, ((tid << (blockDim.x & 31u)) ^ (31u >> (blockDim.x & 31u))) + (unsigned int)((n == 255) ? (long long)__funnelshift_lc(0, (threadIdx.x == 0xffffffffu ? threadIdx.y : tid), 7) : v_38));
    }
    _fv ^= 63242u;
    _fv ^= 0x9c229edau;
    _fv ^= v_1;
    _fv ^= v_3;
    _fv ^= (short)(0x613f4ca2u ^ __brevll(v_35));
    _fv ^= dsw_0;
    _fv ^= ov;
  }
  if (!tid) output[0] = _fv;
}
int main() {
  unsigned int *d, *o, h = 0;
  cudaMalloc(&d, 4);
  cudaMemset(d, 0, 4);
  cudaMalloc(&o, 4);
  kernel_main<<<1, 1>>>(o, d, 1);
  cudaMemcpy(&h, o, 4, cudaMemcpyDeviceToHost);
  printf("checksum=%u\n", h);
}
