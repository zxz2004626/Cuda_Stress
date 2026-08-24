#include <cuda_runtime.h>
#include <stdio.h>
using u = unsigned;
using ull = unsigned long long;
#define GPU_SMOD(a, b) ((b) != 0 ? (a) % (b) : 0)
#define GPU_SSHL(a, b) ((b) < sizeof(a) * 8 ? ((ull)(a) << (b)) : 0)
#define GPU_SSHR(a, b) ((b) < sizeof(a) * 8 ? (a) >> (b) : 0)
#define R(x) input[GPU_SMOD(x, n)]
#define M 2654435761
__device__ int arr_0[16];
__device__ u helper_1(u p0, u p1, u p2, int p3) {
#pragma unroll 8
  for (int i_0 = 0; i_0 < ((blockIdx.x & 7) + 1); ++i_0) {
#pragma unroll 8
    for (int i_1 = 0; i_1 < 16; ++i_1)
      p0 -= GPU_SSHL(p3 - i_0, GPU_SSHL(gridDim.y, threadIdx.y));
    if (13914831 >= (127 - p3))
      break;
  }
  return ((((((((((~p1 | __umul64hi(0x7fffffffffffffff, 0xb4965c635f3fc24f)) * M) ^ p0) * M) ^ p1) *
              M) ^
             p2) *
            M) ^
           p3) *
          M) ^
         ~(__funnelshift_l(1, gridDim.y, 25) + p1 * p1);
}
__global__ __launch_bounds__(512, 2) void kernel_main(u *output, u *input, u n, u var_0) {
  u _fv = 0, tid = threadIdx.x + blockIdx.x, cv_14 = 0, cv_32 = 0, cv_29 = 0;
  u _lc_0 = tid * M ^ 2919732171;
  u _lc_3 = tid * 2027808452 ^ 2995902468, _lc_9 = 0, _lc_15 = 0;
  u _lc_11 = (char)(tid * 1788458060 ^ 1091205952);
  u _lc_13 = 0, _lc_14 = 0;
  for (u _rep = 0; _rep < 2; ++_rep) {
    u cv_2 = R(tid);
    u v_0 = __ffsll(cv_2 * blockIdx.x);
    u cv_4 = __byte_perm(4172497276, ~helper_1(1, threadIdx.y, 0, threadIdx.y), 4);
    u cv_5 = R(tid), live_0 = cv_2, mix_0 = live_0;
    _fv = (_fv * M) ^ (mix_0 ^ arr_0[helper_1(42, 12345, 0, 57005) & 15]);
    u cv_7 = var_0 ? tid : threadIdx.x & 31;
    _fv = _fv * M ^ cv_7;
    __shared__ u pc_comm_0[1];
    pc_comm_0[0] = 0;
    u pc_val_0 = R(tid), pc_r2_0 = pc_comm_0[threadIdx.x];
    pc_comm_0[0] = pc_r2_0;
    u pc_r3_0 = pc_comm_0[threadIdx.x];
    u cv_12 = GPU_SSHL(1, cv_4) ^ GPU_SSHR(v_0, cv_4);
    cv_14 = (cv_14 * 387276917 + _rep * 2842855661) ^ _fv;
    for (u i_0 = 0; i_0 < 4; ++i_0) {
      u cv_15 = input[(u)GPU_SMOD(live_0 * 32ll + pc_r2_0, n)];
      _fv = _fv * M ^ cv_15;
      _fv = (_fv * M) ^ (input[GPU_SMOD(var_0 + cv_2 * 33, n)] + *input +
                         input[GPU_SMOD(var_0 + cv_5 * 33, n)]);
      u cv_17 = (mix_0 & 32767) % ((pc_val_0 & 7) + 1);
      _fv = (_fv * M) ^ ((live_0 & 32767) % ((live_0 & 7) + 1));
      for (u i_1 = 0; i_1 < 3; ++i_1) {
        _fv = (_fv * M) ^ ((pc_r2_0 & 255) % ((pc_r2_0 & 3) + 1));
        _fv = (_fv * M) ^ (u)((pc_r3_0 ^ (pc_r3_0 >> 31)) * 0x94d049bb133111eb);
        u ptx_acc_0 = (R(tid) ^ (R(tid + 7) * 2654435769));
        u rs_v[16] = {R(tid + 1),  R(tid + 4),  R(tid + 7u), R(tid + 10), R(tid + 13), R(tid + 16),
                      R(tid + 19), R(tid + 22), R(tid + 25), R(tid + 28), R(tid + 31), R(tid + 34),
                      R(tid + 37), R(tid + 40), R(tid + 43), R(tid + 46)};
        ptx_acc_0 ^= rs_v[0] ^ rs_v[1] ^ rs_v[2] ^ rs_v[3] ^ rs_v[4] ^ rs_v[5] ^ rs_v[6] ^ rs_v[7] ^
                     rs_v[8] ^ rs_v[9] ^ rs_v[10] ^ rs_v[11] ^ rs_v[12] ^ rs_v[13] ^ rs_v[14] ^
                     rs_v[15];
        u b0 = R(tid + 5), b1 = R(tid + 19u);
        ptx_acc_0 ^= __brev(__popc(b0) ^ __clz(b1)) ^ __ffs(b0 ^ b1) ^ __byte_perm(b0, b1, 8961);
        u j0 = R(tid + 1), j1 = R(tid + 5u), j2 = R(tid + 9), j3 = R(tid + 23);
        ptx_acc_0 ^= __sad(j0, j1, 0) ^ __mul24(j0 & 0xffffff, j1 & 0xffffff) ^
                     max(min(j0, j1), min(j2, j3));
        u la_arr_0[8];
        for (u la_i_0 = 0; la_i_0 < 8; ++la_i_0)
          la_arr_0[la_i_0] = input[(tid + la_i_0 * 7 + 1) % n];
        u la_didx_0 = (tid * 107 + 5) & 7;
        ptx_acc_0 ^= la_arr_0[la_didx_0] ^ la_arr_0[(la_didx_0 + 3) & 7];
        cv_17 ^= ptx_acc_0;
        _fv = _fv * M ^ cv_17;
      }
      for (u i_4 = 0; i_4 < 3; ++i_4)
        mix_0 ^= i_4 * cv_12;
      cv_14 ^= i_0;
    }
    _fv = (_fv * M) ^ cv_14;
    cv_29 = ((cv_7 ? 1 : cv_4) % ((ull)arr_0[0] | 1)) ^
            ((cv_29 * 1401181143 + _rep * 3041533379) ^ _fv);
    _fv = (_fv * M) ^ cv_29;
    u cv_31 = cv_7;
    for (u i_5 = 0; i_5 < ((pc_val_0 & 3) + 1); ++i_5)
      cv_31 = cv_31 * 3266489909 ^ i_5;
    cv_32 = ((((pc_r2_0 >> 8) & 255) + ((pc_r2_0 >> 24) & 255) + ((pc_r2_0 >> 16) & 255) +
              (pc_r2_0 & 65535)) ^
             ((cv_32 * 2415085369 + _rep * 3240211097) ^ _fv));
    _fv = (_fv * M) ^ cv_32;
    u dsw_0 = 3735928559;
    switch (threadIdx.x & 3) {
    case 0:
    case 1:
      dsw_0 ^= (dsw_0 ^ 2246822507) * 3266489909;
    case 2:
      dsw_0 ^= dsw_0 + ((dsw_0 << 13) ^ (dsw_0 >> 7));
    case 3:;
    }
    _fv = (_fv * M) ^ dsw_0;
    _fv = (_fv * M) ^ cv_31;
    _lc_9 += 6 ^ __clz(_lc_11 | 1);
    _lc_13 += __ffs(_lc_15 | 1);
    _lc_14 += 21 ^ __clz(_lc_0 | 1);
    _lc_15 ^= helper_1(_lc_0, 0, 0, _lc_3);
    _fv = (_fv * M) ^ _lc_9 ^ _lc_11 ^ _lc_13 ^ _lc_14 ^ _lc_15;
  }
  *output = _fv;
}
int main() {
  u *data;
  cudaMallocManaged((void **)&data, 4);
  *data = 12345;
  kernel_main<<<1, 1>>>(data, data, 1, 1);
  cudaDeviceSynchronize();
  printf("output=%u\n", *data);
}
