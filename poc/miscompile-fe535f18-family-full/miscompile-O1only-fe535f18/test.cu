#include <cuda_runtime.h>
#include <stdio.h>
#define GPU_SMOD(a, b)                                                         \
  ((b) != 0 && !((__typeof__(b))(-1) < (__typeof__(b))(0) &&                   \
                 (b) == (__typeof__(b))(-1) &&                                 \
                 (a) == (__typeof__(a))(1ULL << (sizeof(a) * 8 - 1)))          \
       ? (a) % (b)                                                             \
       : 0)
#define GPU_SSHL(a, b)                                                         \
  ((b) >= 0 && (b) < (long long)(sizeof(a) * 8)                                \
       ? ((unsigned long long)(a) << (b))                                      \
       : 0ULL)
#define GPU_SSHR(a, b)                                                         \
  ((b) >= 0 && (b) < (long long)(sizeof(a) * 8) ? (a) >> (b) : 0)
#define GPU_SSUB(a, b)                                                         \
  ((__typeof__((a) + (b)))((unsigned long long)(a) - (unsigned long long)(b)))
#define GPU_SMUL(a, b)                                                         \
  ((__typeof__((a) + (b)))((unsigned long long)(a) * (unsigned long long)(b)))
__device__ long long arr_0[16];
__device__ unsigned int helper_1(int p0, unsigned int p1, unsigned int p2,
                                 int p3) {
#pragma unroll 8
  for (int i_0 = 0;
       i_0 < ((((int)(((unsigned int)((((int)((-(blockIdx.x))))))))) & 7) + 1);
       i_0 += 1) {
#pragma unroll 8
    for (int i_1 = 0; i_1 < 16; i_1 += 1) {
      p0 = GPU_SSUB(
          p0, GPU_SSHL(GPU_SSUB(p3, i_0),
                       GPU_SSHL(((int)(gridDim.y)), ((int)(threadIdx.y)))));
    }
    if (((2 ^ 13914829) >= GPU_SSUB(127, p3))) {
      break;
    }
  }
  unsigned int v_1 =
      (~((__funnelshift_l(((unsigned int)(blockDim.x)),
                          ((unsigned int)(gridDim.y)), ((unsigned int)(25u))) +
          GPU_SMUL(p1, p1))));
  return (
      GPU_SMUL(
          (GPU_SMUL(
               (GPU_SMUL(
                    (GPU_SMUL(
                         (GPU_SMUL(
                              ((((int)((
                                  (~(p1)) |
                                  ((unsigned int)(__umul64hi(
                                      ((unsigned long long)(9223372036854775807ull)),
                                      ((unsigned long long)(13012689755203158607ull)))))))))),
                              2654435761u) ^
                          ((unsigned int)(p0))),
                         2654435761u) ^
                     ((unsigned int)(p1))),
                    2654435761u) ^
                ((unsigned int)(p2))),
               2654435761u) ^
           ((unsigned int)(p3))),
          2654435761u) ^
      ((unsigned int)(v_1)));
}
__global__ __launch_bounds__(512, 2) void kernel_main(
    unsigned int *__restrict__ output, const unsigned int *__restrict__ input,
    int n, int var_0) {
  unsigned int _fv = 0u;
  unsigned int tid = (threadIdx.x + GPU_SMUL(blockIdx.x, blockDim.x));
  unsigned int cv_14 = 0u;
  unsigned long long cv_29 = 0ull;
  unsigned int cv_32 = 0u;
  unsigned int _lc_0 = (GPU_SMUL(tid, 2654435761u) ^ 2919732171u);
  unsigned int _lc_1 = (GPU_SMUL(tid, 1013904226u) ^ 3617551530u);
  unsigned int _lc_2 = (GPU_SMUL(tid, 3668339987u) ^ 232526946u);
  unsigned long long _lc_3 =
      ((unsigned long long)((GPU_SMUL(tid, 2027808452u) ^ 2995902468u)));
  unsigned long long _lc_9 =
      ((unsigned long long)((GPU_SMUL(tid, 774553834u) ^ 3793526316u)));
  unsigned short _lc_10 =
      ((unsigned short)((GPU_SMUL(tid, 3428989595u) ^ 3363927304u)));
  unsigned char _lc_11 =
      ((unsigned char)((GPU_SMUL(tid, 1788458060u) ^ 1091205952u)));
  unsigned int _lc_13 = (GPU_SMUL(tid, 2802362286u) ^ 188595738u);
  unsigned int _lc_14 = (GPU_SMUL(tid, 1161830751u) ^ 3586542586u);
  unsigned long long _lc_15 =
      ((unsigned long long)((GPU_SMUL(tid, 3816266512u) ^ 4218185022u)));
  for (int _rep = 0; _rep < 2; _rep += 1) {
    unsigned long long cv_2 = ((unsigned long long)(input[GPU_SMOD(tid, ((unsigned int)(n)))]));
    unsigned int v_0 =
        ((unsigned int)(__ffsll(
             ((unsigned long long)(((unsigned long long)(GPU_SMUL(
                 ((unsigned long long)(cv_2)),
                 ((unsigned long long)(blockIdx.x))))))))));
    unsigned int cv_3 = GPU_SMOD(threadIdx.x, 32u);
    unsigned long long cv_4 = ((unsigned long long)(__byte_perm(
        4172497276u,
        (4294967295u ^ helper_1(((int)(blockDim.x)), threadIdx.y, 0u,
                                ((int)(threadIdx.y)))),
        4u)));
    unsigned int cv_5 = input[GPU_SMOD(tid, ((unsigned int)(n)))];
    unsigned int live_0 = input[GPU_SMOD((tid + 1u), ((unsigned int)(n)))];
    unsigned int live_2;
    unsigned int live_3 = live_0;
    unsigned int live_4 = live_0;
    unsigned int mix_0 = live_0;
    unsigned int cv_0 =
        (mix_0 ^
         ((unsigned int)(arr_0[GPU_SMOD(helper_1(42, live_0, blockIdx.x, 57005), 16u)])));
    _fv = (GPU_SMUL(_fv, 2654435761u) ^ ((unsigned int)(cv_0)));
    int cv_7 =
        ((((unsigned int)(0)) != ((unsigned int)(var_0))) ? ((int)(tid))
                                                              : ((int)(cv_3)));
    _fv = (GPU_SMUL(_fv, 2654435761u) ^ ((unsigned int)(cv_7)));
    __shared__ unsigned int pc_comm_0[2];
    pc_comm_0[0u] = 0u;
    pc_comm_0[1u] = 0u;
    unsigned int pc_val_0 = input[GPU_SMOD(tid, ((unsigned int)(n)))];
    __syncthreads();
    unsigned int pc_r2_0 = pc_comm_0[threadIdx.x];
    pc_comm_0[1u] = pc_r2_0;
    unsigned int pc_r3_0 = pc_comm_0[threadIdx.x];
    live_2 = pc_r2_0;
    unsigned char cv_10 = (unsigned char)pc_val_0;
    unsigned int cv_12 =
        (GPU_SSHL(((unsigned int)(var_0)), ((unsigned int)(cv_4))) ^
         GPU_SSHR(v_0, ((unsigned int)(cv_4))));
    unsigned int cv_13 = live_4;
    cv_14 = (((unsigned int)(tid)) ^
             ((GPU_SMUL(((unsigned int)(cv_14)), 387276917u) +
               GPU_SMUL(((unsigned int)(_rep)), 2842855661u)) ^
              _fv));
    for (int i_0 = 0; i_0 < 4; i_0 += 1) {
      unsigned int cv_15 = input[((unsigned int)(GPU_SMOD(
          (GPU_SMUL(((unsigned long long)(cv_13)), 32ull) +
           ((unsigned long long)(pc_r2_0))),
          ((unsigned long long)(n)))))];
      _fv = (GPU_SMUL(_fv, 2654435761u) ^ ((unsigned int)(cv_15)));
      unsigned int cv_buf_0[3];
      cv_buf_0[0u] = input[GPU_SMOD((((unsigned int)(var_0)) + GPU_SMUL(((unsigned int)(cv_2)), 33u)), ((unsigned int)(n)))];
      cv_buf_0[1u] = input[GPU_SMOD((((unsigned int)(var_0)) + GPU_SMUL(((unsigned int)(0)), 127u)), ((unsigned int)(n)))];
      cv_buf_0[2u] = input[GPU_SMOD((((unsigned int)(var_0)) + GPU_SMUL(cv_5, 33u)), ((unsigned int)(n)))];
      unsigned int cv_16 = ((cv_buf_0[0u] + cv_buf_0[1u]) + cv_buf_0[2u]);
      _fv = (GPU_SMUL(_fv, 2654435761u) ^ ((unsigned int)(cv_16)));
      unsigned int cv_17 =
          ((mix_0 & 32767u) % ((((unsigned int)(cv_10)) & 7u) + 1u));
      unsigned int cv_18 = ((live_3 & 32767u) % ((live_4 & 7u) + 1u));
      _fv = (GPU_SMUL(_fv, 2654435761u) ^ ((unsigned int)(cv_18)));
      for (int i_1 = 0; i_1 < 4; i_1 += 1) {
        unsigned int cv_21 = ((pc_r2_0 & 255u) % ((live_2 & 3u) + 1u));
        _fv = (GPU_SMUL(_fv, 2654435761u) ^ ((unsigned int)(cv_21)));
        unsigned long long cv_25 =
            GPU_SMUL((((unsigned long long)(pc_r3_0)) ^
                      GPU_SSHR(((unsigned long long)(pc_r3_0)), 31ull)),
                     10723151780598845931ull);
        _fv = (GPU_SMUL(_fv, 2654435761u) ^ ((unsigned int)(cv_25)));
        unsigned int ptx_acc_0 =
            (input[GPU_SMOD(tid, ((unsigned int)(n)))] ^
             GPU_SMUL(input[GPU_SMOD((tid + 7u), ((unsigned int)(n)))],
                      2654435769u));
        unsigned int rs_v_0 =
            input[GPU_SMOD((tid + 1u), ((unsigned int)(n)))];
        unsigned int rs_v_1 =
            input[GPU_SMOD((tid + 4u), ((unsigned int)(n)))];
        unsigned int rs_v_2 =
            input[GPU_SMOD((tid + 7u), ((unsigned int)(n)))];
        unsigned int rs_v_3 =
            input[GPU_SMOD((tid + 10u), ((unsigned int)(n)))];
        unsigned int rs_v_4 =
            input[GPU_SMOD((tid + 13u), ((unsigned int)(n)))];
        unsigned int rs_v_5 =
            input[GPU_SMOD((tid + 16u), ((unsigned int)(n)))];
        unsigned int rs_v_6 =
            input[GPU_SMOD((tid + 19u), ((unsigned int)(n)))];
        unsigned int rs_v_7 =
            input[GPU_SMOD((tid + 22u), ((unsigned int)(n)))];
        unsigned int rs_v_8 =
            input[GPU_SMOD((tid + 25u), ((unsigned int)(n)))];
        unsigned int rs_v_9 =
            input[GPU_SMOD((tid + 28u), ((unsigned int)(n)))];
        unsigned int rs_v_10 =
            input[GPU_SMOD((tid + 31u), ((unsigned int)(n)))];
        unsigned int rs_v_11 =
            input[GPU_SMOD((tid + 34u), ((unsigned int)(n)))];
        unsigned int rs_v_12 =
            input[GPU_SMOD((tid + 37u), ((unsigned int)(n)))];
        unsigned int rs_v_13 =
            input[GPU_SMOD((tid + 40u), ((unsigned int)(n)))];
        unsigned int rs_v_14 =
            input[GPU_SMOD((tid + 43u), ((unsigned int)(n)))];
        unsigned int rs_v_15 =
            input[GPU_SMOD((tid + 46u), ((unsigned int)(n)))];
        ptx_acc_0 = (ptx_acc_0 ^ (rs_v_0 ^ rs_v_1 ^ rs_v_2 ^ rs_v_3 ^ rs_v_4 ^ rs_v_5 ^ rs_v_6 ^ rs_v_7 ^ rs_v_8 ^ rs_v_9 ^ rs_v_10 ^ rs_v_11 ^ rs_v_12 ^ rs_v_13 ^ rs_v_14 ^ rs_v_15));
        unsigned int bo_x_0 = input[GPU_SMOD((tid + 5u), ((unsigned int)(n)))];
        unsigned int bo_y_0 = input[GPU_SMOD((tid + 19u), ((unsigned int)(n)))];
        unsigned int bo_brev_0 = __brev((__popc(bo_x_0) ^ __clz(bo_y_0)));
        unsigned int bo_ffs_0 = ((unsigned int)(__ffs((bo_x_0 ^ bo_y_0))));
        unsigned int bo_bp_0 = __byte_perm(bo_x_0, bo_y_0, 8961u);
        ptx_acc_0 = (ptx_acc_0 ^ (bo_brev_0 ^ (bo_ffs_0 ^ bo_bp_0)));
        unsigned int ii_a_0 = input[GPU_SMOD((tid + 1u), ((unsigned int)(n)))];
        unsigned int ii_b_0 = input[GPU_SMOD((tid + 5u), ((unsigned int)(n)))];
        unsigned int ii_c_0 = input[GPU_SMOD((tid + 9u), ((unsigned int)(n)))];
        unsigned int ii_d_0 = input[GPU_SMOD((tid + 23u), ((unsigned int)(n)))];
        unsigned int ii_sad_0 = __sad(ii_a_0, ii_b_0, 0u);
        unsigned int ii_m24_0 = ((unsigned int)(__mul24((ii_a_0 & 16777215u), (ii_b_0 & 16777215u))));
        unsigned int ii_mnab_0 = min(ii_a_0, ii_b_0);
        unsigned int ii_mncd_0 = min(ii_c_0, ii_d_0);
        unsigned int ii_mxmm_0 = max(ii_mnab_0, ii_mncd_0);
        ptx_acc_0 = (ptx_acc_0 ^ (ii_sad_0 ^ (ii_m24_0 ^ ii_mxmm_0)));
        unsigned int la_arr_0[8];
        for (int la_i_0 = 0; la_i_0 < 8; la_i_0 += 1) {
          la_arr_0[la_i_0] =
              input[(tid + (unsigned int)la_i_0 * 7u + 1u) % (unsigned int)n] ^
              (tid * ((unsigned int)la_i_0 + 1u));
        }
        unsigned int la_didx_0 = (tid * 0x6Bu + 5u) % 8u;
        unsigned int la_rd_0 = la_arr_0[la_didx_0];
        unsigned int la_rd2_0 = la_arr_0[(la_didx_0 + 3u) % 8u];
        ptx_acc_0 = ptx_acc_0 ^ la_rd_0 ^ la_rd2_0;
        cv_17 = (cv_17 ^ ptx_acc_0);
        _fv = (GPU_SMUL(_fv, 2654435761u) ^ ((unsigned int)(cv_17)));
      }
      for (int i_4 = 0; i_4 < 6; i_4 += 1) {
        mix_0 = (mix_0 ^ ((unsigned int)(GPU_SMUL(i_4, cv_12))));
      }
      cv_14 = (GPU_SMUL(cv_14, 2654435761u) ^ ((unsigned int)(i_0)));
    }
    _fv = (GPU_SMUL(_fv, 2654435761u) ^ ((unsigned int)(cv_14)));
    cv_29 = ((unsigned long long)((
        ((unsigned int)(((unsigned long long)((
            ((unsigned long long)(((unsigned int)(((unsigned long long)(((cv_7 >= 1) ? 1ull : cv_4))))))) %
            (((unsigned long long)(arr_0[0])) |
             1ull)))))) ^
        ((GPU_SMUL(((unsigned int)(cv_29)), 1401181143u) +
          GPU_SMUL(((unsigned int)(_rep)), 3041533379u)) ^
         _fv))));
    _fv = (GPU_SMUL(_fv, 2654435761u) ^ ((unsigned int)(cv_29)));
    unsigned int cv_31 = ((unsigned int)(cv_7));
    for (int i_5 = 0; i_5 < ((((int)(((unsigned int)(cv_10)))) & 3) + 1);
         i_5 += 1) {
      cv_31 = (GPU_SMUL(cv_31, 3266489909u) ^ ((unsigned int)(i_5)));
    }
    cv_32 = (((unsigned int)((
                 ((((unsigned int)(((unsigned char)((pc_r2_0 >> 8u))))) +
                   ((unsigned int)(((unsigned char)((pc_r2_0 >> 24u)))))) +
                  ((unsigned int)(((unsigned char)((pc_r2_0 >> 16u)))))) +
                 ((unsigned int)(((unsigned short)((pc_r2_0 >> 0u)))))))) ^
             ((GPU_SMUL(((unsigned int)(cv_32)), 2415085369u) +
               GPU_SMUL(((unsigned int)(_rep)), 3240211097u)) ^
              _fv));
    _fv = (GPU_SMUL(_fv, 2654435761u) ^ ((unsigned int)(cv_32)));
    unsigned int dsw_0 = (tid ^ 3735928559u);
    switch ((tid & 3u)) {
    case 0: {
    }
    case 1: {
      dsw_0 = (dsw_0 ^ GPU_SMUL((dsw_0 ^ 2246822507u), 3266489909u));
    }
    case 2: {
      dsw_0 = (dsw_0 ^ (dsw_0 + (GPU_SSHL(dsw_0, 13u) ^ GPU_SSHR(dsw_0, 7u))));
    }
    case 3: {
    }
    }
    _fv = (GPU_SMUL(_fv, 2654435761u) ^ ((unsigned int)(dsw_0)));
    unsigned int cv_37 = (cv_31 ^ ((unsigned int)((short)cv_7)));
    _fv = (GPU_SMUL(_fv, 2654435761u) ^ ((unsigned int)(cv_37)));
    _lc_9 = ((unsigned long long)((((unsigned int)(_lc_9)) +
                                   (__popc(((unsigned int)(_lc_10))) ^
                                    __clz((((unsigned int)(_lc_11)) | 1u))))));
    _lc_13 =
        (((unsigned int)(_lc_13)) + __ffs((((unsigned int)(_lc_15)) | 1u)));
    _lc_14 =
        (((unsigned int)(_lc_14)) + (__popc(((unsigned int)(_lc_15))) ^
                                     __clz((((unsigned int)(_lc_0)) | 1u))));
    _lc_15 = ((
        unsigned long long)((((unsigned int)(_lc_15)) ^
                             helper_1(((int)(((unsigned int)(_lc_0)))),
                                      ((unsigned int)(((unsigned int)(_lc_1)))),
                                      ((unsigned int)(((unsigned int)(_lc_2)))),
                                      ((int)(((unsigned int)(_lc_3))))))));
    _fv =
        ((((((((((((((((GPU_SMUL(_fv, 2654435761u) ^ ((unsigned int)(_lc_0))) ^
                       ((unsigned int)(_lc_1))) ^
                      ((unsigned int)(_lc_2))) ^
                     ((unsigned int)(_lc_3)))))))) ^
               ((unsigned int)(_lc_9))) ^
              ((unsigned int)(_lc_10))) ^
             ((unsigned int)(_lc_11)))) ^
           ((unsigned int)(_lc_13))) ^
          ((unsigned int)(_lc_14))) ^
         ((unsigned int)(_lc_15)));
  }
  if (tid == 0u) {
    output[0] = _fv;
  }
}
int main(void) {
  const int n = 1;
  unsigned int h_input[1];
  unsigned int h_output[1];
  unsigned int *d_input = NULL, *d_output = NULL;
  cudaMalloc((void **)&d_input, sizeof(unsigned int));
  cudaMalloc((void **)&d_output, sizeof(unsigned int));
  static long long zero_arr_0[16];
  cudaMemcpyToSymbol(arr_0, zero_arr_0, sizeof(zero_arr_0));
  h_input[0] = 12345u;
  cudaMemcpy(d_input, h_input, sizeof(unsigned int), cudaMemcpyHostToDevice);
  kernel_main<<<1, 1>>>(d_output, d_input, n, 1);
  cudaMemcpy(h_output, d_output, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  unsigned long long hash = 42;
  hash ^= h_output[0] + 0x9e3779b9ULL + (hash << 6) + (hash >> 2);
  printf("checksum=%llu\n", hash);
}
