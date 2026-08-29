#include <cuda_runtime.h>
#include <stdio.h>
#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); return 1; } } while (0)
#define GPU_SMOD(a, b) ((b) != 0 ? (a) % (b) : 0)
#define GPU_SSHL(a, b) ((b) >= 0 && (b) < (long long)(sizeof(a) * 8) ? ((unsigned long long)(a) << (b)) : 0ULL)
#define GPU_SSHR(a, b) ((b) >= 0 && (b) < (long long)(sizeof(a) * 8) ? (a) >> (b) : 0)
#define GPU_SSUB(a, b) ((a) - (b))
#define GPU_SMUL(a, b) ((a) * (b))
#define R(x) input[GPU_SMOD((x), ((unsigned int)(n)))]
__device__ long long arr_0[16];
__device__ unsigned int helper_1(int p0, unsigned int p1, unsigned int p2, int p3) {
#pragma unroll 8
  for (int i_0 = 0; i_0 < (((int)(blockIdx.x) & 7) + 1); i_0 += 1) {
#pragma unroll 8
    for (int i_1 = 0; i_1 < 16; i_1 += 1) p0 = GPU_SSUB(p0, GPU_SSHL(GPU_SSUB(p3, i_0), GPU_SSHL(((int)(gridDim.y)), ((int)(threadIdx.y)))));
    if (((2 ^ 13914829) >= GPU_SSUB(127, p3))) break;
  }
  return ((((((((((((int)(((~(p1)) | ((unsigned int)(__umul64hi(((unsigned long long)(9223372036854775807ull)), ((unsigned long long)(13012689755203158607ull)))))))) * 2654435761u) ^ ((unsigned int)(p0))) * 2654435761u) ^ ((unsigned int)(p1))) * 2654435761u) ^ ((unsigned int)(p2))) * 2654435761u) ^ ((unsigned int)(p3))) * 2654435761u) ^ ((unsigned int)(~((__funnelshift_l(((unsigned int)(blockDim.x)), ((unsigned int)(gridDim.y)), ((unsigned int)(25u))) + GPU_SMUL(p1, p1))))));
}
__global__ __launch_bounds__(512, 2) void kernel_main(unsigned int *output, const unsigned int *input, int n, int var_0) {
  unsigned int _fv = 0u, tid = (threadIdx.x + GPU_SMUL(blockIdx.x, blockDim.x)), cv_14 = 0u, cv_32 = 0u;
  unsigned long long cv_29 = 0ull;
  unsigned int _lc_0 = (GPU_SMUL(tid, 2654435761u) ^ 2919732171u), _lc_1 = (GPU_SMUL(tid, 1013904226u) ^ 3617551530u), _lc_2 = (GPU_SMUL(tid, 3668339987u) ^ 232526946u);
  unsigned long long _lc_3 = ((unsigned long long)((GPU_SMUL(tid, 2027808452u) ^ 2995902468u))), _lc_9 = ((unsigned long long)((GPU_SMUL(tid, 774553834u) ^ 3793526316u))), _lc_15 = ((unsigned long long)((GPU_SMUL(tid, 3816266512u) ^ 4218185022u)));
  unsigned short _lc_10 = ((unsigned short)((GPU_SMUL(tid, 3428989595u) ^ 3363927304u)));
  unsigned char _lc_11 = ((unsigned char)((GPU_SMUL(tid, 1788458060u) ^ 1091205952u)));
  unsigned int _lc_13 = (GPU_SMUL(tid, 2802362286u) ^ 188595738u), _lc_14 = (GPU_SMUL(tid, 1161830751u) ^ 3586542586u);
  for (int _rep = 0; _rep < 2; _rep += 1) {
    unsigned long long cv_2 = ((unsigned long long)(R(tid))); unsigned int v_0 = ((unsigned int)(__ffsll(cv_2 * ((unsigned long long)(blockIdx.x)))));
    unsigned long long cv_4 = __byte_perm(4172497276u, (4294967295u ^ helper_1(((int)(blockDim.x)), threadIdx.y, 0u, ((int)(threadIdx.y)))), 4u);
    unsigned int cv_5 = R(tid), live_0 = R(tid + 1u), mix_0 = live_0;
    _fv = (GPU_SMUL(_fv, 2654435761u) ^ ((unsigned int)(((mix_0 ^ ((unsigned int)(arr_0[GPU_SMOD(helper_1(42, 12345u, 0u, 57005), 16u)])))))));
    int cv_7 = ((((unsigned int)(0)) != ((unsigned int)(var_0))) ? ((int)(tid)) : ((int)(GPU_SMOD(threadIdx.x, 32u))));
    _fv = (GPU_SMUL(_fv, 2654435761u) ^ ((unsigned int)(cv_7)));
    __shared__ unsigned int pc_comm_0[1]; pc_comm_0[0u] = 0u;
    unsigned int pc_val_0 = R(tid), pc_r2_0 = pc_comm_0[threadIdx.x];
    pc_comm_0[0u] = pc_r2_0; unsigned int pc_r3_0 = pc_comm_0[threadIdx.x];
    unsigned int cv_12 = (GPU_SSHL(((unsigned int)(var_0)), ((unsigned int)(cv_4))) ^ GPU_SSHR(v_0, ((unsigned int)(cv_4)))); cv_14 = ((cv_14 * 387276917u + _rep * 2842855661u) ^ _fv);
    for (int i_0 = 0; i_0 < 4; i_0 += 1) {
      unsigned int cv_15 = input[((unsigned int)(GPU_SMOD((live_0 * 32ull + pc_r2_0), ((unsigned long long)(n)))))];
      _fv = (GPU_SMUL(_fv, 2654435761u) ^ ((unsigned int)(cv_15)));
      _fv = (GPU_SMUL(_fv, 2654435761u) ^ (((input[GPU_SMOD((var_0 + ((unsigned int)(cv_2)) * 33u), ((unsigned int)(n)))] + input[GPU_SMOD((var_0 + 0u), ((unsigned int)(n)))]) + input[GPU_SMOD((var_0 + cv_5 * 33u), ((unsigned int)(n)))])));
      unsigned int cv_17 = ((mix_0 & 32767u) % ((pc_val_0 & 7u) + 1u)); _fv = (GPU_SMUL(_fv, 2654435761u) ^ ((live_0 & 32767u) % ((live_0 & 7u) + 1u)));
      for (int i_1 = 0; i_1 < 4; i_1 += 1) {
        _fv = (GPU_SMUL(_fv, 2654435761u) ^ ((pc_r2_0 & 255u) % ((pc_r2_0 & 3u) + 1u)));
        _fv = (GPU_SMUL(_fv, 2654435761u) ^ ((unsigned int)(GPU_SMUL((((unsigned long long)(pc_r3_0)) ^ GPU_SSHR(((unsigned long long)(pc_r3_0)), 31ull)), 10723151780598845931ull))));
        unsigned int ptx_acc_0 = (R(tid) ^ GPU_SMUL(R(tid + 7u), 2654435769u));
        unsigned int rs_v[16] = {R(tid + 1u), R(tid + 4u), R(tid + 7u), R(tid + 10u), R(tid + 13u), R(tid + 16u), R(tid + 19u), R(tid + 22u),
                                R(tid + 25u), R(tid + 28u), R(tid + 31u), R(tid + 34u), R(tid + 37u), R(tid + 40u), R(tid + 43u), R(tid + 46u)};
        ptx_acc_0 = (ptx_acc_0 ^ (rs_v[0] ^ rs_v[1] ^ rs_v[2] ^ rs_v[3] ^ rs_v[4] ^ rs_v[5] ^ rs_v[6] ^ rs_v[7] ^ rs_v[8] ^ rs_v[9] ^ rs_v[10] ^ rs_v[11] ^ rs_v[12] ^ rs_v[13] ^ rs_v[14] ^ rs_v[15]));
        unsigned int bo[2] = {R(tid + 5u), R(tid + 19u)};
        ptx_acc_0 = (ptx_acc_0 ^ ((__brev((__popc(bo[0]) ^ __clz(bo[1])))) ^ ((((unsigned int)(__ffs((bo[0] ^ bo[1]))))) ^ (__byte_perm(bo[0], bo[1], 8961u)))));
        unsigned int ii[4] = {R(tid + 1u), R(tid + 5u), R(tid + 9u), R(tid + 23u)};
        ptx_acc_0 = (ptx_acc_0 ^ ((__sad(ii[0], ii[1], 0u)) ^ ((((unsigned int)(__mul24((ii[0] & 16777215u), (ii[1] & 16777215u))))) ^ (max((min(ii[0], ii[1])), (min(ii[2], ii[3])))))));
        unsigned int la_arr_0[8];
        for (int la_i_0 = 0; la_i_0 < 8; la_i_0 += 1) la_arr_0[la_i_0] = input[(tid + (unsigned int)la_i_0 * 7u + 1u) % (unsigned int)n] ^ (tid * ((unsigned int)la_i_0 + 1u));
        unsigned int la_didx_0 = (tid * 0x6Bu + 5u) % 8u; ptx_acc_0 = ptx_acc_0 ^ (la_arr_0[la_didx_0]) ^ (la_arr_0[(la_didx_0 + 3u) % 8u]);
        cv_17 = (cv_17 ^ ptx_acc_0); _fv = (GPU_SMUL(_fv, 2654435761u) ^ ((unsigned int)(cv_17)));
      }
      for (int i_4 = 0; i_4 < 3; i_4 += 1) mix_0 = (mix_0 ^ (i_4 * cv_12));
      cv_14 = (cv_14 * 2654435761u) ^ i_0;
    }
    _fv = (GPU_SMUL(_fv, 2654435761u) ^ ((unsigned int)(cv_14)));
    cv_29 = ((unsigned long long)((((unsigned int)(((unsigned long long)((((unsigned long long)(((unsigned int)(((unsigned long long)(((cv_7 >= 1) ? 1ull : cv_4))))))) % (((unsigned long long)(arr_0[0])) | 1ull)))))) ^
             ((GPU_SMUL(((unsigned int)(cv_29)), 1401181143u) + GPU_SMUL(((unsigned int)(_rep)), 3041533379u)) ^ _fv))));
    _fv = (GPU_SMUL(_fv, 2654435761u) ^ ((unsigned int)(cv_29)));
    unsigned int cv_31 = ((unsigned int)(cv_7));
    for (int i_5 = 0; i_5 < ((pc_val_0 & 3u) + 1u); i_5 += 1) cv_31 = (cv_31 * 3266489909u) ^ i_5;
    cv_32 = (((unsigned int)((((((unsigned int)(((unsigned char)((pc_r2_0 >> 8u))))) + ((unsigned int)(((unsigned char)((pc_r2_0 >> 24u)))))) + ((unsigned int)(((unsigned char)((pc_r2_0 >> 16u)))))) + ((unsigned int)(((unsigned short)((pc_r2_0 >> 0u)))))))) ^
             ((GPU_SMUL(((unsigned int)(cv_32)), 2415085369u) + GPU_SMUL(((unsigned int)(_rep)), 3240211097u)) ^ _fv));
    _fv = (GPU_SMUL(_fv, 2654435761u) ^ ((unsigned int)(cv_32)));
    unsigned int dsw_0 = 3735928559u;
    switch ((tid & 3u)) {
    case 0: {}
    case 1: { dsw_0 = (dsw_0 ^ GPU_SMUL((dsw_0 ^ 2246822507u), 3266489909u)); }
    case 2: { dsw_0 = (dsw_0 ^ (dsw_0 + (GPU_SSHL(dsw_0, 13u) ^ GPU_SSHR(dsw_0, 7u)))); }
    case 3: {}
    }
    _fv = (GPU_SMUL(_fv, 2654435761u) ^ ((unsigned int)(dsw_0))); _fv = (GPU_SMUL(_fv, 2654435761u) ^ ((unsigned int)(cv_31)));
    _lc_9 = ((unsigned long long)((((unsigned int)(_lc_9)) + (__popc(((unsigned int)(_lc_10))) ^ __clz((((unsigned int)(_lc_11)) | 1u))))));
    _lc_13 = (((unsigned int)(_lc_13)) + __ffs((((unsigned int)(_lc_15)) | 1u))); _lc_14 = (((unsigned int)(_lc_14)) + (__popc(((unsigned int)(_lc_15))) ^ __clz((((unsigned int)(_lc_0)) | 1u))));
    _lc_15 = ((unsigned long long)((((unsigned int)(_lc_15)) ^ helper_1(((int)(((unsigned int)(_lc_0)))), ((unsigned int)(((unsigned int)(_lc_1)))), ((unsigned int)(((unsigned int)(_lc_2)))), ((int)(((unsigned int)(_lc_3))))))));
    _fv = (((((((((((_fv * 2654435761u) ^ ((unsigned int)(_lc_0))) ^ ((unsigned int)(_lc_1))) ^ ((unsigned int)(_lc_2))) ^ ((unsigned int)(_lc_3))) ^ ((unsigned int)(_lc_9))) ^ ((unsigned int)(_lc_10))) ^ ((unsigned int)(_lc_11))) ^ ((unsigned int)(_lc_13))) ^ ((unsigned int)(_lc_14))) ^ ((unsigned int)(_lc_15)));
  }
  output[0] = _fv;
}
int main(void) {
  unsigned int h_input[1] = {12345u}, h_output[1], *d_input = NULL, *d_output = NULL;
  CUDA_CHECK(cudaMalloc((void **)&d_input, sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc((void **)&d_output, sizeof(unsigned int)));
  CUDA_CHECK(cudaMemcpy(d_input, h_input, sizeof(unsigned int), cudaMemcpyHostToDevice));
  kernel_main<<<1, 1>>>(d_output, d_input, 1, 1);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(h_output, d_output, sizeof(unsigned int), cudaMemcpyDeviceToHost));
  unsigned long long hash = 42;
  hash ^= h_output[0] + 0x9e3779b9ULL + (hash << 6) + (hash >> 2);
  printf("checksum=%llu\n", hash);
}
