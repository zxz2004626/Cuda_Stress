#include <stdio.h>
#include <cuda_runtime.h>

#define GPU_SMOD(a, b) ((b) != 0 ? (a) % (b) : 0)
#define GPU_SSHL(a, b) ((b) < (sizeof(a) * 8) ? ((a) << (b)) : 0ULL)
#define GPU_SSHR(a, b) ((a) >> (b))
#define GPU_CUDA_CHECK(call) do { cudaError_t _gpu_err = (call); if (_gpu_err != cudaSuccess) return 2; } while (0)

__device__ bool arr_1;
__device__ short arr_3[1][2];

__global__ void kernel_main(unsigned int* output, const unsigned int* input, int n) {
  unsigned int tid = (threadIdx.x + (blockIdx.x * blockDim.x));
  bool v_2 = (short)__byte_perm(tid, tid, 3u);
  unsigned int v_5 = gridDim.x;
  unsigned int v_6 =
      (((((((28250u))) / ((((((__brevll(((9223372036854775807ull))))) ^
                            ((__clzll(0)))))) |
                          1ull)))) |
       ((((((GPU_SSHL(224139538u, (tid & 31u)) ^
             GPU_SSHR(tid, (tid & 31u))))) /
          (((__ldg((&(input[GPU_SMOD(tid, ((n)))]))))) | 1ull)))));
  bool v_7 = -arr_1;
  unsigned int v_16 = arr_3[threadIdx.y][1];
  (void)(((GPU_SSHL(((((unsigned char)68u))),
                     (((v_7 <= v_2) ? __funnelshift_rc(0, 0, 0) : 0) & 31u)) ^
            GPU_SSHR(0, (((v_7 <= v_2) ? __funnelshift_rc(0, 0, 0) : 0) & 31u)))));
  unsigned int v_30 =
      ((1772223913u ^ 37u) ^
       GPU_SSHR(((((((((unsigned int)((18252946604593616694ull / (v_16 + 1ull)))))))))),
                ((__ldg((&(input[GPU_SMOD(tid, ((n)))]))) & v_6) & 31u)));
  bool v_41 = input[0];
  bool v_45 =
      ((1 + input[0])) -
      ((bool)((GPU_SSHL((v_30 - threadIdx.y), 0) ^ GPU_SSHR((v_41), 0))));
  unsigned int _fv = input[GPU_SMOD(tid, ((n)))] ^ v_2 ^ v_5 ^ v_45;
  output[tid] = _fv;
}

int main() {
  unsigned int h_output;
  unsigned int *d_input, *d_output;
  GPU_CUDA_CHECK(cudaMalloc(&d_input, sizeof(unsigned int)));
  GPU_CUDA_CHECK(cudaMalloc(&d_output, sizeof(unsigned int)));
  GPU_CUDA_CHECK(cudaMemset(d_input, 0, sizeof(unsigned int)));
  kernel_main<<<1, 1>>>(d_output, d_input, 1);
  GPU_CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(unsigned int), cudaMemcpyDeviceToHost));
  printf("checksum=%u\n", h_output);
  GPU_CUDA_CHECK(cudaFree(d_output));
  GPU_CUDA_CHECK(cudaFree(d_input));
}
