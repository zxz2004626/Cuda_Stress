#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

__device__ void helper_0(unsigned p0, unsigned p1) {
  for (int i_5 = 0; i_5 < ((((p0) ? (p1) / (p0) : 0) * p1) & 7) + 1; ++i_5) {
    for (int i_6 = 0; i_6 < 1; ++i_6)
      ;
  }
  for (int i_8 = 0; i_8 < 64; ++i_8) {
    (void)__popcll(__funnelshift_rc(p0, 13, 18));
  }
}

extern "C" __global__ void kernel_main(unsigned *output,
                                       const unsigned *input) {
  unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned v_1 = (tid == 1) ? 1 : (tid * 3412208876u);
  unsigned v_19 = __ldg(input + (tid & 32767));
  unsigned v_21 = __clz(3 | tid);
  unsigned v_23 = 232 - ((char)(__popc(gridDim.x) ^ 240));
  unsigned v_25 = (-v_19) + gridDim.x;
  unsigned v_30 =
      (37 * ((!v_21) ? (unsigned short)blockIdx.x
                     : (unsigned short)__byte_perm(0, 1982713227, 1)));
  helper_0(v_1, v_21);
  unsigned v_41 = (blockDim.y | -v_23) + (v_23 - 42);
  unsigned dsw_1 = 0;
  unsigned _fv = input[tid & 32767];
  for (int _rep = 0; _rep < 8; ++_rep) {
    for (int i_1 = 0; i_1 < ((((v_1 - v_41) ^ 37) & 7) + 1); ++i_1) {
      switch (((v_25 + 3) & 3)) {
      case 2:
        dsw_1 = (38269 | v_30);
        v_19 = __ldg(input + (tid & 32767));
      }
    }
    _fv ^= dsw_1;
  }
  output[tid] = _fv;
}

static void check_cuda(cudaError_t err, const char *expr, const char *file,
                       int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s:%d: %s: %s\n", file, line, expr,
            cudaGetErrorString(err));
    exit(2);
  }
}

#define CUDA_CHECK(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

static uint32_t mix32(uint32_t x) {
  x += 0x9e3779b9u;
  x ^= x >> 16;
  x *= 0x85ebca6bu;
  x ^= x >> 13;
  x *= 0xc2b2ae35u;
  x ^= x >> 16;
  return x;
}

static uint64_t checksum64(const std::vector<unsigned> &data) {
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < data.size(); ++i) {
    h ^= data[i];
    h *= 1099511628211ull;
    h ^= i;
  }
  return h;
}

int main(int argc, char **argv) {
  int device = 0;
  if (argc > 1)
    device = atoi(argv[1]);

  const unsigned n = 32768;
  const dim3 grid(128, 1, 1);
  const dim3 block(256, 1, 1);

  std::vector<unsigned> input(n);
  std::vector<unsigned> output(n, 0);
  for (unsigned i = 0; i < n; ++i)
    input[i] = mix32(i);

  unsigned *d_input = 0;
  unsigned *d_output = 0;
  CUDA_CHECK(cudaSetDevice(device));
  CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(unsigned)));
  CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(unsigned)));
  CUDA_CHECK(
      cudaMemcpy(d_input, input.data(), n * sizeof(unsigned), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_output, 0, n * sizeof(unsigned)));

  kernel_main<<<grid, block>>>(d_output, d_input);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(output.data(), d_output, n * sizeof(unsigned),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_input));

  printf("status=OK\n");
  printf("checksum=%llu\n", (unsigned long long)checksum64(output));
  printf("grid=%u,%u,%u\n", grid.x, grid.y, grid.z);
  printf("block=%u,%u,%u\n", block.x, block.y, block.z);
  return 0;
}
