#include <cuda_runtime.h>
#include <stdio.h>

#define D(a, b) ((b) ? (a) / (b) : 0)

__device__ unsigned int helper_0(unsigned int p0, unsigned int p1) {
  p0 = 0u;
  p1 = (signed char)(1241937557u ^ p1);
  p0 = ((~((signed char)threadIdx.x) + (signed char)((unsigned short)p0)) |
        (signed char)__funnelshift_rc(
            (unsigned int)((unsigned short)((short)p1)), 1u, 5u));
  p0 = (short)__popc((int)(p0 + -1385588091));
  p1 = p0 * 1338290939u;
  p0 = p0 ^ __brev((threadIdx.x >= 3852573818u) ? D(2307802752u, p1)
                                                : 2654435769u);

  for (int i_3 = 0; i_3 < (((int)D(p1, p0) & 7) + 1); i_3 += 1) {
    int dw_2 = 0;
    do {
      p0 = 0u;
      dw_2++;
    } while (dw_2 < 2);
  }

  for (int i_5 = 0; i_5 < 64; i_5 += 1) {
    p1 = 4294967231u;
    p0 = (((unsigned long long)p1
           << ((((unsigned char)92u <= (unsigned char)threadIdx.x) ? p0
                                                                    : 921948663u) &
               31u)) ^
          (3771765082u >>
           ((((unsigned char)92u <= (unsigned char)threadIdx.x) ? p0
                                                                : 921948663u) &
            31u)));
  }
  return p0;
}

extern "C" __global__ void kernel_main(unsigned int *output) {
  output[0] = helper_0(0u, 0u) + helper_0(0u, 0u);
}

int main() {
  unsigned int *d_output;
  cudaMalloc((void **)&d_output, sizeof(unsigned int));
  kernel_main<<<dim3(1), dim3(1), 0>>>(d_output);
  cudaDeviceSynchronize();
  puts("checksum=0");
  return 0;
}
