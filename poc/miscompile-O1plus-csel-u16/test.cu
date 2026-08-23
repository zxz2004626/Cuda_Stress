#include <stdio.h>
#define S(a,b) (((a) < 0x80000000u) ? (a) % (b) : 0)
__global__ void k(unsigned int *o, const unsigned int *i, int n) {
  unsigned int f = 0u, t = threadIdx.x;
  for (int r = 0; r < 2; r += 1) {
    unsigned int p = (i[S(t, n)] ^ i[S((t + 7u), n)]);
    unsigned int rr[12];

    for (int j = 0; j < 12; j += 1) {
      rr[j] = i[S((t + 4u + (unsigned int)j * 3u), n)];
    }

    for (int j = 0; j < 12; j += 1) p ^= rr[j];

    if (t == 9999u) o[3] = p;
    unsigned int s = i[S(t, n)];

    if (i[S((t * 4u + 2u), n)] < i[S((t * 4u), n)]) {
      unsigned short c = ((f < (t / 32u)) ? ((unsigned short)((signed char)s)) : ((unsigned short)s));
      f = ((unsigned int)c);
      if (t == 2u) o[2] = (unsigned int)c;
    }
  }
}
int main(void) {
  unsigned int h_i[256], h_o[4];
  unsigned int *d_i = NULL, *d_o = NULL;
  cudaMalloc((void **)&d_i, 256 * sizeof(unsigned int));
  cudaMalloc((void **)&d_o, 4 * sizeof(unsigned int));
  cudaMemset(d_o, 0, 4 * sizeof(unsigned int));
  for (int i = 0; i < 256; i++) h_i[i] = (unsigned int)i * 2654435761u + 12345u;
  cudaMemcpy(d_i, h_i, 256 * sizeof(unsigned int), cudaMemcpyHostToDevice);
  k<<<1, 256>>>(d_o, d_i, 256);
  cudaMemcpy(h_o, d_o, 4 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  printf("checksum=%u\n", h_o[2]);
  return 0;
}
