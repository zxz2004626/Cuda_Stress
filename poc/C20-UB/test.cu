#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

// UB-proof arithmetic (division/modulo guard, shift masking,
// signed-via-unsigned overflow)
#define GPU_SDIV(a, b) ((b) != 0 && !((__typeof__(b))(-1) < (__typeof__(b))(0) && (b) == (__typeof__(b))(-1) && (a) == (__typeof__(a))(1ULL << (sizeof(a) * 8 - 1))) ? (a) / (b) : 0)
#define GPU_SMOD(a, b) ((b) != 0 && !((__typeof__(b))(-1) < (__typeof__(b))(0) && (b) == (__typeof__(b))(-1) && (a) == (__typeof__(a))(1ULL << (sizeof(a) * 8 - 1))) ? (a) % (b) : 0)
#define GPU_SSHL(a, b) ((b) >= 0 && (b) < (long long)(sizeof(a) * 8) ? ((unsigned long long)(a) << (b)) : 0ULL)
#define GPU_SSHR(a, b) ((b) >= 0 && (b) < (long long)(sizeof(a) * 8) ? (a) >> (b) : 0)
#define GPU_SADD(a, b) ((__typeof__((a) + (b)))((unsigned long long)(a) + (unsigned long long)(b)))
#define GPU_SSUB(a, b) ((__typeof__((a) + (b)))((unsigned long long)(a) - (unsigned long long)(b)))
#define GPU_SMUL(a, b) ((__typeof__((a) + (b)))((unsigned long long)(a) * (unsigned long long)(b)))

// Divergence-illusion macros
#define GPU_FAKE_DIVERGE(idx, off) ((idx) - (off))
#define GPU_GROUP_DIVERGE(idx, val) ((idx) * 0 + (val))

#define GPU_CUDA_CHECK(call) do { cudaError_t _gpu_err = (call); if (_gpu_err != cudaSuccess) { fprintf(stderr, "CUDA_ERROR %s:%d: %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(_gpu_err)); return 2; } } while (0)

extern "C" __global__ __launch_bounds__(512, 2) void kernel_main(unsigned int* __restrict__ output, const unsigned int* __restrict__ input, int n, long long var_0, unsigned char var_1, unsigned long long var_2) {
    unsigned int tid = 0u;
    unsigned short v_37 = ((unsigned short)0u);
    unsigned short v_39 = ((unsigned short)0u);
    unsigned int _fv = input[GPU_SMOD(tid, ((unsigned int)(n)))];
    for (int _rep = 0; _rep < 2; _rep += 1) {
        for (int i_0 = 0; i_0 < (0 + 1); i_0 += 1) {
            {
                do {
                    v_39--;
                } while (0);
            }
            {
                int dw_4 = 0;
                do {
                    v_37 += ((unsigned short)(__umulhi(((unsigned int)(((v_39 < ((unsigned short)40501u)) ? blockIdx.x : (GPU_SSUB(((signed char)0), ((signed char)(GPU_GROUP_DIVERGE(blockIdx.x, GPU_GROUP_DIVERGE(blockIdx.x, blockIdx.x))))))))), ((unsigned int)(57005u)))));
                    dw_4++;
                } while ((dw_4 < 4));
            }
        }
        {
            int dw_5 = 0;
            do {
                {
                    int dw_6 = 0;
                    do {
                        for (int i_1 = 0; i_1 < 8; i_1 += 1) {
                            if ((tid < ((unsigned int)(n)))) {
                                atomicXor(&(output[GPU_SMOD(0u, ((unsigned int)(n)))]), ((unsigned int)(__brevll(0ull))));
                            }
                        }
                        dw_6++;
                    } while ((dw_6 < 2));
                }
                dw_5++;
            } while ((dw_5 < 2));
        }
        _fv = (_fv ^ ((unsigned int)(v_37)));
    }
    if ((tid < ((unsigned int)(n)))) {
        output[0u] = _fv;
    }
}

int main(void) {
    const int n = 16384;
    const int buf_size = n;
    const int in_size = buf_size * 32;

    // Host allocations
    unsigned int *h_input = (unsigned int*)malloc(in_size * sizeof(unsigned int));
    unsigned int *h_output = (unsigned int*)malloc(buf_size * sizeof(unsigned int));
    if (!h_input || !h_output) { fprintf(stderr, "malloc failed\n"); return 1; }

    // Deterministic input initialization (dist=hash) happens in run_pass()
    const char *input_salt_text = getenv("CUDA_STRESS_INPUT_SALT");
    const unsigned long long input_salt = (input_salt_text && *input_salt_text) ? strtoull(input_salt_text, NULL, 0) : 0ULL;

    // Device allocations
    unsigned int *d_input = NULL, *d_output = NULL;
    GPU_CUDA_CHECK(cudaMalloc((void**)&d_input, in_size * sizeof(unsigned int)));
    GPU_CUDA_CHECK(cudaMalloc((void**)&d_output, buf_size * sizeof(unsigned int)));

    // Launch configuration
    dim3 grid(64);
    dim3 block(256);

    // Checksum-only mode: skip all warmup and timing launches

    // Deterministic checksum pass (run_pass is reusable in-process)
    auto run_pass = [&](unsigned long long salt) -> int {
        for (int i = 0; i < in_size; i++) {
            h_input[i] = (unsigned int)i * 2654435761u + 12345u;
            if (salt != 0ULL) {
                unsigned long long z = salt + 0x9e3779b97f4a7c15ULL * ((unsigned long long)(unsigned int)i + 1ULL);
                z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
                z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
                z ^= z >> 31;
                unsigned int salt_word = (unsigned int)(z ^ (z >> 32));
                h_input[i] ^= salt_word ? salt_word : 0xa5a5a5a5u;
            }
        }
        GPU_CUDA_CHECK(cudaMemcpy(d_input, h_input, in_size * sizeof(unsigned int), cudaMemcpyHostToDevice));
        GPU_CUDA_CHECK(cudaMemset(d_output, 0, buf_size * sizeof(unsigned int)));
        kernel_main<<<grid, block, 0>>>(d_output, d_input, n, 9039304369631583614ll, ((unsigned char)21u), 5903491134938927595ull);
        GPU_CUDA_CHECK(cudaGetLastError());
        GPU_CUDA_CHECK(cudaDeviceSynchronize());
        GPU_CUDA_CHECK(cudaMemcpy(h_output, d_output, buf_size * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        unsigned long long hash = 42;
        for (int i = 0; i < buf_size; i++) {
            unsigned long long val = h_output[i];
            hash ^= val + 0x9e3779b9ULL + (hash << 6) + (hash >> 2);
        }
        printf("checksum=%llu\n", hash);
        return 0;
    };

    run_pass(input_salt);
    if (getenv("CUDA_STRESS_ALSO_SALT")) { run_pass(1ULL); }

    // Cleanup
    GPU_CUDA_CHECK(cudaFree(d_input));
    GPU_CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output);
    return 0;
}
