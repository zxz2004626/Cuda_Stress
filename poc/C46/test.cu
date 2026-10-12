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

__forceinline__ __device__ unsigned int helper_3(unsigned int p0, int p1) {
    if ((((unsigned char)0u) >= ((unsigned char)0u))) {
        switch ((__funnelshift_r(0u, 0u, 0u) % 3u)) {
            case 0: {
                break;
            }
            case 1: {
                signed char v_5 = ((signed char)((0u ^ GPU_SSHR(__funnelshift_r(0u, 0u, 0u), 0u))));
                break;
            }
            case 2: {
                unsigned int v_6 = 0u;
                break;
            }
            default: {
                break;
            }
        }
    }
    return 0u;
}

extern "C" __global__ __launch_bounds__(384, 1) void kernel_main(unsigned int* __restrict__ output, const unsigned int* __restrict__ input, int n, unsigned int var_0, unsigned short var_1, short var_2) {
    unsigned int tid = (threadIdx.x + GPU_SMUL(blockIdx.x, blockDim.x));
    unsigned char v_6 = ((unsigned char)0u);
    unsigned int v_21 = ((unsigned int)(GPU_SMUL(((unsigned long long)(((unsigned short)(((unsigned char)((blockIdx.x))))))), (0ll | 42ull))));
    bool v_24 = 0;
    unsigned int v_29 = ((GPU_SSHL((2654435769u & v_21), (v_21 & 31u)) ^ 0u) + 0u);
    unsigned int _fv = input[GPU_SMOD(tid, ((unsigned int)(n)))];
    for (int _rep = 0; _rep < 2; _rep += 1) {
        if (((~(0ll)) < 0ll)) {
            for (int i_0 = 0; i_0 < 16; i_0 += 1) {
                {
                    int dw_1 = 0;
                    do {
                        v_24 |= ((bool)(((long long)(((unsigned long long)(((unsigned long long)(((short)(v_29))))))))));
                        v_6 = ((unsigned char)(helper_3(0u, 0)));
                        dw_1++;
                    } while ((dw_1 < 4));
                }
            }
        }
        unsigned int egres_0 = (__ffsll(0ull) ^ (GPU_SSHL(input[GPU_SMOD(tid, ((unsigned int)(n)))], (((input[GPU_SMOD(tid, ((unsigned int)(n)))] ^ (tid ^ ((unsigned int)(var_0)))) & 30583u) & 31u)) | GPU_SSHR((tid ^ ((unsigned int)(var_0))), (32u - (((input[GPU_SMOD(tid, ((unsigned int)(n)))] ^ (tid ^ ((unsigned int)(var_0)))) & 30583u) & 31u)))));
        _fv = (GPU_SMUL(_fv, 2654435761u) ^ (((unsigned int)(GPU_SMUL(egres_0, 2654435761u))) ^ GPU_SMUL(((unsigned int)(tid)), 2246822507u)));
        _fv = (_fv ^ ((unsigned int)(v_24)));
    }
    if ((tid < ((unsigned int)(n)))) {
        output[tid] = _fv;
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
        kernel_main<<<grid, block, 0>>>(d_output, d_input, n, 4112119587u, ((unsigned short)19169u), ((short)-17667));
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
