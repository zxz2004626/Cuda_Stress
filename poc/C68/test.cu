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

extern "C" __global__ __launch_bounds__(384, 1) void kernel_main(unsigned int* __restrict__ output, const unsigned int* __restrict__ input, int n, bool var_0, bool var_1, unsigned long long var_2) {
    unsigned int tid = (threadIdx.x + GPU_SMUL(blockIdx.x, blockDim.x));
    unsigned int _cse_0 = (((unsigned int)(((short)12951))) & 31u);
    unsigned int v_0 = ((unsigned int)((((unsigned long long)((GPU_SSHL(((unsigned int)(((((unsigned short)(tid)))))), _cse_0) ^ GPU_SSHR(GPU_SMOD(tid, 13u), _cse_0)))) % (((unsigned long long)((((long long)(GPU_SMOD(tid, 755492371u)))))) | 1ull))));
    unsigned char v_2 = (((unsigned char)(((unsigned long long)(((((unsigned int)((((unsigned int)(v_0))))))))))) - ((unsigned char)0u));
    long long v_3 = GPU_SSUB(0ll, 5367907040675418934ll);
    unsigned long long v_5 = (~(((unsigned long long)(blockIdx.x))));
    unsigned long long v_6 = (~((GPU_SSHL(0ull, ((unsigned long long)(gridDim.x))) ^ ((unsigned long long)(13u)))));
    unsigned int v_9 = (GPU_SMOD(((unsigned int)((12255360134829657984ull / (1ull | 1ull)))), ((unsigned int)((0ull / (((unsigned long long)(threadIdx.x)) | 1ull))))) ^ (GPU_SSHL(GPU_SSHR(gridDim.x, blockDim.x), 0u) ^ GPU_SSHR((((((unsigned char)(gridDim.x)) == v_2) ? 42ull : -384846986)), 0u)));
    unsigned long long v_10 = (((((unsigned char)(n)) > (v_2 & ((unsigned char)42u))) ? ((unsigned short)((((((((unsigned short)34440u) > ((unsigned short)56087u)) ? v_2 : ((unsigned short)12055u))))))) : ((unsigned char)0u)));
    long long v_22 = ((long long)((GPU_SMOD(((unsigned long long)(((unsigned int)((((signed char)(v_0))))))), ((unsigned long long)(GPU_SMUL(((unsigned long long)(((unsigned long long)(blockDim.x)))), ((unsigned long long)(v_6)))))) / (((v_5 & ((unsigned long long)(gridDim.x))) & ((unsigned long long)((((unsigned long long)(blockDim.x)) / (v_10 | 1ull))))) | 1ull))));
    unsigned int v_28 = ((unsigned int)(((signed char)(((((((unsigned short)(blockDim.x)) != ((unsigned short)0u)) ? ((unsigned char)0u) : -6557624890992476128ll)) % ((var_2 - 0ull) | 1ull))))));
    int v_29 = (0 ? ((signed char)0) : ((unsigned char)0u));
    unsigned int _cse_6 = (GPU_SDIV(blockDim.y, (~(3475032045u))) & 31u);
    unsigned long long v_31 = ((unsigned long long)((GPU_SSHL(((unsigned int)((v_5 / (GPU_SDIV(12784972066293982486ull, ((unsigned long long)(blockDim.x))) | 1ull)))), _cse_6) ^ GPU_SSHR(((unsigned int)(GPU_SMUL(((unsigned long long)(GPU_SSHR(v_28, v_9))), ((unsigned long long)((~(v_28))))))), _cse_6))));
    unsigned char v_36 = ((0) & ((unsigned char)0u));
    unsigned int v_38 = ((unsigned int)((GPU_SMOD((v_31 & ((unsigned long long)(blockDim.x))), (((((int)(gridDim.x)) >= v_29) ? ((unsigned char)127u) : ((signed char)1)))) % (((v_31 - v_10) - (((bool)(blockDim.x)))) | 1ull))));
    unsigned int _fv = input[GPU_SMOD(tid, ((unsigned int)(n)))];
    for (int _rep = 0; _rep < 1; _rep += 1) {
        if ((((short)(((int)(blockDim.y)))) >= (((-1125850072 < n) ? 349683356175876423ull : ((short)0))))) {
            if ((v_36 < ((unsigned char)(threadIdx.x)))) {
                {
                }
            } else {
                if ((v_22 != v_3)) {
                } else {
                    v_0 = (((long long)(GPU_SMUL(((unsigned long long)(((long long)(0ull)))), 0ull))));
                }
            }
        }
        _fv = (_fv ^ ((unsigned int)(v_0)));
        _fv = (_fv ^ ((unsigned int)(v_9)));
        _fv = (_fv ^ ((unsigned int)(v_38)));
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
    const int shared_bytes = 4096;

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
        kernel_main<<<grid, block, shared_bytes>>>(d_output, d_input, n, false, true, 5903491134938927595ull);
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
