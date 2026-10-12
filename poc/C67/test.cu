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

extern "C" __global__ __launch_bounds__(512, 2) void kernel_main(unsigned int* __restrict__ output, const unsigned int* __restrict__ input, int n, unsigned char var_0, unsigned long long var_1, long long var_2) {
    unsigned int tid = (threadIdx.x + GPU_SMUL(blockIdx.x, blockDim.x));
    unsigned int _cse_1 = (1892112825u & 31u);
    unsigned long long v_0 = ((((signed char)0) == ((signed char)-63)) ? 0ull : ((unsigned long long)((((unsigned long long)((GPU_SSHL(tid, _cse_1) ^ GPU_SSHR(tid, _cse_1)))) / (GPU_SSHR(8617359382650639998ull, 9223372036854775807ull) | 1ull)))));
    long long v_16 = (((long long)((((unsigned long long)(GPU_SMUL(((unsigned long long)(var_1)), ((unsigned long long)(9223372036854775807ull))))) / ((~(var_1)) | 1ull)))) | ((long long)((~(2610840123u)))));
    unsigned int _cse_2 = (tid & 31u);
    unsigned int v_18 = (GPU_SSHL((GPU_SSHL(4119396644u, _cse_2) ^ GPU_SSHR(((unsigned int)(GPU_SMUL(((unsigned long long)(tid)), ((unsigned long long)(3991717137u))))), _cse_2)), (GPU_SMUL(tid, ((unsigned int)((v_0 / (v_0 | 1ull))))) & 31u)) ^ 0u);
    signed char v_25 = ((signed char)((GPU_SSHL(0u, 0u) ^ GPU_SSHR(((2303288736u + 0u) - (blockIdx.y ^ v_18)), 0u))));
    int v_26 = (~(GPU_SSHR(GPU_SSUB(0, n), ((int)((16426438756286437730ull % (17368068424718127389ull | 1ull)))))));
    unsigned short v_30 = (((((((int)(gridDim.y)) < n) ? v_16 : 0)) | 0ll));
    unsigned short v_31 = (GPU_SSUB(((unsigned short)(__ballot_sync(4294967295u, (v_25 > ((signed char)-61))))), (~(GPU_SSUB(0, v_26)))));
    unsigned int _fv = input[GPU_SMOD(tid, ((unsigned int)(n)))];
    for (int _rep = 0; _rep < 1; _rep += 1) {
        _fv = (_fv ^ ((unsigned int)(v_30)));
        _fv = (_fv ^ ((unsigned int)(v_31)));
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
        kernel_main<<<grid, block, 0>>>(d_output, d_input, n, ((unsigned char)18u), 16694769789140031405ull, 5903491134938927614ll);
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
