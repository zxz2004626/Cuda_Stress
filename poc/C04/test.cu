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

extern "C" __global__ __launch_bounds__(384, 1) void kernel_main(unsigned int* __restrict__ output, const unsigned int* __restrict__ input, int n, unsigned short var_0, unsigned char var_1, unsigned char var_2) {
    unsigned int tid = (threadIdx.x + GPU_SMUL(blockIdx.x, blockDim.x));
    unsigned long long v_5 = (0ull + (-(((unsigned long long)(((signed char)63))))));
    unsigned int _cse_0 = (GPU_SSHR(tid, threadIdx.x) & 31u);
    short v_9 = GPU_SSUB(((short)0), ((short)((GPU_SSHL(((unsigned int)(v_5)), _cse_0) ^ 0u))));
    short v_30 = GPU_SSUB(GPU_SSUB(((short)0), (((short)1024) | v_9)), GPU_SADD(((short)(0u)), ((short)0)));
    unsigned int v_42 = (((unsigned int)(0)) ^ ((unsigned int)((((tid < 1893709643u) ? 42ull : 0ull) / (0ull | 1ull)))));
    unsigned long long v_43 = ((unsigned long long)(__funnelshift_r(0u, ((unsigned int)(((v_9 < ((short)9029)) ? GPU_SMUL(gridDim.x, 0u) : (0u + v_42)))), ((unsigned int)(19u)))));
    int v_44 = (0 | 0);
    short v_46 = GPU_SSUB(((((unsigned long long)(blockIdx.x)) < 8825242320755776291ull) ? ((v_44 >= ((int)(gridDim.y))) ? v_30 : ((short)-32768)) : ((short)0)), ((short)0));
    unsigned int _fv = input[GPU_SMOD(tid, ((unsigned int)(n)))];
    for (int _rep = 0; _rep < 1; _rep += 1) {
        _fv = (_fv ^ ((unsigned int)(v_43)));
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
        kernel_main<<<grid, block, shared_bytes>>>(d_output, d_input, n, ((unsigned short)63242u), ((unsigned char)21u), ((unsigned char)24u));
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
