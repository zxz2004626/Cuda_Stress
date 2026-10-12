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

extern "C" {
    __device__ long long arr_3[13][13][13];
}

extern "C" __global__ __launch_bounds__(256, 1) void kernel_main(unsigned int* __restrict__ output, const unsigned int* __restrict__ input, int n, bool var_0, long long var_1, unsigned long long var_2) {
    unsigned int tid = (threadIdx.x + GPU_SMUL(blockIdx.x, blockDim.x));
    signed char v_0 = GPU_SSUB(((signed char)0), ((signed char)0));
    long long v_1 = ((long long)((((unsigned int)(__brev(((unsigned int)(((unsigned int)((((unsigned long long)(arr_3[GPU_SMOD(tid, 13u)][GPU_SMOD(37u, 13u)][GPU_SMOD(0u, 13u)])) % ((-(((unsigned long long)(blockDim.x)))) | 1ull))))))))) ^ ((unsigned int)(0ull)))));
    unsigned long long v_11 = (((unsigned long long)(GPU_GROUP_DIVERGE(blockIdx.x, ((unsigned int)(arr_3[GPU_SMOD(2u, 13u)][GPU_SMOD(blockIdx.x, 13u)][GPU_SMOD(tid, 13u)]))))) + (-(GPU_SMUL(42ull, ((unsigned long long)(gridDim.x))))));
    long long v_14 = ((long long)((255u ^ 252645135u)));
    signed char v_19 = (((signed char)0) & ((signed char)0));
    long long v_21 = ((long long)(GPU_SMUL(((unsigned long long)(v_1)), ((unsigned long long)(GPU_SSUB(0ll, ((long long)(__byte_perm(gridDim.x, 37u, 0u)))))))));
    unsigned int v_22 = GPU_GROUP_DIVERGE(blockIdx.x, __ldg((&(input[GPU_SMOD(tid, ((unsigned int)(n)))]))));
    unsigned int _cse_1 = (gridDim.x & 31u);
    unsigned int _cse_2 = (v_22 & 31u);
    unsigned long long v_24 = ((((unsigned long long)((GPU_SSHL(gridDim.x, _cse_1) ^ GPU_SSHR(13u, _cse_1)))) ^ ((255 > n) ? 1ull : 18446744073709551615ull)) | ((unsigned long long)(GPU_GROUP_DIVERGE(blockIdx.x, (GPU_SSHL(v_22, _cse_2) ^ GPU_SSHR(255u, _cse_2))))));
    unsigned int v_25 = ((unsigned int)((((((unsigned short)(threadIdx.x)) >= ((unsigned short)0u)) ? ((bool)(blockIdx.x)) : ((bool)(threadIdx.x))) - ((bool)(((0u)))))));
    unsigned int v_31 = (~((((unsigned int)((9223372036854775807ull / (v_11 | 1ull)))) + (0u & threadIdx.y))));
    unsigned long long v_34 = (((((signed char)(GPU_SMUL(((unsigned long long)(((signed char)0))), ((unsigned long long)(v_0))))) < ((signed char)(((unsigned int)(__clz(((unsigned int)(v_22)))))))) ? (gridDim.x ^ ((((signed char)-1) < ((signed char)(threadIdx.x))) ? blockIdx.x : 0u)) : ((((signed char)-1) ^ v_19) & (~(v_0)))));
    unsigned int v_35 = (((((v_14 <= v_21) ? v_25 : ((signed char)0))) > ((bool)((var_2 / (v_34 | 1ull))))) ? (GPU_SSHL(((unsigned int)(((unsigned char)50u))), (((unsigned int)(((unsigned short)((2u))))) & 31u)) ^ GPU_SSHR(((unsigned int)(((unsigned short)0u))), (((unsigned int)(((unsigned short)(((short)0))))) & 31u))) : 0u);
    unsigned long long v_37 = v_24;
    unsigned int _fv = input[GPU_SMOD(tid, ((unsigned int)(n)))];
    for (int _rep = 0; _rep < 1; _rep += 1) {
        _fv = (_fv ^ ((unsigned int)(v_31)));
        _fv = (_fv ^ ((unsigned int)(v_35)));
        _fv = (_fv ^ ((unsigned int)(v_37)));
    }
    if ((tid < ((unsigned int)(n)))) {
        output[tid] = _fv;
    }
}

int main(void) {
    const int n = 22528;
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

    // Zero-initialize __device__ arrays for determinism
    {
        static char _zeros_arr_3[2197 * sizeof(long long)];
        memset(_zeros_arr_3, 0, 2197 * sizeof(long long));
        GPU_CUDA_CHECK(cudaMemcpyToSymbol(arr_3, _zeros_arr_3, 2197 * sizeof(long long)));
    }

    // Launch configuration
    dim3 grid(88);
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
        kernel_main<<<grid, block, 0>>>(d_output, d_input, n, false, 7471397752285255614ll, 5903491134938927595ull);
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
