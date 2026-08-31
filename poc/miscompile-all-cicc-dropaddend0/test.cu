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
    __device__ bool arr_0[7];
    __device__ int arr_1[7];
    __device__ unsigned int arr_2[7];
    __device__ signed char arr_3[7][7];
    __device__ unsigned char arr_4[16];
    __device__ unsigned long long arr_5[11][11];
}

__device__ unsigned int helper_0(int p0, unsigned int p1, unsigned int p2) {
    if ((((unsigned int)(__brev(((unsigned int)(blockIdx.x))))) > ((unsigned int)(__popc(((unsigned int)(p1))))))) {
        p1 = (((unsigned int)(1257290146)) + GPU_SSHR(255u, p1));
    }
    if ((((int)(blockDim.x)) < (-2 | p0))) {
        p1 = GPU_SSHR(GPU_SSHL(p1, threadIdx.x), (1146876176u ^ p2));
    } else {
        p2 = ((4294967295u | 3890289086u) & ((2u == p2) ? p2 : 2039404531u));
    }
    p0 = ((((int)(__funnelshift_r(((unsigned int)(3958058115u)), ((unsigned int)(1196703580u)), ((unsigned int)(6u))))) ^ ((unsigned int)(__popcll(((unsigned long long)(9223372036854775807ull)))))));
    p0 = ((int)((0ull % (((unsigned long long)(GPU_GROUP_DIVERGE(blockIdx.x, p2))) | 1ull))));
    unsigned int v_0 = ((unsigned int)((((unsigned long long)((1ull / (0ull | 1ull)))) % ((((unsigned long long)((3580748498724636866ull % (12784548627349405984ull | 1ull)))) | (blockDim.x & threadIdx.x)) | 1ull))));
    p1 -= ((GPU_SSHL(p1, (2u & 31u)) ^ GPU_SSHR(37u, (2u & 31u))) + (v_0 + p2));
    if (((2681779396u ^ p2) > ((unsigned int)(p0)))) {
        ((unsigned int)(__ffs(((unsigned int)(((unsigned int)((42ull / (42ull | 1ull)))))))));
        if ((gridDim.y >= p1)) {
            v_0 = GPU_SMUL((((unsigned int)(__ffsll(((unsigned long long)(18446744073709551615ull))))) ^ ((unsigned int)(__clzll(((unsigned long long)(42ull)))))), __byte_perm(v_0, blockIdx.x, 2u));
        }
    }
    if ((threadIdx.x > blockIdx.x)) {
        p2 = ((v_0 ^ p2) + 2414426764u);
    }
    ((int)((((unsigned int)(__brevll(((unsigned long long)((9223372036854775807ull | ((int)(blockDim.y)))))))) ^ ((unsigned int)(__popcll(((unsigned long long)(((unsigned long long)((14147153419636440816ull % (((unsigned long long)(blockDim.x)) | 1ull))))))))))));
    return (GPU_SMUL((GPU_SMUL((GPU_SMUL((GPU_SMUL(GPU_SDIV(((p2 <= blockDim.y) ? GPU_SMOD(p1, 2654435769u) : ((848763119 >= 255) ? p1 : v_0)), blockIdx.y), 2654435761u) ^ ((unsigned int)(p0))), 2654435761u) ^ ((unsigned int)(p1))), 2654435761u) ^ ((unsigned int)(p2))), 2654435761u) ^ ((unsigned int)(v_0)));
}

__forceinline__ __device__ unsigned int helper_1(unsigned int p0, unsigned int p1, int p2) {
    p1 = GPU_SMUL((p1 - p1), GPU_SMUL(3918366792u, p1));
    p0 = gridDim.y;
    if (((p0 + p0) == (GPU_SSHL(2654435769u, (2941440455u & 31u)) ^ GPU_SSHR(3614115936u, (2941440455u & 31u))))) {
        p0 = (p0 ^ (__ffsll((((unsigned long long)((GPU_SMUL(p1, p0) + GPU_FAKE_DIVERGE(threadIdx.x, threadIdx.x))))))));
        (GPU_SSHL((255u ^ p1), ((p1 ^ 3269487175u) & 31u)) ^ GPU_SSHR(GPU_SMOD(p0, 255u), ((p1 ^ 3269487175u) & 31u)));
        if ((1989844171u < p1)) {
            if ((threadIdx.x >= blockDim.x)) {
                unsigned int _dead_0 = 3735936685u;
                _dead_0 = blockDim.y;
            }
        }
    }
    unsigned int v_0 = GPU_GROUP_DIVERGE(blockIdx.x, (0u & GPU_GROUP_DIVERGE(blockIdx.x, p0)));
    p0 = 2u;
    int v_1 = GPU_SDIV((((int)((2748143973u + GPU_FAKE_DIVERGE(threadIdx.x, threadIdx.x)))) & GPU_SADD(((int)(threadIdx.x)), ((int)(blockIdx.x)))), ((int)(GPU_GROUP_DIVERGE(blockIdx.x, (4294967295u & 213020132u)))));
    __threadfence_block();
    p2 = 7;
    return (GPU_SMUL((GPU_SMUL((GPU_SMUL((GPU_SMUL((GPU_SMUL(p0, 2654435761u) ^ ((unsigned int)(p0))), 2654435761u) ^ ((unsigned int)(p1))), 2654435761u) ^ ((unsigned int)(p2))), 2654435761u) ^ ((unsigned int)(v_0))), 2654435761u) ^ ((unsigned int)(v_1)));
}

extern "C" __global__ __launch_bounds__(768, 1) void kernel_main(unsigned int* __restrict__ output, const unsigned int* __restrict__ input, int n, int var_0, unsigned int var_1, unsigned int var_2) {
    unsigned int _fv = 0u;
    unsigned int tid = (threadIdx.x + GPU_SMUL(blockIdx.x, blockDim.x));
    unsigned int x = (GPU_SMUL(blockIdx.x, blockDim.x) + threadIdx.x);
    unsigned int y = (GPU_SMUL(blockIdx.y, blockDim.y) + threadIdx.y);
    unsigned int idx = (GPU_SMUL(y, ((unsigned int)(n))) + x);
    unsigned int cv_1 = 0u;
    unsigned int cv_4 = 0u;
    unsigned int cv_8 = 0u;
    unsigned int cv_9 = 0u;
    unsigned int cv_10 = 0u;
    unsigned int cv_11 = 0u;
    unsigned long long cv_12 = 0ull;
    unsigned int cv_14 = 0u;
    unsigned long long cv_15 = 0ull;
    unsigned int cv_17 = 0u;
    unsigned int cv_18 = 0u;
    unsigned int cv_22 = 0u;
    unsigned int _lc_0 = (GPU_SMUL(tid, 2654435761u) ^ 3741039821u);
    unsigned int _lc_1 = (GPU_SMUL(tid, 1013904226u) ^ 3582195659u);
    unsigned int _lc_2 = (GPU_SMUL(tid, 3668339987u) ^ 2994447647u);
    unsigned long long _lc_3 = ((unsigned long long)((GPU_SMUL(tid, 2027808452u) ^ 254606082u)));
    unsigned short _lc_4 = ((unsigned short)((GPU_SMUL(tid, 387276917u) ^ 3297522989u)));
    unsigned char _lc_5 = ((unsigned char)((GPU_SMUL(tid, 3041712678u) ^ 1449114023u)));
    unsigned int _lc_6 = (GPU_SMUL(tid, 1401181143u) ^ 4004484856u);
    unsigned int _lc_7 = (GPU_SMUL(tid, 4055616904u) ^ 3490147386u);
    unsigned int _lc_8 = (GPU_SMUL(tid, 2415085369u) ^ 789068941u);
    unsigned long long _lc_9 = ((unsigned long long)((GPU_SMUL(tid, 774553834u) ^ 150615275u)));
    unsigned short _lc_10 = ((unsigned short)((GPU_SMUL(tid, 3428989595u) ^ 1217254689u)));
    unsigned char _lc_11 = ((unsigned char)((GPU_SMUL(tid, 1788458060u) ^ 3854913976u)));
    for (int _rep = 0; _rep < 3; _rep += 1) {
        unsigned int cv_0 = ((unsigned int)(tid));
        unsigned int dsw_0 = (tid ^ 3735928559u);
        switch ((tid & 3u)) {
            case 0: {
                dsw_0 = (dsw_0 ^ (GPU_SMUL(dsw_0, 2654435769u) ^ GPU_SSHR(dsw_0, 16u)));
                break;
            }
            case 1: {
                dsw_0 = (dsw_0 ^ GPU_SMUL((dsw_0 ^ 2246822507u), 3266489909u));
                break;
            }
            case 2: {
                dsw_0 = (dsw_0 ^ (dsw_0 + (GPU_SSHL(dsw_0, 13u) ^ GPU_SSHR(dsw_0, 7u))));
                break;
            }
            case 3: {
                dsw_0 = (dsw_0 ^ ((GPU_SSHL(dsw_0, 7u) | GPU_SSHR(dsw_0, 25u)) ^ 668265263u));
                break;
            }
            default: {
                dsw_0 = (dsw_0 + 1367130551u);
                break;
            }
        }
        unsigned int dsw_1 = (tid ^ 3735928559u);
        switch ((tid & 3u)) {
            case 0: {
                dsw_1 = (dsw_1 ^ (GPU_SMUL(dsw_1, 2654435769u) ^ GPU_SSHR(dsw_1, 16u)));
                break;
            }
            case 1: {
                dsw_1 = (dsw_1 ^ GPU_SMUL((dsw_1 ^ 2246822507u), 3266489909u));
                break;
            }
            case 2: {
                dsw_1 = (dsw_1 ^ (dsw_1 + (GPU_SSHL(dsw_1, 13u) ^ GPU_SSHR(dsw_1, 7u))));
                break;
            }
            case 3: {
                dsw_1 = (dsw_1 ^ ((GPU_SSHL(dsw_1, 7u) | GPU_SSHR(dsw_1, 25u)) ^ 668265263u));
                break;
            }
            default: {
                dsw_1 = (dsw_1 + 1367130551u);
                break;
            }
        }
        _lc_8 = ((((unsigned int)(_lc_8)) ^ helper_0(((int)(((unsigned int)(_lc_9)))), ((unsigned int)(((unsigned int)(_lc_10)))), ((unsigned int)(((unsigned int)(_lc_11)))))) ^ ((unsigned int)(cv_8)));
        _fv = ((((((((((((GPU_SMUL(_fv, 2654435761u) ^ ((unsigned int)(_lc_0))) ^ ((unsigned int)(_lc_1))) ^ ((unsigned int)(_lc_2))) ^ ((unsigned int)(_lc_3))) ^ ((unsigned int)(_lc_4))) ^ ((unsigned int)(_lc_5))) ^ ((unsigned int)(_lc_6))) ^ ((unsigned int)(_lc_7))) ^ ((unsigned int)(_lc_8))) ^ ((unsigned int)(_lc_9))) ^ ((unsigned int)(_lc_10))) ^ ((unsigned int)(_lc_11)));
    }
    if (((x < ((unsigned int)(n))) && (y < ((unsigned int)(n))))) {
        output[idx] = (output[idx] ^ _fv);
    }
}

int main(void) {
    const int n = 128;
    const int buf_size = n * n;
    const int in_size = buf_size;

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
        static char _zeros_arr_0[7 * sizeof(bool)];
        memset(_zeros_arr_0, 0, 7 * sizeof(bool));
        GPU_CUDA_CHECK(cudaMemcpyToSymbol(arr_0, _zeros_arr_0, 7 * sizeof(bool)));
    }
    {
        static char _zeros_arr_1[7 * sizeof(int)];
        memset(_zeros_arr_1, 0, 7 * sizeof(int));
        GPU_CUDA_CHECK(cudaMemcpyToSymbol(arr_1, _zeros_arr_1, 7 * sizeof(int)));
    }
    {
        static char _zeros_arr_2[7 * sizeof(unsigned int)];
        memset(_zeros_arr_2, 0, 7 * sizeof(unsigned int));
        GPU_CUDA_CHECK(cudaMemcpyToSymbol(arr_2, _zeros_arr_2, 7 * sizeof(unsigned int)));
    }
    {
        static char _zeros_arr_3[49 * sizeof(signed char)];
        memset(_zeros_arr_3, 0, 49 * sizeof(signed char));
        GPU_CUDA_CHECK(cudaMemcpyToSymbol(arr_3, _zeros_arr_3, 49 * sizeof(signed char)));
    }
    {
        static char _zeros_arr_4[16 * sizeof(unsigned char)];
        memset(_zeros_arr_4, 0, 16 * sizeof(unsigned char));
        GPU_CUDA_CHECK(cudaMemcpyToSymbol(arr_4, _zeros_arr_4, 16 * sizeof(unsigned char)));
    }
    {
        static char _zeros_arr_5[121 * sizeof(unsigned long long)];
        memset(_zeros_arr_5, 0, 121 * sizeof(unsigned long long));
        GPU_CUDA_CHECK(cudaMemcpyToSymbol(arr_5, _zeros_arr_5, 121 * sizeof(unsigned long long)));
    }

    // Launch configuration
    dim3 grid(8, 8);
    dim3 block(16, 16);

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
        kernel_main<<<grid, block, 0>>>(d_output, d_input, n, 1964635965, 1218333439u, 2619514586u);
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
