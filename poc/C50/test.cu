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

extern "C" __global__ __launch_bounds__(512, 2) void kernel_main(unsigned int* __restrict__ output, const unsigned int* __restrict__ input, int n, long long var_0, long long var_1, long long var_2) {
    unsigned int tid = (threadIdx.x + GPU_SMUL(blockIdx.x, blockDim.x));
    unsigned int _cse_0 = (0u & 31u);
    int v_3 = ((int)((GPU_SSHL(tid, _cse_0) ^ GPU_SSHR(0u, _cse_0))));
    bool v_21 = 0;
    short v_29 = GPU_SSUB(((short)0), ((short)((GPU_SSHL((((v_3 != 0) ? 31u : 9223372036854775807ull)), 0u) ^ GPU_SSHR((~(tid)), 0u)))));
    unsigned int v_43 = ((((((unsigned char)0u) < ((unsigned char)0u)) ? ((unsigned short)0u) : ((unsigned short)0u)) >= (((unsigned short)0u) + ((unsigned short)0u))) ? ((((short)1) > v_29) ? ((((int)(blockDim.x)) < 7) ? v_29 : v_29) : ((short)(__ldg((&(input[GPU_SMOD(tid, ((unsigned int)(n)))])))))) : (((((unsigned short)0u) > ((unsigned short)0u)) ? 0ll : 0)));
    unsigned int v_45 = ((~((4294967295u ^ 0))) | GPU_SDIV(((signed char)(((unsigned short)1u))), ((signed char)(18446744073709551615ull))));
    unsigned int _cse_7 = (37u & 31u);
    int v_46 = ((((int)(((((unsigned char)(threadIdx.x)) > ((unsigned char)(gridDim.x))) ? v_45 : threadIdx.x))) & (((7u <= tid) ? var_2 : 0ll) | ((long long)((GPU_SSHL(37u, _cse_7) ^ GPU_SSHR(tid, _cse_7)))))));
    unsigned int _fv = input[GPU_SMOD(tid, ((unsigned int)(n)))];
    for (int _rep = 0; _rep < 1; _rep += 1) {
        unsigned int eg_la_0[8];
        unsigned int eg_acc_0 = ((eg_la_0[0] ^ eg_la_0[3]) ^ eg_la_0[7]);
        _fv = (GPU_SMUL(_fv, 2654435761u) ^ ((unsigned int)(eg_acc_0)));
        switch (((GPU_SSHL((~(7u)), (((2863311530u == v_43) ? blockIdx.x : v_43) & 31u)) ^ GPU_SSHR((~(blockIdx.y)), (((2863311530u == v_43) ? blockIdx.x : v_43) & 31u))) % 4u)) {
            case 0: {
                break;
            }
            case 1: {
                break;
            }
            case 2: {
                break;
            }
            case 3: {
                for (int i_6 = 0; i_6 < ((((int)((-(gridDim.y)))) & 3) + 1); i_6 += 1) {
                    unsigned int _cse_8 = (((unsigned int)(v_46)) & 31u);
                    unsigned int _cse_9 = (v_43 & 31u);
                    v_21 = ((bool)((GPU_SSHL(((unsigned int)(((unsigned short)0u))), _cse_8) ^ GPU_SSHR((GPU_SSHL(1431655765u, _cse_9) ^ GPU_SSHR(3u, _cse_9)), _cse_8))));
                }
                break;
            }
            default: {
                break;
            }
        }
        _fv = (_fv ^ ((unsigned int)(v_21)));
        _fv = (_fv ^ ((unsigned int)(eg_acc_0)));
    }
    if ((tid < ((unsigned int)(n)))) {
        output[tid] = _fv;
    }
}

int main(void) {
    const int n = 32768;
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
    dim3 grid(128);
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
        kernel_main<<<grid, block, 0>>>(d_output, d_input, n, 9039304369631583614ll, 7471397752285255614ll, 5903491134938927614ll);
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
