// C68-min: minimal reproducer distilled from C68/test.cu.
//
// Trigger: ptxas -O2/-O3 constant-miscompilation (CUDA >= 12.2 toolchains).
// Every thread's output differs from the correct value by the same constant
// XOR (0xa6 for this file; 0x3b in test.cu's original configuration). All
// single terms dump identically at -O1 and -O2; the wrong value only appears
// when v_0/v_9/v_38 are combined by the three-XOR output chain, so the whole
// 3-XOR tree is required.
//
// Build & run (RTX 3090 / sm_86):
//   nvcc -std=c++17 -arch=sm_86 -w min.cu -o min && ./min
//   nvcc ... -Xptxas=-O2 min.cu -o min_O2 && ./min_O2
//
// Expected checksums (this exact file):
//   ptxas -O0 / -O1        : 13369739290192476610
//   ptxas -O2 / -O3        : 13609370734393575647
// (CUDA 12.2 and 13.3 verified, sm_86)
//
// Reduction notes (each remaining construct probed; touching it loses the
// differential):
//   - v_38  : the miscompiled lane-uniform constant term (every output wrong
//             by the constant XOR at -O2/-O3). It depends on v_31, the (u64)(u32)
//             truncation of ~(u64)blockIdx.x / ((12784972066293982486ull /
//             (u64)blockDim.x) | 1) with blockIdx.x = 0, plus the dead
//             `^ GPU_SSHR((u32)(GPU_SSHR(v_28,v_9) * ~v_28), 0)` term (always 0).
//   - v_28  : value is 0, but the convoluted definition itself is required
//             (replacing it with `0u` kills the trigger).
//   - v_10  : value is 0, but the definition (which reads `n`) is required.
//   - v_22  : feeds the `v_22 != 0` branch condition; the branch body is empty
//             and the guarded assignment inside never executes.
//   - v_0/v_2: per-lane terms XORed into the output; v_2 also selects in v_9.
//   - the three-XOR output tree must stay: any 1-term or 2-term output is
//     computed identically at -O1 and -O2.
// The kernel is UB-free; compute-sanitizer memcheck/initcheck/racecheck clean.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// UB-proof arithmetic (division/modulo guard, shift masking,
// signed-via-unsigned overflow)
#define GPU_SMOD(a, b) ((b) != 0 && !((__typeof__(b))(-1) < (__typeof__(b))(0) && (b) == (__typeof__(b))(-1) && (a) == (__typeof__(a))(1ULL << (sizeof(a) * 8 - 1))) ? (a) % (b) : 0)
#define GPU_SSHL(a, b) ((b) >= 0 && (b) < (long long)(sizeof(a) * 8) ? ((unsigned long long)(a) << (b)) : 0ULL)
#define GPU_SSHR(a, b) ((b) >= 0 && (b) < (long long)(sizeof(a) * 8) ? (a) >> (b) : 0)
#define GPU_SMUL(a, b) ((__typeof__((a) + (b)))((unsigned long long)(a) * (unsigned long long)(b)))

#define GPU_CUDA_CHECK(call) do { cudaError_t _gpu_err = (call); if (_gpu_err != cudaSuccess) { fprintf(stderr, "CUDA_ERROR %s:%d: %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(_gpu_err)); return 2; } } while (0)

__global__ void kernel_main(unsigned int* output, const unsigned int* input, int n, unsigned long long var_2) {
    unsigned int tid = (threadIdx.x + GPU_SMUL(blockIdx.x, blockDim.x));
    unsigned int v_0 = ((unsigned int)(((((((unsigned long long)(tid & 0xFFFFu)) << 23u) ^ ((tid % 13u) >> 23u))) % (((unsigned long long)((((long long)((tid % 755492371u)))))) | 1ull))));
    unsigned char v_2 = ((unsigned char)(v_0));
    unsigned long long v_5 = (~(((unsigned long long)(blockIdx.x))));
    unsigned long long v_6 = (~((unsigned long long)13u));
    unsigned int v_9 = ((unsigned int)0u ^ (GPU_SSHL(GPU_SSHR(gridDim.x, blockDim.x), 0u) ^ GPU_SSHR((((((unsigned char)(gridDim.x)) == v_2) ? 42ull : -384846986)), 0u)));
    unsigned long long v_10 = (((((unsigned char)(n)) > (v_2 & ((unsigned char)42u))) ? ((unsigned short)((((((((unsigned short)34440u) > ((unsigned short)56087u)) ? v_2 : ((unsigned short)12055u))))))) : ((unsigned char)0u)));
    long long v_22 = ((long long)((GPU_SMOD(((unsigned long long)(((unsigned int)((((signed char)(v_0))))))), ((unsigned long long)(GPU_SMUL(((unsigned long long)(((unsigned long long)(blockDim.x)))), ((unsigned long long)(v_6)))))) / (((v_5 & ((unsigned long long)(gridDim.x))) & ((unsigned long long)((((unsigned long long)(blockDim.x)) / (v_10 | 1ull))))) | 1ull))));
    unsigned int v_28 = ((unsigned int)(((signed char)(((((((unsigned short)(blockDim.x)) != ((unsigned short)0u)) ? ((unsigned char)0u) : -6557624890992476128ll)) % ((var_2 - 0ull) | 1ull))))));
    unsigned long long v_31 = ((unsigned long long)((GPU_SSHL(((unsigned int)((v_5 / ((12784972066293982486ull / (unsigned long long)blockDim.x) | 1ull)))), 0u) ^ GPU_SSHR(((unsigned int)(GPU_SMUL(((unsigned long long)(GPU_SSHR(v_28, v_9))), ((unsigned long long)((~(v_28))))))), 0u))));
    unsigned int v_38 = ((unsigned int)((v_31 & 255u) % (((v_31 - v_10) - 1ull) | 1ull)));
    unsigned int _fv = input[tid];
    if (((unsigned char)0u < ((unsigned char)(threadIdx.x)))) {
        {
        }
    } else {
        if ((v_22 != 0)) {
        } else {
            v_0 = (((long long)(GPU_SMUL(((unsigned long long)(((long long)(0ull)))), 0ull))));
        }
    }
    _fv = (_fv ^ ((unsigned int)(v_0)));
    _fv = (_fv ^ ((unsigned int)(v_9)));
    _fv = (_fv ^ ((unsigned int)(v_38)));
    output[tid] = _fv;
}

int main(void) {
    const int n = 8;
    const int buf_size = n;
    const int in_size = buf_size;

    // Host allocations
    unsigned int *h_input = (unsigned int*)malloc(in_size * sizeof(unsigned int));
    unsigned int *h_output = (unsigned int*)malloc(buf_size * sizeof(unsigned int));
    if (!h_input || !h_output) { fprintf(stderr, "malloc failed\n"); return 1; }

    // Deterministic input initialization (dist=hash) happens in run_pass()
    // Device allocations
    unsigned int *d_input = NULL, *d_output = NULL;
    GPU_CUDA_CHECK(cudaMalloc((void**)&d_input, in_size * sizeof(unsigned int)));
    GPU_CUDA_CHECK(cudaMalloc((void**)&d_output, buf_size * sizeof(unsigned int)));

    // Launch configuration
    dim3 grid(1);
    dim3 block(8);

    // Checksum-only mode: skip all warmup and timing launches

    // Deterministic checksum pass (run_pass is reusable in-process)
    auto run_pass = [&](unsigned long long salt) -> int {
        for (int i = 0; i < in_size; i++) {
            h_input[i] = (unsigned int)i;
        }
        GPU_CUDA_CHECK(cudaMemcpy(d_input, h_input, in_size * sizeof(unsigned int), cudaMemcpyHostToDevice));
        GPU_CUDA_CHECK(cudaMemset(d_output, 0, buf_size * sizeof(unsigned int)));
        kernel_main<<<grid, block>>>(d_output, d_input, n, 5903491134938927595ull);
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

    run_pass(0ULL);

    // Cleanup
    GPU_CUDA_CHECK(cudaFree(d_input));
    GPU_CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output);
    return 0;
}
