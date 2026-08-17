// Reduced from:
//   poc/cuda-11.8_miscompile/generator_ir__35d576a5d6f55918/test.cu
//
// Cross-toolchain result:
//   cuda-10.1 / cuda-11.4, ptxas -O0..-O3 : checksum=1
//   cuda-11.8,               ptxas -O0     : checksum=1
//   cuda-11.8,               ptxas -O1..-O3: checksum=0
//
// The kernel deliberately keeps several otherwise-dead values around.  They are
// part of the minimal PTX scheduling/register context that makes ptxas 11.8 O1
// miscompile v_36 below; deleting or simplifying most of them makes the bug
// disappear.  v_36 is the only value that differs between O0 and O1.
//
#include <stdio.h>
#include <cuda_runtime.h>

// UB-proof arithmetic (division/modulo guard, shift masking,
// signed-via-unsigned overflow)
#define GPU_SDIV(a, b) ((b) != 0 && !((__typeof__(b))(-1) < (__typeof__(b))(0) && (b) == (__typeof__(b))(-1) && (a) == (__typeof__(a))(1ULL << (sizeof(a) * 8 - 1))) ? (a) / (b) : 0)
#define GPU_SMOD(a, b) ((b) != 0 && !((__typeof__(b))(-1) < (__typeof__(b))(0) && (b) == (__typeof__(b))(-1) && (a) == (__typeof__(a))(1ULL << (sizeof(a) * 8 - 1))) ? (a) % (b) : 0)
#define GPU_SSHL(a, b) ((b) >= 0 && (b) < (long long)(sizeof(a) * 8) ? ((unsigned long long)(a) << (b)) : 0ULL)
#define GPU_SSHR(a, b) ((b) >= 0 && (b) < (long long)(sizeof(a) * 8) ? (a) >> (b) : 0)
#define GPU_SSUB(a, b) ((__typeof__((a) + (b)))((unsigned long long)(a) - (unsigned long long)(b)))

// Divergence-illusion macros
#define GPU_GROUP_DIVERGE(idx, val) ((idx) * 0 + (val))

#define GPU_CUDA_CHECK(call) do { cudaError_t _gpu_err = (call); if (_gpu_err != cudaSuccess) { fprintf(stderr, "CUDA_ERROR %s:%d: %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(_gpu_err)); return 2; } } while (0)


extern "C" __global__ __launch_bounds__(384, 2) void kernel_main(unsigned int* __restrict__ output, const unsigned int* __restrict__ input, int n, bool var_0, bool var_1, unsigned long long var_2) {
    unsigned int tid = (threadIdx.x + (blockIdx.x * blockDim.x));
    unsigned int ov_0 = 0u;
    signed char ov_1 = ((signed char)0);
    unsigned char ov_2 = ((unsigned char)0u);
    int ov_3 = 0;
    long long ov_4 = 0ll;
    bool ov_5 = false;
    unsigned long long ov_6 = 0ull;
    int ov_7 = 0;
    long long v_0 = ((long long)(gridDim.x));
    bool v_1 = ((bool)((208952228u ^ ((unsigned int)((((unsigned long long)(GPU_SMOD(tid, 2654435769u))) * ((unsigned long long)(tid))))))));
    unsigned char v_2 = ((unsigned char)0u);
    unsigned long long v_3 = 0ull;
    unsigned int v_4 = 0u;
    unsigned short v_5 = ((unsigned short)0u);
    unsigned char v_6 = ((unsigned char)0u);
    int v_7 = 0;
    long long v_8 = (~(GPU_SSHR(((long long)(((unsigned char)167u))), GPU_SDIV(-7293850184320197ll, v_0))));
    unsigned short v_9 = ((unsigned short)(GPU_GROUP_DIVERGE(blockIdx.x, GPU_SSHR((1431655765u ^ tid), GPU_SDIV(blockIdx.x, gridDim.x)))));
    unsigned short v_10 = (((GPU_SDIV(v_9, ((unsigned short)(threadIdx.x))) <= ((unsigned short)(GPU_GROUP_DIVERGE(blockIdx.x, 3u)))) ? ((v_9 == ((unsigned short)0u)) ? (((short)1)) : GPU_SMOD(-5254999677867705437ll, v_0)) : GPU_SSHL(GPU_SDIV(((unsigned char)(blockIdx.x)), ((unsigned char)(gridDim.x))), v_6)));
    signed char v_11 = ((signed char)0);
    unsigned long long v_12 = (((tid > tid) ? ((((unsigned char)(gridDim.x)) == v_2) ? var_2 : v_3) : ((unsigned long long)(__ldg((&(input[GPU_SMOD(tid, ((unsigned int)(n)))])))))) | GPU_SDIV((v_3 ^ var_2), ((unsigned long long)(gridDim.x))));
    int v_13 = 0;
    short v_14 = ((short)0);
    long long v_15 = 0ll;
    int v_16 = ((int)((((unsigned long long)(((int)((((unsigned long long)(GPU_SDIV(42, ((int)(threadIdx.y))))) * ((unsigned long long)(((int)((((unsigned long long)(v_7)) * ((unsigned long long)(69866620)))))))))))) * ((unsigned long long)(GPU_SDIV(((int)(((unsigned short)((v_4))))), ((int)(__ldg((&(input[GPU_SMOD(tid, ((unsigned int)(n)))])))))))))));
    unsigned int v_17 = __ldg((&(input[GPU_SMOD(tid, ((unsigned int)(n)))])));
    bool v_18 = ((bool)((((unsigned long long)((~(((((short)42) == ((short)13569)) ? true : var_1))))) * ((unsigned long long)(var_0)))));
    unsigned char v_19 = ((unsigned char)0u);
    short v_20 = ((short)0);
    unsigned long long v_21 = 0ull;
    unsigned short v_22 = ((unsigned short)0u);
    unsigned char v_23 = ((((signed char)(((short)(v_17)))) >= GPU_SSUB(((signed char)0), v_11)) ? ((unsigned char)((((unsigned long long)(((unsigned char)(GPU_GROUP_DIVERGE(blockIdx.x, v_4))))) * ((unsigned long long)(((unsigned char)(__ldg((&(input[GPU_SMOD(tid, ((unsigned int)(n)))])))))))))) : ((unsigned char)((((unsigned long long)(((unsigned char)((57005u ^ tid))))) * ((unsigned long long)(((unsigned char)((((unsigned long long)(v_6)) * ((unsigned long long)(((unsigned char)54u))))))))))));
    unsigned int v_24 = 0u;
    bool v_25 = false;
    unsigned int v_26 = 0u;
    unsigned long long v_27 = 0ull;
    unsigned long long v_28 = ((unsigned long long)(__ldg((&(input[GPU_SMOD(tid, ((unsigned int)(n)))])))));
    long long v_29 = 0ll;
    long long v_30 = 0ll;
    int v_31 = (((GPU_SSHR(v_4, v_17) == __ldg((&(input[GPU_SMOD(tid, ((unsigned int)(n)))])))) ? ((unsigned char)((3849760952u ^ GPU_SSHR(v_24, blockIdx.y)))) : ((long long)(((int)(GPU_GROUP_DIVERGE(blockIdx.x, threadIdx.x)))))));
    int v_32 = 0;
    unsigned long long v_33 = 0ull;
    short v_34 = ((short)0);
    unsigned short v_35 = ((unsigned short)0u);
    bool v_36 = GPU_SMOD(((((signed char)-39) >= ((signed char)-110)) ? (v_1 - v_18) : (false | ((bool)(gridDim.x)))), (((unsigned char)(v_8))));
    unsigned short v_37 = ((unsigned short)0u);
    short v_38 = ((short)0);
    bool v_39 = false;
    unsigned long long v_40 = 0ull;
    unsigned char v_41 = ((unsigned char)0u);
    unsigned int v_42 = 0u;
    unsigned long long v_43 = 0ull;
    unsigned int v_44 = 0u;
    unsigned short v_45 = ((((bool)(__ldg((&(input[GPU_SMOD(tid, ((unsigned int)(n)))]))))) <= ((bool)((((long long)(threadIdx.x)))))) ? GPU_SMOD(GPU_SSHL(((unsigned short)(blockDim.x)), ((unsigned short)(blockIdx.x))), (v_10 | v_22)) : ((unsigned short)((((unsigned long long)(((blockDim.x == 3893118615u) ? ((unsigned short)(blockIdx.y)) : v_35))) * ((unsigned long long)((v_11)))))));
    unsigned int su_0 = input[GPU_SMOD(tid, ((unsigned int)(n)))];
    unsigned int su_3 = input[GPU_SMOD(tid, ((unsigned int)(n)))];
    for (int _rep = 0; _rep < 3; _rep += 1) {
        if ((tid < ((unsigned int)(n)))) {
            atomicAdd(&(output[GPU_SMOD(tid, ((unsigned int)(n)))]), ((((signed char)-40) >= ((signed char)42)) ? GPU_SDIV(su_3, su_0) : (~(su_0))));
        }
    }
    if ((tid < ((unsigned int)(n)))) {
        output[tid] = ((unsigned int)(v_36));
    }
}

int main(void) {
    unsigned int h_input[1] = {12345u};
    unsigned int h_output = 0xdeadbeefu;
    unsigned int *d_input = NULL, *d_output = NULL;
    GPU_CUDA_CHECK(cudaMalloc((void**)&d_input, sizeof(h_input)));
    GPU_CUDA_CHECK(cudaMalloc((void**)&d_output, sizeof(h_output)));
    GPU_CUDA_CHECK(cudaMemcpy(d_input, h_input, sizeof(h_input), cudaMemcpyHostToDevice));
    kernel_main<<<1, 32>>>(d_output, d_input, 1, false, true, 5903491134938927595ull);
    GPU_CUDA_CHECK(cudaGetLastError());
    GPU_CUDA_CHECK(cudaDeviceSynchronize());
    GPU_CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(h_output), cudaMemcpyDeviceToHost));
    printf("checksum=%u\n", h_output);
    GPU_CUDA_CHECK(cudaFree(d_input));
    GPU_CUDA_CHECK(cudaFree(d_output));
    return 0;
}
