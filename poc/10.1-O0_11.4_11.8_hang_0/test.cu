#define GPU_SDIV(a, b) ((b) != 0 && !((__typeof__(b))(-1) < (__typeof__(b))(0) && (b) == (__typeof__(b))(-1) && (a) == (__typeof__(a))(1ULL << (sizeof(a) * 8 - 1))) ? (a) / (b) : 0)
#define GPU_SMOD(a, b) ((b) != 0 && !((__typeof__(b))(-1) < (__typeof__(b))(0) && (b) == (__typeof__(b))(-1) && (a) == (__typeof__(a))(1ULL << (sizeof(a) * 8 - 1))) ? (a) % (b) : 0)
#define GPU_SSHL(a, b) ((b) >= 0 && (b) < (long long)(sizeof(a) * 8) ? ((unsigned long long)(a) << (b)) : 0ULL)
#define GPU_SSHR(a, b) ((b) >= 0 && (b) < (long long)(sizeof(a) * 8) ? (a) >> (b) : 0)

__device__ unsigned char arr_2[1][1];

__global__ void kernel_main(unsigned int* output, const unsigned int* input, int n) {
    unsigned int tid = threadIdx.x;
    unsigned long v_6 = ((unsigned long long)((((unsigned long long)((((unsigned long long)((GPU_SSHL(1917228391u, tid) ^ GPU_SSHR(tid, tid)))) / ((((unsigned long long)(blockDim.x)) ^ ((unsigned long long)(gridDim.x))) | 1ull)))) / (((unsigned long long)(__funnelshift_r(((unsigned int)(((unsigned int)((((unsigned long long)(blockIdx.x)) / (((unsigned long long)(gridDim.y)) | 1ull)))))), ((unsigned int)((~(1400014205u)))), ((unsigned int)(31u))))) | 1ull))));
    unsigned v_11 = arr_2[0][0];
    unsigned v_12 = GPU_SMOD((-(GPU_SDIV(tid, tid))), (__ldg((&(input[GPU_SMOD(tid, ((unsigned int)(n)))]))) ^ 6u));
    unsigned v_15 = (((unsigned int)(__brevll(((unsigned long long)((-((v_11 + v_6)))))))) ^ ((unsigned int)(__clzll(((unsigned long long)((-(((-345523426 >= -635445154) ? 1ull : 0ull)))))))));
    unsigned v_17 = __ldg((&(input[GPU_SMOD(tid, ((unsigned int)(n)))]))) * GPU_SMOD(__any_sync(0xffffu, tid), ((unsigned char)(__byte_perm(v_15, v_12, 7))));
    unsigned short v_28 = (-(((((unsigned short)53977u) & 19169u) & ((unsigned short)((v_6 % (((unsigned long long)(tid)) | 1ull)))))));
    unsigned long v_35 = (~(((unsigned long long)((GPU_SSHL((blockDim.y & 1u), ((v_12 - 0u) & 31u)) ^ GPU_SSHR((4072879068u + v_15), ((v_12 - 0u) & 31u)))))));
    unsigned short v_47 = GPU_SSHR(((unsigned short)(((unsigned int)(__brevll(((unsigned long long)(((unsigned long long)(0u))))))))), ((0 != ((bool)(blockIdx.x))) ? (-(v_28)) : (((unsigned short)1u) + ((unsigned short)16049u))));
        __syncthreads();
        output[tid] = ((unsigned int)v_35) ^ ((unsigned int)v_47);
}

int main(){
    unsigned int *d_input, *d_output;
    cudaMalloc((void**)&d_input, 64);
    cudaMalloc((void**)&d_output, 64);
    cudaMemset(d_input, 1, 64);
    kernel_main<<<1, 16>>>(d_output, d_input, 16);
    return cudaDeviceSynchronize();
}
