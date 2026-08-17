// Reduced from:
//   poc/prefix_scan_multi_toolchain_hang/generator_ir__0ac10dd2c222f9fb/test.cu
//
// Differential hang result (4s timeout):
//   cuda-10.1 O0                      -> HANG
//   cuda-10.1 O1, O2, O3              -> checksum
//   cuda-11.4 O0, O1, O2, O3          -> HANG
//   cuda-11.8 O0, O1, O2, O3          -> HANG
//
// Launch: grid=(1), block=(32), n=32.  The prefix scan sweep phase, the
// `_ci` post-processing loop, unused helpers and register-pressure stress
// were removed while preserving every target hang.
//
#include <stdio.h>
#include <stdint.h>
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
#define GPU_GROUP_DIVERGE(idx, val) ((idx) * 0 + (val))

#define GPU_CUDA_CHECK(call) do { cudaError_t _gpu_err = (call); if (_gpu_err != cudaSuccess) { fprintf(stderr, "CUDA_ERROR %s:%d: %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(_gpu_err)); return 2; } } while (0)

extern "C" {
    __device__ unsigned short arr_0[16];
    __device__ bool arr_1[11];
    __device__ unsigned char arr_2[11][11];
}

extern "C" __global__ __launch_bounds__(1024, 1) void kernel_main(unsigned int* __restrict__ output, const unsigned int* __restrict__ input, int n, int var_0, unsigned short var_1, short var_2) {
    unsigned int tid = (((blockIdx.x + (blockIdx.y * gridDim.x)) * (blockDim.x * blockDim.y)) + (threadIdx.x + (threadIdx.y * blockDim.x)));
    unsigned int ov_0 = ((unsigned int)(input[GPU_SMOD((tid + 0u), ((unsigned int)(n)))]));
    unsigned int ov_1 = ((unsigned int)(input[GPU_SMOD((tid + 1u), ((unsigned int)(n)))]));
    int ov_2 = ((int)(input[GPU_SMOD((tid + 2u), ((unsigned int)(n)))]));
    unsigned int ov_3 = ((unsigned int)(input[GPU_SMOD((tid + 3u), ((unsigned int)(n)))]));
    int ov_4 = ((int)(input[GPU_SMOD((tid + 4u), ((unsigned int)(n)))]));
    long long ov_5 = ((long long)(input[GPU_SMOD((tid + 5u), ((unsigned int)(n)))]));
    short ov_6 = ((short)(input[GPU_SMOD((tid + 6u), ((unsigned int)(n)))]));
    unsigned short ov_7 = ((unsigned short)(input[GPU_SMOD((tid + 7u), ((unsigned int)(n)))]));
    unsigned char v_0 = (((((blockDim.x | blockDim.x) - ((unsigned int)(arr_2[GPU_SMOD(3u, 11u)][GPU_SMOD(tid, 11u)]))))));
    signed char v_1 = GPU_SDIV(((signed char)(arr_1[GPU_SMOD(tid, 11u)])), (GPU_SSHL(((signed char)-106), ((signed char)63)) ^ ((signed char)(((unsigned int)(__brevll(((unsigned long long)(18446744073709551615ull)))))))));
    bool v_2 = (((bool)((((unsigned long long)((GPU_SSHL(1444388438u, (2u & 31u)) ^ GPU_SSHR(tid, (2u & 31u))))) % (((unsigned long long)((((unsigned int)(__brev(((unsigned int)(tid))))) ^ ((unsigned int)(__ffsll(((unsigned long long)(0ull)))))))) | 1ull)))) - (((bool)(((2u)))) & (((var_2 >= ((short)(threadIdx.x))) ? ((unsigned char)165u) : ((signed char)28)))));
    bool v_4 = ((bool)((((unsigned long long)(__sad(((int)(((int)((807975337763374679ull / (5117523997147834457ull | 1ull)))))), ((int)(((int)(__byte_perm(2u, tid, 1u))))), ((unsigned int)((31u ^ 1431655765u)))))) % (((unsigned long long)(arr_1[GPU_SMOD(tid, 11u)])) | 1ull))));
    long long v_5 = GPU_SSUB(((long long)((~(937321914u)))), ((long long)((GPU_SSHL((tid ^ tid), ((tid & tid) & 31u)) ^ GPU_SSHR((gridDim.y + 3856395371u), ((tid & tid) & 31u))))));
    unsigned long long v_6 = ((unsigned long long)((((unsigned long long)((((unsigned long long)((GPU_SSHL(1917228391u, (tid & 31u)) ^ GPU_SSHR(tid, (tid & 31u))))) / ((((unsigned long long)(blockDim.x)) ^ ((unsigned long long)(gridDim.x))) | 1ull)))) / (((unsigned long long)(__funnelshift_r(((unsigned int)(((unsigned int)((((unsigned long long)(blockIdx.x)) / (((unsigned long long)(gridDim.y)) | 1ull)))))), ((unsigned int)((~(1400014205u)))), ((unsigned int)(31u))))) | 1ull))));
    long long v_8 = ((((true < v_4) || (var_1 > ((unsigned short)(blockDim.x)))) == ((bool)(__ffs(((int)(var_0)))))) ? GPU_SMOD((((tid == tid) ? ((unsigned short)16108u) : 48879u)), ((unsigned short)((((unsigned int)(__ffsll(((unsigned long long)(v_6))))) ^ ((unsigned int)(__ffsll(((unsigned long long)(v_6))))))))) : GPU_SADD(((long long)(__ldg((&(input[GPU_SMOD(tid, ((unsigned int)(n)))]))))), ((long long)(__umulhi(((unsigned int)(tid)), ((unsigned int)(tid)))))));
    bool v_10 = ((((int)((v_6 / (v_6 | 1ull)))) >= ((int)((v_6 % (((unsigned long long)(gridDim.x)) | 1ull))))) && ((~(((int)(blockIdx.x)))) <= ((int)((((unsigned int)(__brev(((unsigned int)(tid))))) ^ ((unsigned int)(__brevll(((unsigned long long)(((unsigned long long)(blockDim.x))))))))))));
    unsigned long long v_11 = ((unsigned long long)(arr_2[GPU_SMOD((2622358555u ^ (((unsigned int)(__brev(((unsigned int)(4294967295u))))) ^ ((unsigned int)(__brevll(((unsigned long long)(1199521876453071623ull))))))), 11u)][GPU_SMOD(tid, 11u)]));
    unsigned int v_12 = GPU_SMOD((-(GPU_SDIV(tid, threadIdx.x))), (__ldg((&(input[GPU_SMOD(tid, ((unsigned int)(n)))]))) ^ ((unsigned int)(__ffs(((int)(((int)(blockDim.x)))))))));
    unsigned long long v_14 = ((unsigned long long)(arr_2[GPU_SMOD((2310148064u ^ 252645135u), 11u)][GPU_SMOD(((1679988386u | gridDim.y) - (2568506529u ^ 2863311530u)), 11u)]));
    unsigned int v_15 = (((unsigned int)(__brevll(((unsigned long long)((-((v_11 + v_6)))))))) ^ ((unsigned int)(__clzll(((unsigned long long)((-(((-345523426 >= -635445154) ? 1ull : 0ull)))))))));
    unsigned long long v_16 = ((unsigned long long)((2899463633u ^ 1431655765u)));
    unsigned char v_17 = (((unsigned char)(__ldg((&(input[GPU_SMOD(tid, ((unsigned int)(n)))]))))) * GPU_SMOD(((unsigned char)(__any_sync(4294967295u, (gridDim.x <= tid)))), ((unsigned char)(__byte_perm(v_15, v_12, 7u)))));
    int v_18 = GPU_SSHR(__clz(((int)(((int)(arr_2[GPU_SMOD(blockDim.x, 11u)][GPU_SMOD(v_12, 11u)]))))), GPU_SDIV(((int)(arr_1[GPU_SMOD(blockDim.x, 11u)])), (~(127))));
    int v_19 = ((int)(__sad(((int)(GPU_SSUB((v_18 | v_18), ((int)((((unsigned int)(__brev(((unsigned int)(v_15))))) ^ ((unsigned int)(__clz(((unsigned int)(255u))))))))))), ((int)(((((int)(blockDim.x)) & v_18) | ((v_15 >= v_12) ? 1925324810 : v_18)))), ((unsigned int)(GPU_SSHL(GPU_SDIV(57005u, threadIdx.x), ((unsigned int)(arr_0[GPU_SMOD(2325181947u, 16u)]))))))));
    int v_20 = n;
    bool v_21 = ((bool)((((unsigned long long)(arr_0[GPU_SMOD(((var_2 <= ((short)-12532)) ? tid : tid), 16u)])) / (((unsigned long long)((((unsigned int)(__ffsll(((unsigned long long)(((unsigned long long)(__byte_perm(v_15, threadIdx.y, 2u)))))))) ^ ((unsigned int)(__clzll(((unsigned long long)(((unsigned long long)((16422765846487066235ull % (0ull | 1ull)))))))))))) | 1ull))));
    signed char v_23 = GPU_SMUL(GPU_SSUB(((signed char)((((unsigned int)(__ffsll(((unsigned long long)(v_11))))) ^ ((unsigned int)(__brevll(((unsigned long long)(v_6)))))))), ((signed char)(((unsigned int)(__brevll(((unsigned long long)(15271286985664979040ull)))))))), (GPU_SSUB(v_1, ((signed char)(blockDim.x))) & ((signed char)((((unsigned long long)(blockIdx.y)) % (18183411686775930824ull | 1ull))))));
    unsigned char v_24 = (((unsigned char)(((((unsigned int)(arr_2[GPU_SMOD(2654435769u, 11u)][GPU_SMOD(blockIdx.x, 11u)])))))) & ((((unsigned char)(blockDim.x)) * ((unsigned char)1u)) | ((unsigned char)((((short)(threadIdx.x)))))));
    int v_25 = (~((~((v_19 & v_18)))));
    unsigned int v_26 = (((13u * v_12) + ((unsigned int)((((v_15)))))) + GPU_SMOD((((unsigned int)(__ffsll(((unsigned long long)(((unsigned long long)(threadIdx.x))))))) ^ ((unsigned int)(__brev(((unsigned int)(2601449824u)))))), (gridDim.x | v_12)));
    unsigned int v_27 = __ldg((&(input[GPU_SMOD(tid, ((unsigned int)(n)))])));
    unsigned short v_28 = (-(((((unsigned short)53977u) & var_1) & ((unsigned short)((v_6 % (((unsigned long long)(threadIdx.x)) | 1ull)))))));
    unsigned long long v_29 = ((unsigned long long)(__popc(((int)(((int)(((((short)(((unsigned int)(((((signed char)(blockIdx.x)))))))))))))))));
    unsigned int v_30 = (GPU_SSHL(((v_26 ^ 746181890u) | (blockIdx.x ^ v_12)), (((unsigned int)(arr_2[GPU_SMOD(((unsigned int)(((int)(((unsigned char)((1045875537u))))))), 11u)][GPU_SMOD(v_15, 11u)])) & 31u)) ^ GPU_SSHR(GPU_SDIV((v_27 * v_27), ((unsigned int)(arr_2[GPU_SMOD(blockDim.y, 11u)][GPU_SMOD(tid, 11u)]))), (((unsigned int)(arr_2[GPU_SMOD(((unsigned int)(((int)(((unsigned char)((1045875537u))))))), 11u)][GPU_SMOD(v_15, 11u)])) & 31u)));
    unsigned int v_33 = (((blockDim.x - 1u) | (v_15 - v_30)) * ((unsigned int)((((unsigned long long)(((signed char)(((long long)(v_26)))))) / (((unsigned long long)((GPU_SSHL(2410000987u, (tid & 31u)) ^ GPU_SSHR(v_27, (tid & 31u))))) | 1ull)))));
    unsigned int v_34 = ((unsigned int)((((~(((unsigned long long)(gridDim.x)))) ^ ((unsigned long long)((v_11 / (v_16 | 1ull))))) / (((unsigned long long)(((354181807201933678ull + 15439405798484683840ull) / ((v_14 + 13843219006634354105ull) | 1ull)))) | 1ull))));
    unsigned long long v_35 = (~(((unsigned long long)((GPU_SSHL((blockDim.y & 1u), ((v_12 - v_27) & 31u)) ^ GPU_SSHR((4072879068u + v_15), ((v_12 - v_27) & 31u)))))));
    bool v_37 = ((bool)(((short)(((__byte_perm(((unsigned int)((((unsigned int)(__popcll(((unsigned long long)(1ull))))) ^ ((unsigned int)(__clzll(((unsigned long long)(1ull)))))))), ((unsigned int)(blockDim.y)), ((unsigned int)(4u)))))))));
    long long v_38 = v_5;
    unsigned long long v_40 = ((unsigned long long)((((unsigned long long)(threadIdx.x)) / (((unsigned long long)((((unsigned int)(__clz(((unsigned int)((1615650706u - blockIdx.x)))))) ^ ((unsigned int)(__clzll(((unsigned long long)(((unsigned long long)(__usad(((unsigned int)(gridDim.x)), ((unsigned int)(v_27)), ((unsigned int)(v_15))))))))))))) | 1ull))));
    unsigned int v_41 = (((unsigned int)(__ffsll(((unsigned long long)((((unsigned long long)(gridDim.x)) ^ v_40)))))) ^ ((((unsigned int)(__ffs(((unsigned int)(tid))))) ^ ((unsigned int)(__clzll(((unsigned long long)(7951810411198542408ull)))))) + (3204561766u | blockDim.x)));
    unsigned int v_42 = 2654435769u;
    long long v_43 = (((long long)((((unsigned long long)(((unsigned int)(__ffsll(((unsigned long long)(16486552543943701789ull))))))) % ((1ull ^ v_40) | 1ull)))) | ((long long)(arr_1[GPU_SMOD(((unsigned int)(__clzll(((unsigned long long)(((unsigned long long)(gridDim.x))))))), 11u)])));
    int v_44 = (GPU_SMUL(GPU_SSHL(((int)(threadIdx.x)), v_18), ((int)((v_42)))) ^ (~((var_0 | v_25))));
    long long v_46 = (~(((long long)((((unsigned int)(__brev(((unsigned int)((v_33 & gridDim.x)))))) ^ ((unsigned int)(__popcll(((unsigned long long)(((unsigned long long)((GPU_SSHL(255u, (v_34 & 31u)) ^ GPU_SSHR(57005u, (v_34 & 31u)))))))))))))));
    unsigned short v_47 = GPU_SSHR(((unsigned short)(((unsigned int)(__brevll(((unsigned long long)(((unsigned long long)(GPU_GROUP_DIVERGE(blockIdx.x, v_33)))))))))), ((v_4 != ((bool)(blockIdx.x))) ? (-(v_28)) : (((unsigned short)1u) + ((unsigned short)16049u))));
    short v_49 = ((short)((4000620364u ^ 4278255360u)));
    unsigned int _fv = input[GPU_SMOD(tid, ((unsigned int)(n)))];
    for (int _rep = 0; _rep < 1; _rep += 1) {
        __shared__ unsigned int temp_0[256];
        unsigned int idx_0 = ((2u * (blockIdx.x * blockDim.x)) + threadIdx.x);
        temp_0[threadIdx.x] = 0u;
        if ((idx_0 < ((unsigned int)(n)))) {
            temp_0[threadIdx.x] = input[idx_0];
        }
        temp_0[(threadIdx.x + blockDim.x)] = 0u;
        if (((idx_0 + blockDim.x) < ((unsigned int)(n)))) {
            temp_0[(threadIdx.x + blockDim.x)] = input[(idx_0 + blockDim.x)];
        }
        __syncthreads();
        if ((threadIdx.x < 128u)) {
            temp_0[((1u * ((2u * threadIdx.x) + 2u)) - 1u)] = (temp_0[((1u * ((2u * threadIdx.x) + 2u)) - 1u)] + temp_0[((1u * ((2u * threadIdx.x) + 1u)) - 1u)]);
        }
        __syncthreads();
        if ((threadIdx.x < 64u)) {
            temp_0[((2u * ((2u * threadIdx.x) + 2u)) - 1u)] = (temp_0[((2u * ((2u * threadIdx.x) + 2u)) - 1u)] + temp_0[((2u * ((2u * threadIdx.x) + 1u)) - 1u)]);
        }
        __syncthreads();
        if ((threadIdx.x < 32u)) {
            temp_0[((4u * ((2u * threadIdx.x) + 2u)) - 1u)] = (temp_0[((4u * ((2u * threadIdx.x) + 2u)) - 1u)] + temp_0[((4u * ((2u * threadIdx.x) + 1u)) - 1u)]);
        }
        __syncthreads();
        if ((threadIdx.x < 16u)) {
            temp_0[((8u * ((2u * threadIdx.x) + 2u)) - 1u)] = (temp_0[((8u * ((2u * threadIdx.x) + 2u)) - 1u)] + temp_0[((8u * ((2u * threadIdx.x) + 1u)) - 1u)]);
        }
        __syncthreads();
        if ((threadIdx.x < 8u)) {
            temp_0[((16u * ((2u * threadIdx.x) + 2u)) - 1u)] = (temp_0[((16u * ((2u * threadIdx.x) + 2u)) - 1u)] + temp_0[((16u * ((2u * threadIdx.x) + 1u)) - 1u)]);
        }
        __syncthreads();
        if ((threadIdx.x < 4u)) {
            temp_0[((32u * ((2u * threadIdx.x) + 2u)) - 1u)] = (temp_0[((32u * ((2u * threadIdx.x) + 2u)) - 1u)] + temp_0[((32u * ((2u * threadIdx.x) + 1u)) - 1u)]);
        }
        __syncthreads();
        if ((threadIdx.x < 2u)) {
            temp_0[((64u * ((2u * threadIdx.x) + 2u)) - 1u)] = (temp_0[((64u * ((2u * threadIdx.x) + 2u)) - 1u)] + temp_0[((64u * ((2u * threadIdx.x) + 1u)) - 1u)]);
        }
        __syncthreads();
        if ((threadIdx.x < 1u)) {
            temp_0[((128u * ((2u * threadIdx.x) + 2u)) - 1u)] = (temp_0[((128u * ((2u * threadIdx.x) + 2u)) - 1u)] + temp_0[((128u * ((2u * threadIdx.x) + 1u)) - 1u)]);
        }
        __syncthreads();
        __syncthreads();
        if ((idx_0 < ((unsigned int)(n)))) {
            _fv = ((_fv * 2654435761u) ^ (((unsigned int)(temp_0[threadIdx.x])) ^ (((unsigned int)(idx_0)) * 2246822507u)));
        }
        if (((idx_0 + blockDim.x) < ((unsigned int)(n)))) {
            _fv = ((_fv * 2654435761u) ^ (((unsigned int)(temp_0[(threadIdx.x + blockDim.x)])) ^ (((unsigned int)((idx_0 + blockDim.x))) * 2246822507u)));
        }
        v_23 = ((signed char)(((unsigned int)(__popc(((unsigned int)(v_26)))))));
        {
            int w_0 = 0;
            while ((w_0 < 2)) {
#pragma unroll 2
                for (int i_0 = 0; i_0 < 8; i_0 += 1) {
                    v_6 = (((unsigned short)(arr_2[GPU_SMOD((-((2u - 729853709u))), 11u)][GPU_SMOD(__umulhi(((unsigned int)(((unsigned int)((0ull / (v_35 | 1ull)))))), ((unsigned int)((v_26 & 2863311530u)))), 11u)])));
                    v_28 = (GPU_SMUL(((signed char)(((long long)(((((unsigned int)(((unsigned short)((((unsigned long long)(2498163207u))))))))))))), ((signed char)((GPU_SSHL((v_42 | v_12), ((2654435769u & v_12) & 31u)) ^ GPU_SSHR(GPU_SSHR(v_15, v_34), ((2654435769u & v_12) & 31u)))))));
                    v_8 = (((signed char)((((unsigned int)(__brevll(((unsigned long long)(((unsigned long long)((((unsigned int)(__ffs(((unsigned int)(((unsigned int)(arr_1[GPU_SMOD(threadIdx.y, 11u)]))))))) ^ ((unsigned int)(__popcll(((unsigned long long)(((unsigned long long)((3625927488u ^ v_26)))))))))))))))) ^ ((unsigned int)(__ffs(((unsigned int)((-((255u ^ v_34))))))))))));
                    {
                        int w_1 = 0;
                        while ((w_1 < 4)) {
                            v_0 = ((((short)-32768) ^ ((short)((GPU_SMOD(6705878402126746631ull, 7076027505451189404ull) % (((unsigned long long)((((unsigned int)(__ffs(((unsigned int)(1034971804u))))) ^ ((unsigned int)(__popcll(((unsigned long long)(((unsigned long long)(threadIdx.x)))))))))) | 1ull))))));
                            v_5 ^= GPU_SDIV((~(1638931535386871238ll)), v_8);
                            v_47 -= (((signed char)((((unsigned int)(__clz(((unsigned int)((GPU_SSHL(__ldg((&(input[GPU_SMOD(tid, ((unsigned int)(n)))]))), (((unsigned int)((((unsigned long long)(threadIdx.y)) % (2762234487323624350ull | 1ull)))) & 31u)) ^ GPU_SSHR((v_27 + gridDim.y), (((unsigned int)((((unsigned long long)(threadIdx.y)) % (2762234487323624350ull | 1ull)))) & 31u)))))))) ^ ((unsigned int)(__ffs(((unsigned int)(((GPU_SSHL(234282439u, (31u & 31u)) ^ GPU_SSHR(7u, (31u & 31u))) * (13u & gridDim.x)))))))))));
                            v_43 ^= ((long long)(((unsigned int)(__clz(((unsigned int)((tid & ((short)(blockDim.x))))))))));
                            w_1++;
                        }
                    }
                }
                v_49 ^= ((short)((((unsigned long long)(((int)(((unsigned char)(((unsigned long long)(blockIdx.x)))))))) % ((1ull * 18446744073709551615ull) | 1ull))));
                w_0++;
            }
        }
        v_1 = GPU_SADD(((signed char)((3593584025u ^ v_26))), ((signed char)(__ffs(((int)(((int)(blockIdx.x))))))));
        v_38 = ((long long)(__sad(((int)(((int)((v_40 % (6297994222695119503ull | 1ull)))))), ((int)(((int)(__all_sync(4294967295u, (v_23 <= ((signed char)41))))))), ((unsigned int)(((unsigned int)(arr_0[GPU_SMOD(v_33, 16u)])))))));
        if (((3128397187u + v_15) != (-(v_41)))) {
#pragma unroll 4
            for (int i_1 = 0; i_1 < 4; i_1 += 1) {
                v_15 = (((long long)((((unsigned int)(((-(((unsigned int)(arr_2[GPU_SMOD(v_30, 11u)][GPU_SMOD(blockIdx.x, 11u)])))))))))));
                switch ((v_41 % 3u)) {
                    case 0: {
                        v_27 *= ((threadIdx.y ^ threadIdx.x) + GPU_SSHL(v_34, 37u));
                        v_35 &= ((299261318u != v_30) ? ((short)31546) : ((unsigned long long)(((unsigned int)(__brevll(((unsigned long long)(9223372036854775807ull))))))));
                        break;
                    }
                    case 1: {
                        v_37 ^= (((int)((((unsigned long long)((23930518u ^ 252645135u))) % (((unsigned long long)(__popc(((int)(GPU_SSUB(0, v_20)))))) | 1ull)))));
                        break;
                    }
                    case 2: {
                        unsigned short v_50 = ((3325207756u - ((unsigned int)((((unsigned long long)(__ldg((&(input[GPU_SMOD(tid, ((unsigned int)(n)))]))))) / (((unsigned long long)(((unsigned int)(__ffs(((unsigned int)(v_26))))))) | 1ull))))));
                        var_0 = GPU_SMUL(var_0, ((int)((GPU_SSHL(__ldg((&(input[GPU_SMOD(tid, ((unsigned int)(n)))]))), ((1151293920u & 0u) & 31u)) ^ GPU_SSHR(((unsigned int)((v_16 / (v_14 | 1ull)))), ((1151293920u & 0u) & 31u))))));
                        break;
                    }
                    default: {
                        v_24 = ((unsigned char)(((unsigned short)((((signed char)((969586774u - v_26))))))));
                        break;
                    }
                }
                v_34--;
                v_46 = (v_46 ^ (__clzll((((unsigned long long)((v_35 / (((unsigned long long)(__byte_perm(threadIdx.x, v_34, 0u))) | 1ull))))))));
            }
        }
        if ((((v_6 <= v_6) && (n <= v_44)) != GPU_SSHL(false, v_37))) {
            v_8 = GPU_SSUB(v_8, GPU_SSUB(0ll, GPU_SADD(v_8, 42ll)));
        }
        v_14 = ((unsigned long long)((((unsigned int)(__clz(((unsigned int)((tid ^ 4294967295u)))))) ^ ((unsigned int)(__clzll(((unsigned long long)(((unsigned long long)(arr_0[GPU_SMOD(v_15, 16u)]))))))))));
        v_12 = (v_12 ^ (__ffsll((((((unsigned long long)(gridDim.y)) - ((unsigned long long)(gridDim.x))) ^ ((unsigned long long)(__ballot_sync(4294967295u, (550481160u != v_34)))))))));
        _fv = (_fv ^ ((unsigned int)(var_1)));
        _fv = (_fv ^ ((unsigned int)(v_2)));
        _fv = (_fv ^ ((unsigned int)(v_4)));
        _fv = (_fv ^ ((unsigned int)(v_6)));
        _fv = (_fv ^ ((unsigned int)(v_10)));
        _fv = (_fv ^ ((unsigned int)(v_11)));
        _fv = (_fv ^ ((unsigned int)(v_12)));
        _fv = (_fv ^ ((unsigned int)(v_14)));
        _fv = (_fv ^ ((unsigned int)(v_15)));
        _fv = (_fv ^ ((unsigned int)(v_16)));
        _fv = (_fv ^ ((unsigned int)(v_21)));
        _fv = (_fv ^ ((unsigned int)(v_26)));
        _fv = (_fv ^ ((unsigned int)(v_27)));
        _fv = (_fv ^ ((unsigned int)(v_28)));
        _fv = (_fv ^ ((unsigned int)(v_29)));
        _fv = (_fv ^ ((unsigned int)(v_30)));
        _fv = (_fv ^ ((unsigned int)(v_33)));
        _fv = (_fv ^ ((unsigned int)(v_34)));
        _fv = (_fv ^ ((unsigned int)(v_35)));
        _fv = (_fv ^ ((unsigned int)(v_37)));
        _fv = (_fv ^ ((unsigned int)(v_40)));
        _fv = (_fv ^ ((unsigned int)(v_41)));
        _fv = (_fv ^ ((unsigned int)(v_42)));
        _fv = (_fv ^ ((unsigned int)(v_47)));
        _fv = (_fv ^ ((unsigned int)(idx_0)));
        _fv = (_fv ^ ((unsigned int)(ov_0)));
        _fv = (_fv ^ ((unsigned int)(ov_1)));
        _fv = (_fv ^ ((unsigned int)(ov_3)));
        _fv = (_fv ^ ((unsigned int)(ov_7)));
        ov_0 = ((unsigned int)((((unsigned int)(ov_0)) ^ ((((unsigned int)(_rep)) * 2246822507u) + _fv))));
        ov_1 = ((unsigned int)((((unsigned int)(ov_1)) ^ ((((unsigned int)(_rep)) * 2246822507u) + _fv))));
        ov_2 = ((int)((((unsigned int)(ov_2)) ^ ((((unsigned int)(_rep)) * 2246822507u) + _fv))));
        ov_3 = ((unsigned int)((((unsigned int)(ov_3)) ^ ((((unsigned int)(_rep)) * 2246822507u) + _fv))));
        ov_4 = ((int)((((unsigned int)(ov_4)) ^ ((((unsigned int)(_rep)) * 2246822507u) + _fv))));
        ov_5 = ((long long)((((unsigned int)(ov_5)) ^ ((((unsigned int)(_rep)) * 2246822507u) + _fv))));
        ov_6 = ((short)((((unsigned int)(ov_6)) ^ ((((unsigned int)(_rep)) * 2246822507u) + _fv))));
        ov_7 = ((unsigned short)((((unsigned int)(ov_7)) ^ ((((unsigned int)(_rep)) * 2246822507u) + _fv))));

    }
    if ((tid < ((unsigned int)(n)))) {
        output[tid] = _fv;
    }
}

int main(void) {
    const int n = 32;
    unsigned int h_input[32];
    unsigned int h_output[32];
    unsigned int *d_input = NULL, *d_output = NULL;
    for (int i = 0; i < n; i++) h_input[i] = (unsigned int)i * 2654435761u + 12345u;
    GPU_CUDA_CHECK(cudaMalloc((void**)&d_input, sizeof(h_input)));
    GPU_CUDA_CHECK(cudaMalloc((void**)&d_output, sizeof(h_output)));
    GPU_CUDA_CHECK(cudaMemcpy(d_input, h_input, sizeof(h_input), cudaMemcpyHostToDevice));
    kernel_main<<<1, 32, 1024>>>(d_output, d_input, n, 1964635965, ((unsigned short)19169u), ((short)-17667));
    GPU_CUDA_CHECK(cudaGetLastError());
    GPU_CUDA_CHECK(cudaDeviceSynchronize());
    GPU_CUDA_CHECK(cudaMemcpy(h_output, d_output, sizeof(h_output), cudaMemcpyDeviceToHost));
    unsigned long long hash = 42;
    for (int i = 0; i < n; i++) {
        unsigned long long val = h_output[i];
        hash ^= val + 0x9e3779b9ULL + (hash << 6) + (hash >> 2);
    }
    printf("checksum=%llu\n", hash);
    GPU_CUDA_CHECK(cudaFree(d_input));
    GPU_CUDA_CHECK(cudaFree(d_output));
    return 0;
}
