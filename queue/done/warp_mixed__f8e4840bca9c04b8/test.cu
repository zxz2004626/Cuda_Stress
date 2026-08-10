// strategy=warp_mixed  (generated; do not edit)
extern "C" __global__ void test_kernel(
    const unsigned long long* __restrict__ input,
    unsigned long long* __restrict__ output,
    unsigned int input_words,
    unsigned long long seed)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int lane = threadIdx.x & 31u;
    unsigned int warpid = (blockIdx.x * (blockDim.x >> 5u)) + (threadIdx.x >> 5u);
    unsigned int base = warpid * 32u;

    unsigned long long v = input[(base + lane) % input_words] ^ seed;
    unsigned int vu = (unsigned int)(v) ^ (unsigned int)(v >> 32u);
    int pred = (int)((vu & 0x80000000u) != 0u);

    const unsigned int SH = 10u;
    unsigned int m = 0xFFFFFFFFu;
    if (0u) m = (lane < 16u) ? 0xFFFFu : 0xFFFFFFFFu;

    // -- warp ops (ops=ballot, match_all, shfl_idx, activemask) -----------------------------------------
    unsigned int b = 0u, an = 0u, al = 0u, am = 0xFFFFFFFFu, ma = 0u, mall = 0u;
    int mallp = 0;
    unsigned long long s = v;
b = __ballot_sync(m, pred);
s = __shfl_sync(m, v, (lane + 10u) & 31u, 32u);
mall = __match_all_sync(m, vu, &mallp);
    // activemask in a partially-divergent region exercises partial active masks.
    am = __activemask();
    if (lane < 16u) { unsigned int am2 = __activemask(); am ^= am2; }

    // -- long integer chain, predicates, store --------------------------------
    unsigned long long acc = seed ^ v;
    acc = acc * 11400714819323198485ULL + (unsigned long long)b;
    acc ^= s;
    acc = acc * 11400714819323198485ULL + (unsigned long long)ma;
    acc = acc * 11400714819323198485ULL + (unsigned long long)mall;
    for (unsigned int j = 0u; j < 240u; ++j)
        acc = acc * 11400714819323198485ULL + (input[(base + lane + j * 3u) % input_words] ^ (seed >> (j % 63u)));
    acc = pred ? (acc ^ s) : (acc + (s >> 1u));

    output[tid * 2u] = acc ^ (unsigned long long)am;
    output[tid * 2u + 1u] = acc ^ (unsigned long long)mallp ^ (unsigned long long)al
                            ^ (unsigned long long)an ^ v;
}
