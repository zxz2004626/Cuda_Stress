// strategy=memory_address  (generated; do not edit)
__device__ __forceinline__ unsigned long long load_by_width(const unsigned char* p, unsigned int w)
{
    switch (w)
    {
        case 8: { unsigned int v; asm("ld.global.ca.u8 %0, [%1];" : "=r"(v) : "l"(p)); return (unsigned long long)v; }
        case 16: { unsigned int v; asm("ld.global.ca.u16 %0, [%1];" : "=r"(v) : "l"(p)); return (unsigned long long)v; }
        case 64: { unsigned long long v; asm("ld.global.ca.b64 %0, [%1];" : "=l"(v) : "l"(p)); return v; }
    }
    return 0ULL;
}

extern "C" __global__ void test_kernel(
    const unsigned long long* __restrict__ input,
    unsigned long long* __restrict__ output,
    unsigned int input_words,
    unsigned long long seed)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned char* by = (const unsigned char*)input;

    static const unsigned int W[3] = { 8u, 16u, 64u };
    unsigned long long acc = seed ^ 0xDEADBEEFCAFEBABEULL;

    // -- scalar multi-level runtime-addressed reads ---------------------------
    for (unsigned int k = 0u; k < 76u; ++k)
    {
        unsigned int w = W[k % 3u];
        unsigned int base = (tid * 17u + k * 31u + (unsigned int)(seed >> (k % 64u))) % input_words;
        unsigned int off = base * 8u;
        unsigned int sub;
        if (w == 8u)       sub = (unsigned int)((seed >> ((k + 1u) % 64u)) & 7u);
        else if (w == 16u) sub = 2u * (unsigned int)((seed >> ((k + 2u) % 64u)) & 3u);
        else if (w == 32u) sub = 4u * (unsigned int)((seed >> ((k + 3u) % 64u)) & 1u);
        else               sub = 0u;
        unsigned long long v = load_by_width(by + off + sub, w);
        acc = acc * 11400714819323198485ULL + v;
        acc ^= acc >> 27u;
    }

    // -- struct Pair{lo,hi} loads (LDG.E.64) --------------------------------
    for (unsigned int k = 0u; k < 6u; ++k)
    {
        unsigned int base3 = (tid * 3u + k * 11u + (unsigned int)(seed >> (32u + k % 32u))) % input_words;
        struct Pair { unsigned int lo, hi; };
        Pair pr = *(const Pair*)(by + (size_t)base3 * 8u);
        acc = acc * 11400714819323198485ULL + (unsigned long long)pr.lo ^ (unsigned long long)pr.hi;
    }

    // -- local array (spill stress) -------------------------------------------
    unsigned long long local[4];
    for (unsigned int i = 0u; i < 4u; ++i)
    {
        unsigned int base4 = (tid + i * 3u) % input_words;
        local[i] = ((const unsigned long long*)by)[base4] ^ (seed >> (i * 7u % 64u));
    }
    unsigned long long lacc = 0ULL;
    for (unsigned int i = 0u; i < 4u; ++i) lacc = lacc * 11400714819323198485ULL + local[i];

    output[tid * 2u] = acc ^ lacc;
    output[tid * 2u + 1u] = (acc * 3ULL) ^ seed ^ (lacc >> 1u);
}
