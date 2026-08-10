// strategy=integer_chain  (generated; do not edit)
extern "C" __global__ void test_kernel(
    const unsigned long long* __restrict__ input,
    unsigned long long* __restrict__ output,
    unsigned int input_words,
    unsigned long long seed)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long long acc = seed ^ ((unsigned long long)tid * 11400714819323198485ULL) ^ 6364136223846793005ULL;
    for (unsigned int j = 0u; j < 160u; ++j)
    {
        unsigned int idx = (tid * 3u + j * 5u) % input_words;
        unsigned long long v = input[idx] ^ (seed + j);
        unsigned int a = (unsigned int)(v >> 32u), b = (unsigned int)(v ^ acc);
        unsigned int c = (unsigned int)(acc >> 32u) ^ (unsigned int)(seed >> (j % 63u));
        if ((j & 1u) == 0u) { a = __umulhi(a, b); } else { unsigned long long x1 = (unsigned long long)a ^ ((acc * 3u) & 0xFFFFFFFFULL); unsigned long long x2 = (unsigned long long)b ^ (seed & 0xFFFFFFFFULL); unsigned long long hi; asm("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(x1), "l"(x2)); a = (unsigned int)(hi ^ (hi >> 32u)); }
    a = __funnelshift_r(a, b, c & 31u);
    { unsigned int bpos = c & 15u, bwid = 1u + ((c >> 4u) & 15u);
      asm("bfe.u32 %0, %1, %2, %3;" : "=r"(a) : "r"(a), "r"(bpos), "r"(bwid)); }
    { unsigned int bpos = (c >> 5u) & 15u, bwid = 1u + ((c >> 9u) & 15u);
      asm("bfi.b32 %0, %1, %2, %3, %4;" : "=r"(a) : "r"(a), "r"(b), "r"(bpos), "r"(bwid)); }
    a = __byte_perm(a, b, c & 0x77777777u);
    asm volatile("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(a) : "r"(a), "r"(b), "r"(c));
    a = __popc(a ^ b);
    a = __clz(a | 1u);
    a = __brev(a);
    acc = acc * 6364136223846793005ULL + (unsigned long long)a;
    acc = acc ^ (acc >> 29u);
    if ((j & 3u) == 0u) acc = acc + input[(tid + j * 13u) % input_words] ^ (seed >> (j % 63u));
    { unsigned int r = j % 64u; if (r != 0u) acc = (acc << r) | (acc >> (64u - r)); }
    }
    output[tid * 2u] = acc;
    output[tid * 2u + 1u] = acc ^ seed ^ input[tid % input_words];
}
