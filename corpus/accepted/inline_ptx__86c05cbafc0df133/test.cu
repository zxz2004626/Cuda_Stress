// strategy=inline_ptx  (generated; do not edit)
extern "C" __global__ void test_kernel(
    const unsigned long long* __restrict__ input,
    unsigned long long* __restrict__ output,
    unsigned int input_words,
    unsigned long long seed)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // runtime operands feeding the inline PTX block
    unsigned long long w0 = input[(tid * 3u) % input_words] ^ seed;
    unsigned long long w1 = input[(tid * 5u + 1u) % input_words] ^ (seed >> 11u);
    unsigned long long w2 = input[(tid * 7u + 2u) % input_words] ^ (seed >> 29u);
    unsigned long long w3 = input[(tid * 11u + 3u) % input_words] ^ (seed >> 37u);
    unsigned int a0 = (unsigned int)(w0 ^ (w0 >> 32u));
    unsigned int a1 = (unsigned int)(w1 ^ (w1 >> 32u));
    unsigned int a2 = (unsigned int)(w2) ^ (unsigned int)(seed >> 47u);
    unsigned int a3 = (unsigned int)(w3);
    unsigned long long b0 = input[(tid * 13u + 4u) % input_words] ^ (seed * 3u);
    unsigned long long b1 = input[(tid * 17u + 5u) % input_words] ^ (seed >> 13u);

    unsigned long long o0 = 0ULL, o1 = 0ULL;
    asm volatile(
        ".reg .b32 %%xr<8>;\n\t.reg .b64 %%xd<4>;\n\t.reg .pred %%xp<3>;\n\tmov.b32 %%xr0, %2;\n\tmov.b32 %%xr1, %3;\n\tmov.b32 %%xr2, %4;\n\tmov.b32 %%xr3, %5;\n\tmov.b32 %%xr4, 3725182614;\n\tmov.b32 %%xr5, 3665518368;\n\tmov.b32 %%xr6, 3481143568;\n\tmov.b32 %%xr7, 107422264;\n\tmov.u64 %%xd0, %6;\n\tmov.u64 %%xd1, %7;\n\tmov.u64 %%xd2, 0x457b6d03e98e709e;\n\tmov.u64 %%xd3, 0xe178ee944038c5fa;\n\tsetp.eq.u32 %%xp0, %%xr1, %%xr2;\n\tsetp.lt.u32 %%xp1, %%xr5, %%xr4;\n\tsetp.eq.u32 %%xp2, %%xr5, %%xr1;\n\tsetp.eq.u64 %%xp0, %%xd3, %%xd0;\n\txor.b64 %%xd0, %%xd0, %%xd1;\n\tsetp.gt.u64 %%xp2, %%xd1, %%xd0;\n\tclz.b32 %%xr0, %%xr0;\n\tand.b32 %%xr4, %%xr0, 31;\n\tshl.b32 %%xr1, %%xr1, %%xr4;\n\tsub.u64 %%xd1, %%xd2, %%xd0;\n\tcvt.u64.u32 %%xd2, %%xr2;\n\tbrev.b32 %%xr2, %%xr6;\n\tcvt.u64.u32 %%xd3, %%xr2;\n\tmin.u64 %%xd0, %%xd0, %%xd3;\n\tand.b32 %%xr6, %%xr3, 31;\n\tshl.b32 %%xr3, %%xr1, %%xr6;\n\tmov.b32 %%xr4, 2237046488;\n\tbfi.b32 %%xr5, %%xr4, %%xr4, 4, 1;\n\tbrev.b64 %%xd0, %%xd0;\n\tmax.u32 %%xr6, %%xr7, %%xr7;\n\tmov.b32 %%xr7, 328017717;\n\tor.b32 %%xr1, %%xr7, %%xr7;\n\tmul.lo.u64 %%xd2, %%xd2, %%xd2;\n\tmin.u64 %%xd0, %%xd2, %%xd1;\n\tcvt.u32.u64 %%xr6, %%xd2;\n\tand.b32 %%xr6, %%xr6, 63;\n\tshl.b64 %%xd1, %%xd2, %%xr6;\n\tpopc.b32 %%xr7, %%xr0;\n\tbrev.b32 %%xr1, %%xr7;\n\tshr.u32 %%xr5, %%xr1, 26;\n\tcvt.u64.u32 %%xd2, %%xr5;\n\tsetp.lt.u32 %%xp0, %%xr6, %%xr7;\n\tsetp.lt.u64 %%xp1, %%xd0, %%xd0;\n\txor.b64 %%xd0, %%xd3, %%xd0;\n\tsetp.ne.u32 %%xp2, %%xr6, %%xr2;\n\tclz.b32 %%xr7, %%xr0;\n\tclz.b64 %%xr2, %%xd3;\n\tcvt.u32.u64 %%xr0, %%xd2;\n\tor.b64 %%xd2, %%xd3, %%xd3;\n\tpopc.b64 %%xr2, %%xd2;\n\tshf.r.wrap.b32 %%xr6, %%xr5, %%xr2, 5;\n\tselp.b32 %%xr5, %%xr2, %%xr7, %%xp1;\n\tmax.u32 %%xr1, %%xr5, %%xr7;\n\txor.b32 %%xr4, %%xr4, %%xr6;\n\tadd.u64 %%xd0, %%xd1, %%xd0;\n\tor.b32 %%xr5, %%xr7, %%xr2;\n\tmul.lo.u32 %%xr7, %%xr5, %%xr7;\n\tmul.hi.u32 %%xr3, %%xr7, %%xr7;\n\tsetp.gt.u64 %%xp0, %%xd3, %%xd3;\n\tand.b32 %%xr3, %%xr5, 31;\n\tshl.b32 %%xr6, %%xr2, %%xr3;\n\tmul.lo.u64 %%xd2, %%xd0, %%xd3;\n\tbfi.b32 %%xr6, %%xr5, %%xr0, 0, 9;\n\txor.b64 %%xd2, %%xd3, %%xd3;\n\tpopc.b64 %%xr7, %%xd2;\n\tbfi.b32 %%xr4, %%xr7, %%xr7, 1, 8;\n\tand.b32 %%xr4, %%xr2, 31;\n\tshr.u32 %%xr4, %%xr5, %%xr4;\n\tand.b64 %%xd0, %%xd2, %%xd3;\n\tshf.r.wrap.b32 %%xr4, %%xr6, %%xr6, 27;\n\tand.b32 %%xr0, %%xr7, 31;\n\tshl.b32 %%xr5, %%xr1, %%xr0;\n\tclz.b64 %%xr0, %%xd2;\n\tlop3.b32 %%xr6, %%xr0, %%xr5, %%xr7, 0xb4;\n\tand.pred %%xp1, %%xp2, %%xp0;\n\tmov.u64 %%xd1, 0x9d5dcf41a75666cf;\n\tcvt.u32.u64 %%xr6, %%xd1;\n\tand.b32 %%xr6, %%xr6, 63;\n\tshr.u64 %%xd3, %%xd1, %%xr6;\n\tshl.b32 %%xr0, %%xr5, 28;\n\tor.pred %%xp0, %%xp1, %%xp2;\n\tsetp.lt.u64 %%xp1, %%xd2, %%xd3;\n\tshl.b32 %%xr2, %%xr3, 1;\n\tclz.b32 %%xr1, %%xr2;\n\tmul.hi.u64 %%xd3, %%xd3, %%xd1;\n\tsetp.eq.u64 %%xp2, %%xd3, %%xd0;\n\tcvt.u32.u64 %%xr5, %%xd3;\n\tshl.b64 %%xd3, %%xd0, 20;\n\tmul.lo.u32 %%xr6, %%xr3, %%xr3;\n\tsetp.gt.u32 %%xp0, %%xr5, %%xr1;\n\tselp.b32 %%xr3, %%xr7, %%xr2, %%xp0;\n\tsetp.ne.u32 %%xp0, %%xr3, %%xr6;\n\tcvt.u32.u64 %%xr4, %%xd2;\n\tand.b32 %%xr4, %%xr4, 63;\n\tshr.u64 %%xd3, %%xd1, %%xr4;\n\tbfe.u32 %%xr7, %%xr0, 3, 11;\n\tsetp.lt.u64 %%xp1, %%xd2, %%xd2;\n\tmax.u32 %%xr6, %%xr6, %%xr1;\n\tclz.b32 %%xr1, %%xr0;\n\tsetp.ne.u32 %%xp0, %%xr1, %%xr5;\n\tsetp.gt.u32 %%xp2, %%xr1, %%xr1;\n\txor.b32 %%xr7, %%xr1, %%xr1;\n\tcvt.u32.u64 %%xr5, %%xd2;\n\tbrev.b32 %%xr2, %%xr5;\n\tcvt.u64.u32 %%xd2, %%xr0;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tcvt.u64.u32 %%xd2, %%xr1;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tcvt.u64.u32 %%xd2, %%xr2;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tcvt.u64.u32 %%xd2, %%xr3;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tcvt.u64.u32 %%xd2, %%xr4;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tcvt.u64.u32 %%xd2, %%xr5;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tcvt.u64.u32 %%xd2, %%xr6;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tcvt.u64.u32 %%xd2, %%xr7;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\txor.b64 %%xd0, %%xd0, %%xd3;\n\tselp.b32 %%xr0, %%xr1, %%xr2, %%xp0;\n\tcvt.u64.u32 %%xd2, %%xr0;\n\tadd.u64 %%xd0, %%xd0, %%xd2;\n\tselp.b32 %%xr0, %%xr1, %%xr2, %%xp1;\n\tcvt.u64.u32 %%xd2, %%xr0;\n\tadd.u64 %%xd0, %%xd0, %%xd2;\n\tselp.b32 %%xr0, %%xr1, %%xr2, %%xp2;\n\tcvt.u64.u32 %%xd2, %%xr0;\n\tadd.u64 %%xd0, %%xd0, %%xd2;\n\tcvt.u64.u32 %%xd1, %%xr0;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\tmov.u64 %0, %%xd0;\n\tmov.u64 %1, %%xd1;"
        : "=l"(o0), "=l"(o1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(b0), "l"(b1)
        : "memory");

    output[tid * 2u] = o0;
    output[tid * 2u + 1u] = o1;
}
