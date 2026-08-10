  1  // strategy=inline_ptx  (generated; do not edit)
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
        ".reg .b32 %%xr<12>;\n\t
        .reg .b64 %%xd<6>;\n\t
        .reg .pred %%xp<5>;\n\t
        mov.b32 %%xr0, %2;\n\t
        mov.b32 %%xr1, %3;\n\t
        mov.b32 %%xr2, %4;\n\t
        mov.b32 %%xr3, %5;\n\t
        mov.b32 %%xr4, 989845345;\n\t
        mov.b32 %%xr5, 764839310;\n\t
        mov.b32 %%xr6, 1164899354;\n\t
        mov.b32 %%xr7, 3702160615;\n\t
        mov.b32 %%xr8, 227713714;\n\t
        mov.b32 %%xr9, 3789839018;\n\t
        mov.b32 %%xr10, 2668828170;\n\t
        mov.b32 %%xr11, 1785953000;\n\t
        mov.u64 %%xd0, %6;\n\t
        mov.u64 %%xd1, %7;\n\t
        mov.u64 %%xd2, 0xcbe6544a11844df0;\n\t
        mov.u64 %%xd3, 0x0f856f2cbac4ffaa;\n\t
        mov.u64 %%xd4, 0x7b5e769579d49f3c;\n\t
        mov.u64 %%xd5, 0xd0d45fc241979f18;\n\t
        setp.ne.u32 %%xp0, %%xr9, %%xr2;\n\t
        setp.gt.u32 %%xp1, %%xr1, %%xr1;\n\t
        setp.eq.u32 %%xp2, %%xr1, %%xr9;\n\t
        setp.eq.u32 %%xp3, %%xr0, %%xr5;\n\t
        setp.ne.u32 %%xp4, %%xr8, %%xr8;\n\t
        max.u64 %%xd0, %%xd3, %%xd1;\n\t
        xor.b64 %%xd0, %%xd0, %%xd0;\n\t
        add.u64 %%xd0, %%xd0, %%xd1;\n\t
        popc.b64 %%xr1, %%xd5;\n\t
        cvt.u64.u32 %%xd2, %%xr1;\n\t
        xor.b64 %%xd1, %%xd1, %%xd2;\n\t
        setp.lt.u32 %%xp2, %%xr1, %%xr2;\n\t
        selp.b32 %%xr0, %%xr1, %%xr2, %%xp2;\n\t
        cvt.u64.u32 %%xd2, %%xr0;\n\t
        xor.b64 %%xd0, %%xd0, %%xd2;\n\t
        xor.b64 %%xd3, %%xd2, %%xd1;\n\t
        xor.b64 %%xd1, %%xd1, %%xd3;\n\t
        brev.b64 %%xd4, %%xd0;\n\t
        xor.b64 %%xd0, %%xd0, %%xd4;\n\t
        prmt.b32 %%xr5, %%xr2, %%xr2, 0x6402;\n\t
        cvt.u64.u32 %%xd2, %%xr5;\n\t
        xor.b64 %%xd1, %%xd1, %%xd2;\n\t
        max.u32 %%xr6, %%xr5, %%xr2;\n\t
        cvt.u64.u32 %%xd2, %%xr6;\n\t
        xor.b64 %%xd0, %%xd0, %%xd2;\n\t
        cvt.u64.u32 %%xd1, %%xr7;\n\t
        xor.b64 %%xd1, %%xd1, %%xd1;\n\t
        add.u64 %%xd0, %%xd0, %%xd1;\n\t
        add.u32 %%xr8, %%xr4, %%xr2;\n\t
        cvt.u64.u32 %%xd2, %%xr8;\n\t
        xor.b64 %%xd0, %%xd0, %%xd2;\n\t
        and.b64 %%xd3, %%xd2, %%xd2;\n\t
        xor.b64 %%xd1, %%xd1, %%xd3;\n\t
        shl.b32 %%xr10, %%xr6, 7;\n\t
        cvt.u64.u32 %%xd2, %%xr10;\n\t
        xor.b64 %%xd0, %%xd0, %%xd2;\n\t
        max.u64 %%xd5, %%xd5, %%xd5;\n\t
        xor.b64 %%xd1, %%xd1, %%xd5;\n\t
        bfe.u32 %%xr0, %%xr2, 15, 2;\n\t
        cvt.u64.u32 %%xd2, %%xr0;\n\t
        xor.b64 %%xd0, %%xd0, %%xd2;\n\t
        setp.ne.u32 %%xp3, %%xr6, %%xr2;\n\t
        selp.b32 %%xr0, %%xr1, %%xr2, %%xp3;\n\t
        cvt.u64.u32 %%xd2, %%xr0;\n\t
        xor.b64 %%xd1, %%xd1, %%xd2;\n\t
        setp.gt.u32 %%xp4, %%xr0, %%xr2;\n\tselp.b32 %%xr0, %%xr1, %%xr2, %%xp4;\n\tcvt.u64.u32 %%xd2, %%xr0;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tadd.u64 %%xd0, %%xd0, %%xd1;\n\tshf.r.wrap.b32 %%xr3, %%xr0, %%xr2, 24;\n\tcvt.u64.u32 %%xd2, %%xr3;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\tor.pred %%xp1, %%xp4, %%xp0;\n\tselp.b32 %%xr0, %%xr1, %%xr2, %%xp1;\n\tcvt.u64.u32 %%xd2, %%xr0;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tor.pred %%xp2, %%xp4, %%xp2;\n\tselp.b32 %%xr0, %%xr1, %%xr2, %%xp2;\n\tcvt.u64.u32 %%xd2, %%xr0;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\tprmt.b32 %%xr6, %%xr2, %%xr3, 0x7631;\n\tcvt.u64.u32 %%xd2, %%xr6;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tmax.u32 %%xr7, %%xr9, %%xr6;\n\tcvt.u64.u32 %%xd2, %%xr7;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\tsetp.lt.u32 %%xp0, %%xr9, %%xr7;\n\tselp.b32 %%xr0, %%xr1, %%xr2, %%xp0;\n\tcvt.u64.u32 %%xd2, %%xr0;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tmax.u32 %%xr9, %%xr7, %%xr2;\n\tcvt.u64.u32 %%xd2, %%xr9;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\tadd.u64 %%xd0, %%xd0, %%xd1;\n\tshr.u64 %%xd4, %%xd5, 24;\n\txor.b64 %%xd0, %%xd0, %%xd4;\n\txor.b32 %%xr11, %%xr2, %%xr2;\n\tcvt.u64.u32 %%xd2, %%xr11;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\tsub.u32 %%xr0, %%xr4, %%xr11;\n\tcvt.u64.u32 %%xd2, %%xr0;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tmin.u32 %%xr1, %%xr2, %%xr2;\n\tcvt.u64.u32 %%xd2, %%xr1;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\tshr.u64 %%xd2, %%xd2, 46;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tmul.lo.u64 %%xd3, %%xd2, %%xd0;\n\txor.b64 %%xd1, %%xd1, %%xd3;\n\tcvt.u32.u64 %%xr4, %%xd2;\n\tcvt.u64.u32 %%xd2, %%xr4;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tadd.u64 %%xd0, %%xd0, %%xd1;\n\tmin.u32 %%xr5, %%xr4, %%xr4;\n\tcvt.u64.u32 %%xd2, %%xr5;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\tmul.hi.u32 %%xr6, %%xr5, %%xr8;\n\tcvt.u64.u32 %%xd2, %%xr6;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tpopc.b64 %%xr7, %%xd4;\n\tcvt.u64.u32 %%xd2, %%xr7;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\tmul.lo.u64 %%xd2, %%xd3, %%xd0;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tcvt.u32.u64 %%xr7, %%xd2;\n\tand.b32 %%xr7, %%xr7, 63;\n\tshl.b64 %%xd3, %%xd3, %%xr7;\n\txor.b64 %%xd1, %%xd1, %%xd3;\n\tsad.u32 %%xr10, %%xr4, %%xr2, %%xr7;\n\tcvt.u64.u32 %%xd2, %%xr10;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tor.pred %%xp0, %%xp4, %%xp3;\n\tselp.b32 %%xr0, %%xr1, %%xr2, %%xp0;\n\tcvt.u64.u32 %%xd2, %%xr0;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\tadd.u64 %%xd0, %%xd0, %%xd1;\n\tsad.u32 %%xr0, %%xr10, %%xr10, %%xr10;\n\tcvt.u64.u32 %%xd2, %%xr0;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tshl.b32 %%xr1, %%xr8, 4;\n\tcvt.u64.u32 %%xd2, %%xr1;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\txor.b32 %%xr2, %%xr2, %%xr1;\n\tcvt.u64.u32 %%xd2, %%xr2;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tsetp.lt.u64 %%xp4, %%xd3, %%xd3;\n\tselp.b32 %%xr0, %%xr1, %%xr2, %%xp4;\n\tcvt.u64.u32 %%xd2, %%xr0;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\txor.b32 %%xr4, %%xr3, %%xr2;\n\tcvt.u64.u32 %%xd2, %%xr4;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tclz.b64 %%xr5, %%xd3;\n\tcvt.u64.u32 %%xd2, %%xr5;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\tcvt.u64.u32 %%xd0, %%xr5;\n\txor.b64 %%xd0, %%xd0, %%xd0;\n\tadd.u64 %%xd0, %%xd0, %%xd1;\n\tand.pred %%xp3, %%xp0, %%xp2;\n\tselp.b32 %%xr0, %%xr1, %%xr2, %%xp3;\n\tcvt.u64.u32 %%xd2, %%xr0;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\tand.b32 %%xr10, %%xr2, 31;\n\tshl.b32 %%xr8, %%xr8, %%xr10;\n\tcvt.u64.u32 %%xd2, %%xr8;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tmov.u64 %%xd3, 0x7a71214387abe0f5;\n\txor.b64 %%xd1, %%xd1, %%xd3;\n\tselp.b32 %%xr10, %%xr10, %%xr4, %%xp4;\n\tcvt.u64.u32 %%xd2, %%xr10;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tor.b64 %%xd5, %%xd5, %%xd0;\n\txor.b64 %%xd1, %%xd1, %%xd5;\n\tor.b32 %%xr0, %%xr0, %%xr9;\n\tcvt.u64.u32 %%xd2, %%xr0;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\txor.b32 %%xr1, %%xr3, %%xr0;\n\tcvt.u64.u32 %%xd2, %%xr1;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\tadd.u64 %%xd0, %%xd0, %%xd1;\n\tmul.hi.u64 %%xd2, %%xd3, %%xd4;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tcvt.u64.u32 %%xd3, %%xr3;\n\txor.b64 %%xd1, %%xd1, %%xd3;\n\tsetp.eq.u32 %%xp2, %%xr3, %%xr3;\n\tselp.b32 %%xr0, %%xr1, %%xr2, %%xp2;\n\tcvt.u64.u32 %%xd2, %%xr0;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tnot.pred %%xp3, %%xp3;\n\tselp.b32 %%xr0, %%xr1, %%xr2, %%xp3;\n\tcvt.u64.u32 %%xd2, %%xr0;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\tsetp.lt.u32 %%xp4, %%xr3, %%xr3;\n\tselp.b32 %%xr0, %%xr1, %%xr2, %%xp4;\n\tcvt.u64.u32 %%xd2, %%xr0;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tcvt.u32.u64 %%xr7, %%xd1;\n\tcvt.u64.u32 %%xd2, %%xr7;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\tadd.u64 %%xd2, %%xd2, %%xd1;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tadd.u64 %%xd0, %%xd0, %%xd1;\n\tsetp.gt.u32 %%xp2, %%xr3, %%xr1;\n\tselp.b32 %%xr0, %%xr1, %%xr2, %%xp2;\n\tcvt.u64.u32 %%xd2, %%xr0;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\tcvt.u64.u32 %%xd4, %%xr3;\n\txor.b64 %%xd0, %%xd0, %%xd4;\n\tshf.r.wrap.b32 %%xr11, %%xr3, %%xr3, 18;\n\tcvt.u64.u32 %%xd2, %%xr11;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\txor.b64 %%xd0, %%xd4, %%xd4;\n\txor.b64 %%xd0, %%xd0, %%xd0;\n\tmin.u32 %%xr1, %%xr3, %%xr10;\n\tcvt.u64.u32 %%xd2, %%xr1;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\tlop3.b32 %%xr2, %%xr1, %%xr6, %%xr10, 0xc5;\n\tcvt.u64.u32 %%xd2, %%xr2;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tor.b32 %%xr3, %%xr10, %%xr2;\n\tcvt.u64.u32 %%xd2, %%xr3;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\tadd.u64 %%xd0, %%xd0, %%xd1;\n\tmin.u32 %%xr4, %%xr3, %%xr11;\n\tcvt.u64.u32 %%xd2, %%xr4;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tprmt.b32 %%xr5, %%xr8, %%xr5, 0x0752;\n\tcvt.u64.u32 %%xd2, %%xr5;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\tmin.u32 %%xr6, %%xr3, %%xr5;\n\tcvt.u64.u32 %%xd2, %%xr6;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tsetp.gt.u32 %%xp2, %%xr11, %%xr6;\n\tselp.b32 %%xr0, %%xr1, %%xr2, %%xp2;\n\tcvt.u64.u32 %%xd2, %%xr0;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\tand.b32 %%xr3, %%xr2, 31;\n\tshr.u32 %%xr8, %%xr6, %%xr3;\n\tcvt.u64.u32 %%xd2, %%xr8;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tbrev.b32 %%xr9, %%xr11;\n\tcvt.u64.u32 %%xd2, %%xr9;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\tclz.b32 %%xr10, %%xr9;\n\tcvt.u64.u32 %%xd2, %%xr10;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tadd.u64 %%xd0, %%xd0, %%xd1;\n\tmul.hi.u64 %%xd5, %%xd4, %%xd4;\n\txor.b64 %%xd1, %%xd1, %%xd5;\n\tand.b64 %%xd0, %%xd5, %%xd4;\n\txor.b64 %%xd0, %%xd0, %%xd0;\n\tbrev.b32 %%xr1, %%xr10;\n\tcvt.u64.u32 %%xd2, %%xr1;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\tclz.b32 %%xr2, %%xr11;\n\tcvt.u64.u32 %%xd2, %%xr2;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tshf.r.wrap.b32 %%xr3, %%xr9, %%xr2, 17;\n\tcvt.u64.u32 %%xd2, %%xr3;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\tsetp.eq.u64 %%xp1, %%xd0, %%xd3;\n\tselp.b32 %%xr0, %%xr1, %%xr2, %%xp1;\n\tcvt.u64.u32 %%xd2, %%xr0;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tprmt.b32 %%xr5, %%xr5, %%xr9, 0x6373;\n\tcvt.u64.u32 %%xd2, %%xr5;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\tadd.u64 %%xd0, %%xd0, %%xd1;\n\tsetp.lt.u64 %%xp3, %%xd0, %%xd1;\n\tselp.b32 %%xr0, %%xr1, %%xr2, %%xp3;\n\tcvt.u64.u32 %%xd2, %%xr0;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tadd.u64 %%xd1, %%xd1, %%xd3;\n\txor.b64 %%xd1, %%xd1, %%xd1;\n\tbrev.b32 %%xr8, %%xr8;\n\tcvt.u64.u32 %%xd2, %%xr8;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tmul.hi.u64 %%xd3, %%xd0, %%xd5;\n\txor.b64 %%xd1, %%xd1, %%xd3;\n\tor.pred %%xp2, %%xp1, %%xp0;\n\tselp.b32 %%xr0, %%xr1, %%xr2, %%xp2;\n\tcvt.u64.u32 %%xd2, %%xr0;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tshr.u64 %%xd5, %%xd3, 39;\n\txor.b64 %%xd1, %%xd1, %%xd5;\n\txor.b32 %%xr0, %%xr3, %%xr7;\n\tcvt.u64.u32 %%xd2, %%xr0;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tadd.u64 %%xd0, %%xd0, %%xd1;\n\tmin.u64 %%xd1, %%xd1, %%xd1;\n\txor.b64 %%xd1, %%xd1, %%xd1;\n\tclz.b32 %%xr2, %%xr6;\n\tcvt.u64.u32 %%xd2, %%xr2;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tmul.hi.u64 %%xd3, %%xd5, %%xd4;\n\txor.b64 %%xd1, %%xd1, %%xd3;\n\tshl.b64 %%xd4, %%xd5, 51;\n\txor.b64 %%xd0, %%xd0, %%xd4;\n\tmax.u32 %%xr5, %%xr5, %%xr5;\n\tcvt.u64.u32 %%xd2, %%xr5;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\tbrev.b32 %%xr6, %%xr10;\n\tcvt.u64.u32 %%xd2, %%xr6;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tmin.u64 %%xd1, %%xd3, %%xd4;\n\txor.b64 %%xd1, %%xd1, %%xd1;\n\tadd.u64 %%xd0, %%xd0, %%xd1;\n\tor.pred %%xp2, %%xp4, %%xp0;\n\tselp.b32 %%xr0, %%xr1, %%xr2, %%xp2;\n\tcvt.u64.u32 %%xd2, %%xr0;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tsetp.lt.u64 %%xp3, %%xd1, %%xd0;\n\tselp.b32 %%xr0, %%xr1, %%xr2, %%xp3;\n\tcvt.u64.u32 %%xd2, %%xr0;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\tcvt.u64.u32 %%xd4, %%xr1;\n\txor.b64 %%xd0, %%xd0, %%xd4;\n\tbfe.u32 %%xr11, %%xr2, 10, 16;\n\tcvt.u64.u32 %%xd2, %%xr11;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\txor.b32 %%xr0, %%xr3, %%xr0;\n\tcvt.u64.u32 %%xd2, %%xr0;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tshl.b32 %%xr1, %%xr0, 27;\n\tcvt.u64.u32 %%xd2, %%xr1;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\tmul.lo.u32 %%xr2, %%xr1, %%xr0;\n\tcvt.u64.u32 %%xd2, %%xr2;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tadd.u64 %%xd0, %%xd0, %%xd1;\n\tshl.b32 %%xr3, %%xr5, 29;\n\tcvt.u64.u32 %%xd2, %%xr3;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\tand.pred %%xp0, %%xp1, %%xp0;\n\tselp.b32 %%xr0, %%xr1, %%xr2, %%xp0;\n\tcvt.u64.u32 %%xd2, %%xr0;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tand.pred %%xp1, %%xp0, %%xp2;\n\tselp.b32 %%xr0, %%xr1, %%xr2, %%xp1;\n\tcvt.u64.u32 %%xd2, %%xr0;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\txor.b64 %%xd0, %%xd1, %%xd3;\n\txor.b64 %%xd0, %%xd0, %%xd0;\n\tsetp.ne.u32 %%xp3, %%xr1, %%xr2;\n\tselp.b32 %%xr0, %%xr1, %%xr2, %%xp3;\n\tcvt.u64.u32 %%xd2, %%xr0;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\tmul.lo.u32 %%xr8, %%xr1, %%xr5;\n\tcvt.u64.u32 %%xd2, %%xr8;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tand.b32 %%xr9, %%xr10, %%xr7;\n\tcvt.u64.u32 %%xd2, %%xr9;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\tadd.u64 %%xd0, %%xd0, %%xd1;\n\tshl.b64 %%xd4, %%xd5, 42;\n\txor.b64 %%xd0, %%xd0, %%xd4;\n\tselp.b32 %%xr11, %%xr3, %%xr1, %%xp1;\n\tcvt.u64.u32 %%xd2, %%xr11;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\tprmt.b32 %%xr0, %%xr11, %%xr11, 0x2707;\n\tcvt.u64.u32 %%xd2, %%xr0;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tshl.b32 %%xr1, %%xr6, 13;\n\tcvt.u64.u32 %%xd2, %%xr1;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\tpopc.b32 %%xr2, %%xr3;\n\tcvt.u64.u32 %%xd2, %%xr2;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tand.b32 %%xr3, %%xr9, %%xr2;\n\tcvt.u64.u32 %%xd2, %%xr3;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\tbrev.b64 %%xd4, %%xd5;\n\txor.b64 %%xd0, %%xd0, %%xd4;\n\tadd.u64 %%xd0, %%xd0, %%xd1;\n\tcvt.u32.u64 %%xr5, %%xd3;\n\tcvt.u64.u32 %%xd2, %%xr5;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\tadd.u64 %%xd0, %%xd1, %%xd2;\n\txor.b64 %%xd0, %%xd0, %%xd0;\n\tbrev.b32 %%xr7, %%xr7;\n\tcvt.u64.u32 %%xd2, %%xr7;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\tlop3.b32 %%xr8, %%xr7, %%xr4, %%xr5, 0x8d;\n\tcvt.u64.u32 %%xd2, %%xr8;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tshl.b64 %%xd3, %%xd0, 39;\n\txor.b64 %%xd1, %%xd1, %%xd3;\n\tand.pred %%xp3, %%xp3, %%xp0;\n\tselp.b32 %%xr0, %%xr1, %%xr2, %%xp3;\n\tcvt.u64.u32 %%xd2, %%xr0;\n\txor.b64 %%xd0, %%xd0, %%xd2;\n\tsub.u32 %%xr11, %%xr8, %%xr8;\n\tcvt.u64.u32 %%xd2, %%xr11;\n\txor.b64 %%xd1, %%xd1, %%xd2;\n\tadd.u64 %%xd0, %%xd0, %%xd1;\n\tmov.u64 %0, %%xd0;\n\tmov.u64 %1, %%xd1;"
        : "=l"(o0), "=l"(o1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(b0), "l"(b1)
        : "memory");

    output[tid * 2u] = o0;
    output[tid * 2u + 1u] = o1;
}
