// strategy=register_pressure  (generated; do not edit)
__constant__ unsigned long long K[48] = { 2484047370572192747ULL, 4806784749388353819ULL, 12149299439483413981ULL, 15486474717214167555ULL, 12843638809188834083ULL, 5395943977138278283ULL, 5237411262488414821ULL, 13058560787624552361ULL, 14929467400018116177ULL, 8111380137456639577ULL, 1467826829997155597ULL, 1737780278617367729ULL, 14338392823438220671ULL, 1980936773219971189ULL, 7010489353590694193ULL, 12475476578030775339ULL, 7607988854400210619ULL, 14348317769660935639ULL, 15179886032923699275ULL, 13103500301419730467ULL, 11115675641639400087ULL, 12094839788282072639ULL, 15626686333093852983ULL, 2034001744632596117ULL, 1355348063180760279ULL, 12968507255252690767ULL, 18162282659291020861ULL, 15050424069178091027ULL, 1713092301885677027ULL, 8279763169344231089ULL, 1472326700848337295ULL, 9818984027646655919ULL, 3435508561848184819ULL, 1966694073596481945ULL, 8250307550680896585ULL, 3961468053891839373ULL, 6541300973404078735ULL, 2450173903861463267ULL, 933105806759457509ULL, 12940014950688087347ULL, 8191412007852654681ULL, 9480777610452062387ULL, 6782702005454987769ULL, 1324991303101497869ULL, 15848949306114893423ULL, 11290897619458124483ULL, 7227845272003155113ULL, 5696298003307989287ULL };
__device__ __noinline__ unsigned long long chain_0(
    const unsigned long long* __restrict__ in, unsigned int n,
    unsigned long long seed, unsigned int tid)
{
    unsigned long long v[48];
    for (unsigned int i = 0u; i < 48u; ++i)
    {
        unsigned int idx = (tid + i * 5u + 0u * 7u) % n;
        v[i] = (in[idx] ^ (seed >> (i % 63u))) * K[i];
    }
    unsigned long long acc = seed ^ 0u;
    for (unsigned int i = 0u; i < 48u; ++i) acc = acc * 11400714819323198485ULL + v[i];
    for (unsigned int i = 0u; i < 48u; ++i)
        acc ^= v[(i * 17u) % 48u] >> ((i * 3u) & 31u);
    return acc;
}

__device__ __noinline__ unsigned long long chain_1(
    const unsigned long long* __restrict__ in, unsigned int n,
    unsigned long long seed, unsigned int tid)
{
    unsigned long long v[48];
    for (unsigned int i = 0u; i < 48u; ++i)
    {
        unsigned int idx = (tid + i * 5u + 1u * 7u) % n;
        v[i] = (in[idx] ^ (seed >> (i % 63u))) * K[i];
    }
    unsigned long long acc = seed ^ 1u;
    for (unsigned int i = 0u; i < 48u; ++i) acc = acc * 11400714819323198485ULL + v[i];
    for (unsigned int i = 0u; i < 48u; ++i)
        acc ^= v[(i * 17u) % 48u] >> ((i * 3u) & 31u);
    return acc;
}

__device__ __noinline__ unsigned long long chain_2(
    const unsigned long long* __restrict__ in, unsigned int n,
    unsigned long long seed, unsigned int tid)
{
    unsigned long long v[48];
    for (unsigned int i = 0u; i < 48u; ++i)
    {
        unsigned int idx = (tid + i * 5u + 2u * 7u) % n;
        v[i] = (in[idx] ^ (seed >> (i % 63u))) * K[i];
    }
    unsigned long long acc = seed ^ 2u;
    for (unsigned int i = 0u; i < 48u; ++i) acc = acc * 11400714819323198485ULL + v[i];
    for (unsigned int i = 0u; i < 48u; ++i)
        acc ^= v[(i * 17u) % 48u] >> ((i * 3u) & 31u);
    return acc;
}

__device__ __noinline__ unsigned long long chain_3(
    const unsigned long long* __restrict__ in, unsigned int n,
    unsigned long long seed, unsigned int tid)
{
    unsigned long long v[48];
    for (unsigned int i = 0u; i < 48u; ++i)
    {
        unsigned int idx = (tid + i * 5u + 3u * 7u) % n;
        v[i] = (in[idx] ^ (seed >> (i % 63u))) * K[i];
    }
    unsigned long long acc = seed ^ 3u;
    for (unsigned int i = 0u; i < 48u; ++i) acc = acc * 11400714819323198485ULL + v[i];
    for (unsigned int i = 0u; i < 48u; ++i)
        acc ^= v[(i * 17u) % 48u] >> ((i * 3u) & 31u);
    return acc;
}

extern "C" __global__ void test_kernel(
    const unsigned long long* __restrict__ input,
    unsigned long long* __restrict__ output,
    unsigned int input_words,
    unsigned long long seed)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long long acc = seed;
acc = acc * 11400714819323198485ULL ^ chain_0(input, input_words, seed, tid);
    acc = acc * 11400714819323198485ULL ^ chain_1(input, input_words, seed, tid);
    acc = acc * 11400714819323198485ULL ^ chain_2(input, input_words, seed, tid);
    acc = acc * 11400714819323198485ULL ^ chain_3(input, input_words, seed, tid);

    for (unsigned int j = 0u; j < 32u; ++j)
    {
        unsigned int idx = (tid + j * 13u) % input_words;
        acc = acc * 11400714819323198485ULL + (input[idx] ^ (acc >> 33u));
    }
    output[tid * 2u] = acc;
    output[tid * 2u + 1u] = acc ^ seed ^ input[tid % input_words];
}
