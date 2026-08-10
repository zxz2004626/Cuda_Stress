// strategy=register_pressure  (generated; do not edit)
__constant__ unsigned long long K[96] = { 11830564072651626331ULL, 8596043654295544649ULL, 2059331876417447263ULL, 3859323920021305253ULL, 6753553534019586901ULL, 12119703480719231741ULL, 10646148220727536131ULL, 16431977520151695605ULL, 12827897745139877977ULL, 10802891694470833867ULL, 4943866603948274731ULL, 4992702399149252159ULL, 13410830455453579409ULL, 3026830359207489225ULL, 13497953613813801871ULL, 12014108624236152001ULL, 14306491029893547683ULL, 15860576766863248979ULL, 378004392124117847ULL, 17295336249379765987ULL, 13616467583635763367ULL, 15762339175618590105ULL, 14633837530054240361ULL, 12287959636544003921ULL, 10506647069272305093ULL, 15448325439755598979ULL, 8310443195200357243ULL, 14590612195068737575ULL, 10218033228875845861ULL, 9827110844824182319ULL, 4425378181674278401ULL, 10221011445950198771ULL, 498726202458972319ULL, 9298580702048568973ULL, 7047871752314296983ULL, 12761945196710989303ULL, 14745310257149704919ULL, 13316529730909811901ULL, 15795117208598229713ULL, 4247398059914827579ULL, 11743278538033488373ULL, 15543191644960910687ULL, 15410635139527023663ULL, 11820086504044670915ULL, 12221080340860264835ULL, 16336728057882414423ULL, 15590040907721198619ULL, 16558641449321494467ULL, 14181634262317215201ULL, 11145820199678535499ULL, 14134088902389663181ULL, 12143042827581927683ULL, 6506992177264687343ULL, 195133786904604891ULL, 2364725238056318647ULL, 1368407815453026593ULL, 4834737622799579771ULL, 18290538327214079921ULL, 11035274909258631669ULL, 14886186241884642707ULL, 13488690748573527865ULL, 934679050927091845ULL, 4250660013405420109ULL, 4509553170122992857ULL, 8344207829213367409ULL, 17055682345839848647ULL, 14119452712616288653ULL, 3567884642940926219ULL, 11739209220306477411ULL, 5198526770653656991ULL, 7852820467451075485ULL, 3488751177512626627ULL, 10835048582815381111ULL, 14267116896267896297ULL, 12916626619467720265ULL, 9029959771678416637ULL, 15325503565065116915ULL, 16682066782526519439ULL, 1940019556014607467ULL, 9386394782402037791ULL, 1758401553269558631ULL, 8293658910844107967ULL, 11217235593040185623ULL, 15192069855240329041ULL, 5003177400461716181ULL, 713910572703624531ULL, 6141568543631545627ULL, 18435430109990179689ULL, 12879067275432918371ULL, 14667303777272025309ULL, 3222889865136279777ULL, 6366519234543099283ULL, 5689134142966108797ULL, 13664666453247237539ULL, 17342506191189560111ULL, 7557505288606927919ULL };
__device__ __noinline__ unsigned long long chain_0(
    const unsigned long long* __restrict__ in, unsigned int n,
    unsigned long long seed, unsigned int tid)
{
    unsigned long long v[96];
    for (unsigned int i = 0u; i < 96u; ++i)
    {
        unsigned int idx = (tid + i * 5u + 0u * 7u) % n;
        v[i] = (in[idx] ^ (seed >> (i % 63u))) * K[i];
    }
    unsigned long long acc = seed ^ 0u;
    for (unsigned int i = 0u; i < 96u; ++i) acc = acc * 11400714819323198485ULL + v[i];
    for (unsigned int i = 0u; i < 96u; ++i)
        acc ^= v[(i * 17u) % 96u] >> ((i * 3u) & 31u);
    return acc;
}

__device__ __noinline__ unsigned long long chain_1(
    const unsigned long long* __restrict__ in, unsigned int n,
    unsigned long long seed, unsigned int tid)
{
    unsigned long long v[96];
    for (unsigned int i = 0u; i < 96u; ++i)
    {
        unsigned int idx = (tid + i * 5u + 1u * 7u) % n;
        v[i] = (in[idx] ^ (seed >> (i % 63u))) * K[i];
    }
    unsigned long long acc = seed ^ 1u;
    for (unsigned int i = 0u; i < 96u; ++i) acc = acc * 11400714819323198485ULL + v[i];
    for (unsigned int i = 0u; i < 96u; ++i)
        acc ^= v[(i * 17u) % 96u] >> ((i * 3u) & 31u);
    return acc;
}

__device__ __noinline__ unsigned long long chain_2(
    const unsigned long long* __restrict__ in, unsigned int n,
    unsigned long long seed, unsigned int tid)
{
    unsigned long long v[96];
    for (unsigned int i = 0u; i < 96u; ++i)
    {
        unsigned int idx = (tid + i * 5u + 2u * 7u) % n;
        v[i] = (in[idx] ^ (seed >> (i % 63u))) * K[i];
    }
    unsigned long long acc = seed ^ 2u;
    for (unsigned int i = 0u; i < 96u; ++i) acc = acc * 11400714819323198485ULL + v[i];
    for (unsigned int i = 0u; i < 96u; ++i)
        acc ^= v[(i * 17u) % 96u] >> ((i * 3u) & 31u);
    return acc;
}

__device__ __noinline__ unsigned long long chain_3(
    const unsigned long long* __restrict__ in, unsigned int n,
    unsigned long long seed, unsigned int tid)
{
    unsigned long long v[96];
    for (unsigned int i = 0u; i < 96u; ++i)
    {
        unsigned int idx = (tid + i * 5u + 3u * 7u) % n;
        v[i] = (in[idx] ^ (seed >> (i % 63u))) * K[i];
    }
    unsigned long long acc = seed ^ 3u;
    for (unsigned int i = 0u; i < 96u; ++i) acc = acc * 11400714819323198485ULL + v[i];
    for (unsigned int i = 0u; i < 96u; ++i)
        acc ^= v[(i * 17u) % 96u] >> ((i * 3u) & 31u);
    return acc;
}

__device__ __noinline__ unsigned long long chain_4(
    const unsigned long long* __restrict__ in, unsigned int n,
    unsigned long long seed, unsigned int tid)
{
    unsigned long long v[96];
    for (unsigned int i = 0u; i < 96u; ++i)
    {
        unsigned int idx = (tid + i * 5u + 4u * 7u) % n;
        v[i] = (in[idx] ^ (seed >> (i % 63u))) * K[i];
    }
    unsigned long long acc = seed ^ 4u;
    for (unsigned int i = 0u; i < 96u; ++i) acc = acc * 11400714819323198485ULL + v[i];
    for (unsigned int i = 0u; i < 96u; ++i)
        acc ^= v[(i * 17u) % 96u] >> ((i * 3u) & 31u);
    return acc;
}

__device__ __noinline__ unsigned long long chain_5(
    const unsigned long long* __restrict__ in, unsigned int n,
    unsigned long long seed, unsigned int tid)
{
    unsigned long long v[96];
    for (unsigned int i = 0u; i < 96u; ++i)
    {
        unsigned int idx = (tid + i * 5u + 5u * 7u) % n;
        v[i] = (in[idx] ^ (seed >> (i % 63u))) * K[i];
    }
    unsigned long long acc = seed ^ 5u;
    for (unsigned int i = 0u; i < 96u; ++i) acc = acc * 11400714819323198485ULL + v[i];
    for (unsigned int i = 0u; i < 96u; ++i)
        acc ^= v[(i * 17u) % 96u] >> ((i * 3u) & 31u);
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
    acc = acc * 11400714819323198485ULL ^ chain_4(input, input_words, seed, tid);
    acc = acc * 11400714819323198485ULL ^ chain_5(input, input_words, seed, tid);

    for (unsigned int j = 0u; j < 96u; ++j)
    {
        unsigned int idx = (tid + j * 13u) % input_words;
        acc = acc * 11400714819323198485ULL + (input[idx] ^ (acc >> 33u));
    }
    output[tid * 2u] = acc;
    output[tid * 2u + 1u] = acc ^ seed ^ input[tid % input_words];
}
