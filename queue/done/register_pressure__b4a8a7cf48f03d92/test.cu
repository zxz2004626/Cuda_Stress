// strategy=register_pressure  (generated; do not edit)
__constant__ unsigned long long K[96] = { 13704045453389458397ULL, 6522551031523835457ULL, 3127964573470165781ULL, 4300275575039513495ULL, 13673471553935298425ULL, 11489585339946877321ULL, 13131313005578216771ULL, 15925505943023788671ULL, 13789804603408713157ULL, 15643243679554779487ULL, 9659692098011639513ULL, 17905211722119817205ULL, 16127949405287011241ULL, 15309434369502335013ULL, 13784661163054770377ULL, 16574930506431851659ULL, 102881483583470431ULL, 8747356209498052595ULL, 15039618202855418317ULL, 1976610093497966117ULL, 4540715169891606441ULL, 10545095749471346387ULL, 17744920754291457367ULL, 15464241373893946327ULL, 15825447652584048507ULL, 6576316326993915801ULL, 8103627300370547833ULL, 18134630055238797067ULL, 16631066005235887211ULL, 14969060327900255997ULL, 8015548666738830941ULL, 7346137172095517571ULL, 13594267435914734253ULL, 4342615170634514779ULL, 775722943194201201ULL, 14357160234036081999ULL, 3396456406244410071ULL, 272076930630409895ULL, 7007766781481943171ULL, 6268109574965413059ULL, 15050436646286862179ULL, 1032735365346804857ULL, 2869207130939610911ULL, 13533974203502647843ULL, 16667290935865031397ULL, 9655667225062009177ULL, 15048703417346745261ULL, 8221139563177134689ULL, 58083017188312841ULL, 8305478693471835973ULL, 5535769202278836685ULL, 8964152454382845315ULL, 18124085212688681993ULL, 4296581583044919021ULL, 17724270553089899115ULL, 3621284396934989919ULL, 7550921661400907333ULL, 13155094345057265745ULL, 18179955948606516703ULL, 10683540147641036051ULL, 7725087176866225713ULL, 4077860960098851935ULL, 16075318176664384057ULL, 15152219932634734405ULL, 11102609719942081819ULL, 15091934938380831941ULL, 16719571012193148595ULL, 16277708967294179959ULL, 11052678333464013059ULL, 1556091427808429193ULL, 15759865376253299423ULL, 14970005787003984465ULL, 16938514392971087567ULL, 1803573651490628485ULL, 17855372882648875323ULL, 11809851823797104871ULL, 5934098419371518055ULL, 16506480541291889425ULL, 1713574464873064113ULL, 2691988412963010955ULL, 15465167935213801923ULL, 10764691255947639669ULL, 4844818887035591041ULL, 14020161437367285491ULL, 1310836293220062437ULL, 8522875286213563595ULL, 13160452840938117201ULL, 8590066917046790315ULL, 6845736857378706115ULL, 11041135687470654389ULL, 7278279385538850803ULL, 16996681224611998097ULL, 13697441442730732319ULL, 8405976599802399541ULL, 12673320160193415071ULL, 10175504281213096341ULL };
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
