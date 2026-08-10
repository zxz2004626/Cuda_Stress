// strategy=predicate_pressure  (generated; do not edit)
extern "C" __global__ __launch_bounds__(64, 1) void test_kernel(
    const unsigned long long* __restrict__ input,
    unsigned long long* __restrict__ output,
    unsigned int input_words,
    unsigned long long seed)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    static const unsigned long long K[48] = { 10768145586187315651ULL, 10768144486675687537ULL, 10768143387164060325ULL, 10768142287652433115ULL, 10768149984233828111ULL, 10768148884722196925ULL, 10768147785210569713ULL, 10768146685698941479ULL, 10768154382280336475ULL, 10768153282768709257ULL, 10768152183257086269ULL, 10768151083745459059ULL, 10768158780326854055ULL, 10768157680815226837ULL, 10768156581303598601ULL, 10768155481791967423ULL, 10768163178373362419ULL, 10768162078861735201ULL, 10768160979350107989ULL, 10768159879838468491ULL, 10768167576419862591ULL, 10768166476908235373ULL, 10768165377396608161ULL, 10768164277884980951ULL, 10768171974466371851ULL, 10768170874954744761ULL, 10768169775443117549ULL, 10768168675931489315ULL, 10768176372512884311ULL, 10768175273001261189ULL, 10768174173489634105ULL, 10768173073978006895ULL, 10768110401815224227ULL, 10768109302303592913ULL, 10768108202791964677ULL, 10768107103280337595ULL, 10768114799861732591ULL, 10768113700350105373ULL, 10768112600838498641ULL, 10768111501326871431ULL, 10768119197908265531ULL, 10768118098396638313ULL, 10768116998885011101ULL, 10768115899373379795ULL, 10768123595954774791ULL, 10768122496443147701ULL, 10768121396931520489ULL, 10768120297419892255ULL };
    static const unsigned long long A[48] = { 7408628722803885680ULL, 17935520831114394213ULL, 6538751536270392922ULL, 13580272342011228751ULL, 2165486466779756068ULL, 8198200646752479769ULL, 15248175459977244174ULL, 3842952020192535267ULL, 10910939153145833176ULL, 16799547112951192269ULL, 5258653696481533602ULL, 12300041323859677847ULL, 885247752064636556ULL, 7926778419066637153ULL, 13967937020122206038ULL, 2562721139497763659ULL, 9630709234540513056ULL, 16681238201608272661ULL, 8734225259223842570ULL, 15622622440700277759ULL, 4216845826324625364ULL, 11266951447040191433ULL, 18308479811923603390ULL, 5776802448908379027ULL, 12962439387618011016ULL, 1565668890149476477ULL, 8607188424596380754ULL, 14486226235176276039ULL, 2936888861447345212ULL, 9986862540783306769ULL, 17028390768244542470ULL, 5649486338294956283ULL, 11547101617596974288ULL, 285429350816463045ULL, 7317951581597416634ULL, 14358910262001905839ULL, 6421467348582151300ULL, 13318317678532274553ULL, 1940115870872579438ULL, 8981083450465558851ULL, 16031610063908513080ULL, 3473051737522608429ULL, 10649118647265729794ULL, 17690648111677542903ULL, 6293876634973246956ULL, 12182475626852944321ULL, 803720763786464694ULL, 7710133546363879851ULL };

    // -- runtime values ----------------------------------------------------
    unsigned long long x[48];
    #pragma unroll 4
    for (unsigned int i = 0u; i < 48u; ++i)
    {
        unsigned int idx = (tid + i * 7u) % input_words;
        x[i] = input[idx] ^ (seed + i * 11400714819323198485ULL);
        x[i] = x[i] * K[i] + A[i];
    }

    // -- predicates (runtime comparisons), with distant duplicates ---------
    int p[48];
    #pragma unroll 2
    for (unsigned int i = 0u; i < 48u; ++i)
    {
        unsigned int amt = (unsigned int)((seed + i * 5u) & 63u);
        p[i] = (int)(x[i] < (seed >> amt)) | (int)((x[i] >> 1u) >= (seed ^ i));
    }
    // duplicated comparisons at distance: identical operands as p[src] far away.
    #pragma unroll 2
    for (unsigned int i = 43u; i < 48u; ++i)
    {
        unsigned int src = (i * 7u) % 48u;
        unsigned int amt = (unsigned int)((seed + src * 5u) & 63u);
        p[i] = (int)(x[src] < (seed >> amt)) | (int)((x[src] >> 1u) >= (seed ^ src));
    }

    // -- long GPR-heavy region (register pressure between predicate uses) --
    unsigned long long acc = seed ^ 0x123456789ABCDEF0ULL;
    for (unsigned int j = 0u; j < 96u; ++j)
    {
        unsigned int idx = (tid + j * 3u + 1u) % input_words;
        acc = acc * 6364136223846793005ULL + (input[idx] ^ x[j % 48u]);
        acc ^= acc >> 29u;
    }

    // -- small diamonds ----------------------------------------------------
    for (unsigned int d = 0u; d < 16u; ++d)
    {
        int pa = p[d % 48u], pb = p[(d + 8u) % 48u];
        unsigned long long t = x[d % 48u] * K[d % 48u] + seed;
        if (pa && pb) acc += t;
        else          acc ^= (t >> 1u);
    }

    // -- forward branch use ------------------------------------------------
    for (unsigned int i = 0u; i < 36u; ++i)
    {
        if (p[i]) acc ^= x[i];
        else      acc += (x[i] >> 1u);
    }

    // -- reverse select use -------------------------------------------------
    for (unsigned int i = 36u; i < 48u; ++i)
    {
        unsigned long long t = (x[i] * K[i]) ^ seed;
        acc = p[i] ? (acc + t) : (acc ^ t);
    }

    // -- fold all predicates into the signature (predicate -> GPR) ---------
    unsigned long long mask = 0ULL;
    for (unsigned int i = 0u; i < 48u; ++i)
        mask |= ((unsigned long long)(unsigned int)p[i]) << (i % 64u);

    output[tid * 2u] = acc ^ mask;
    output[tid * 2u + 1u] = acc ^ seed ^ mask;
}
