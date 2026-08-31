"""Verbatim stress-section builders for post-loop insertion.

Split from assembler.py for modularity. Each constant/function produces a
self-contained C block that folds into _fv.
"""

from __future__ import annotations

_STRESS_BANK_CONFLICT = """\
{ // bank-conflict stress
    __shared__ unsigned int _bc_smem[1024];
    unsigned int _bc_lane = threadIdx.x % 32u;
    unsigned int _bc_warp = threadIdx.x / 32u;
    for (unsigned int _bc_i = threadIdx.x; _bc_i < 1024u; _bc_i += blockDim.x)
        _bc_smem[_bc_i] = _bc_i * 0x9E3779B9u;
    __syncthreads();
    unsigned int _bc_val = _bc_smem[_bc_lane * 32u + _bc_warp];
    __syncthreads();
    _bc_val ^= _bc_smem[(threadIdx.x * 17u) % 1024u];
    __syncthreads();
    _fv ^= _bc_val;
}"""


def _stress_register_pressure(level: str) -> str:
    n = {"none": 0, "low": 24, "medium": 48, "high": 80, "extreme": 128}.get(level, 0)
    if n == 0:
        return ""
    lines = ["{ // register-pressure stress"]
    for i in range(n):
        lines.append(f"    unsigned int _rp_{i} = threadIdx.x ^ {i}u;")
    sync = level in ("medium", "high", "extreme")
    for rnd in range(3):
        if sync and rnd > 0:
            lines.append("    __syncthreads();")
        for i in range(n):
            j = (i + 1) % n
            k = (i + 7) % n
            lines.append(f"    _rp_{i} ^= _rp_{j} + _rp_{k};")
    for i in range(n):
        lines.append(f"    _fv ^= _rp_{i};")
    lines.append("}")
    return "\n".join(lines)


_STRESS_VECTORIZE = """\
{ // vectorized-load stress
    unsigned int _vl_base = tid * 4u;
    if (_vl_base + 3u < (unsigned int)n) {
        unsigned int _vl_a = input[_vl_base];
        unsigned int _vl_b = input[_vl_base + 1u];
        unsigned int _vl_c = input[_vl_base + 2u];
        unsigned int _vl_d = input[_vl_base + 3u];
        _fv ^= _vl_a ^ _vl_b ^ _vl_c ^ _vl_d;
    }
    unsigned int _vl_e = 0u, _vl_f = 0u, _vl_g = 0u, _vl_h = 0u;
    unsigned int _vl_base2 = (tid ^ 1u) * 4u;
    if (_vl_base2 + 3u < (unsigned int)n) {
        _vl_e = input[_vl_base2];
        _vl_f = input[_vl_base2 + 1u];
        _vl_g = input[_vl_base2 + 2u];
        _vl_h = input[_vl_base2 + 3u];
    }
    __syncthreads();
    _fv ^= _vl_e ^ _vl_f ^ _vl_g ^ _vl_h;
}"""

_STRESS_SWITCH = """\
{ // switch-divergence stress
    unsigned int _sd_acc = tid ^ 0xDEADBEEFu;
    switch (threadIdx.x % 8u) {
        case 0: _sd_acc = _sd_acc * 0x9E3779B9u ^ (_sd_acc >> 16u); break;
        case 1: _sd_acc = (_sd_acc ^ 0x85EBCA6Bu) * 0xC2B2AE35u; break;
        case 2: _sd_acc = _sd_acc + ((_sd_acc << 13u) ^ (_sd_acc >> 7u)); break;
        case 3: _sd_acc = (_sd_acc * 0x6C62272Eu) ^ (_sd_acc >> 15u) ^ (_sd_acc << 5u); break;
        case 4: _sd_acc = (_sd_acc + 0x517CC1B7u) ^ (_sd_acc * 0x27D4EB2Fu); break;
        case 5: _sd_acc = _sd_acc ^ (_sd_acc >> 11u) ^ (_sd_acc << 19u); break;
        case 6: _sd_acc = (_sd_acc * 3u) ^ (_sd_acc >> 8u) ^ 0xA3B1BACDu; break;
        default: _sd_acc = (_sd_acc ^ 0xB5026F5Au) + (_sd_acc >> 3u); break;
    }
    if (threadIdx.x < 16u) {
        unsigned int _sd_x0 = _sd_acc ^ 0x1u;
        unsigned int _sd_x1 = _sd_acc + _sd_x0;
        unsigned int _sd_x2 = _sd_x1 * 0x9E3779B9u;
        unsigned int _sd_x3 = _sd_x2 ^ (_sd_x1 >> 5u);
        _fv ^= _sd_x0 ^ _sd_x1 ^ _sd_x2 ^ _sd_x3;
    } else {
        _fv ^= _sd_acc;
    }
}"""

_STRESS_BARRIER_SPAN = """\
{ // barrier-span stress (int / long long / pointer cost classes)
    unsigned int _bs_int0 = tid * 0x9E3779B9u;
    unsigned int _bs_int1 = tid ^ (_bs_int0 >> 8u);
    unsigned int _bs_int2 = _bs_int0 + _bs_int1;
    unsigned long long _bs_ll0 = (unsigned long long)tid * 0x9E3779B97F4A7C15ULL;
    unsigned long long _bs_ll1 = _bs_ll0 ^ ((unsigned long long)_bs_int0 << 32u);
    unsigned int _bs_ptr0 = (unsigned int)((unsigned long long)input & 0xFFFFFFFFu) ^ tid;
    __syncthreads();
    _fv ^= _bs_int0 ^ _bs_int1 ^ _bs_int2;
    _fv ^= (unsigned int)_bs_ll0 ^ (unsigned int)(_bs_ll1 >> 32u);
    _fv ^= _bs_ptr0;
}"""


def _stress_fp_island(rounds: int, trans: str) -> str:
    return f"""\
{{ // FP island (FFMA / MUFU / double codegen), folds into _fv
    float _fa = (float)(input[tid % (unsigned int)n] & 0xFFFFu) * 1.000113f + 0.5f;
    float _fb = (float)((input[(tid + 1u) % (unsigned int)n] >> 8) & 0xFFFFu) * 0.999737f - 0.25f;
    double _fd = (double)_fa * 1.0000000731 + 3.14159265358979;
    #pragma unroll 1
    for (int _fi = 0; _fi < {rounds}; _fi++) {{
        _fa = _fa * _fb + 1.5f;
        _fb = {trans}(_fa) + _fb * 0.5f;
        _fd = _fd * 0.9999997 + (double)_fa;
        _fa = _fa - (float)((int)_fa);
    }}
    _fa += (float)(_fd - (double)((long long)_fd));
    _fv ^= __float_as_uint(_fa) ^ (unsigned int)(__double_as_longlong(_fd) >> 11);
}}"""


# ---------------------------------------------------------------------------
# public entry
# ---------------------------------------------------------------------------

# Aliases for the assembler's _build_stress_sections.
BANK_CONFLICT_BLOCK = _STRESS_BANK_CONFLICT
VECTORIZE_BLOCK = _STRESS_VECTORIZE
SWITCH_DIVERGENCE_BLOCK = _STRESS_SWITCH
BARRIER_SPAN_BLOCK = _STRESS_BARRIER_SPAN
REG_PRESSURE_BLOCK = _stress_register_pressure
fp_island_block = _stress_fp_island
