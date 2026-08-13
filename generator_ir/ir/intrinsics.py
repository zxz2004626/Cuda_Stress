"""GPU intrinsic catalog.

A declarative table of GPU builtin functions: name, result type, parameter
types, minimum arch, category, and parameter roles.  The entries describe
public CUDA/HIP device API facts and are used by generators/mutations to kx_pick
arch-legal intrinsics.

Type codes used in the table:
    v void      b bool      i int        u unsigned int
    f float     d double    h __half     x bfloat16
    l long long L unsigned long long     P<x> pointer to <x>

Parameter roles: value | mask | const_small | ptr | predicate |
out_ptr_float | out_ptr_double
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from generator_ir.ir.typedecl import GType, T, TPointer


@dataclass(frozen=True)
class Intrinsic:
    name: str
    result: GType
    params: Tuple[GType, ...]
    kx_min_arch: int
    category: str
    roles: Tuple[str, ...]


def _decode(code: str) -> GType:
    if code.startswith("P<"):
        return TPointer(_decode(code[2:-1]))
    table = {
        "v": T.void, "b": T.boolean, "i": T.i32, "u": T.u32,
        "f": T.f32, "d": T.f64, "h": T.f16, "x": T.bf16,
        "l": T.i64, "L": T.u64,
    }
    return table[code]


# Table rows: (name, result, [params], kx_min_arch, category, [roles]).
# A bare list entry of the form ("*", (codes,)) expands one row per code
# substitution -- see _expand below.
_ROWS: List[Tuple] = [
    # -- warp_reduce (sm_80) -------------------------------------------
    *[("__reduce_%s_sync" % op, r, ["u", p], 80, "warp_reduce", ["mask", "value"])
      for op in ("add", "min", "max") for r, p in (("u", "u"), ("i", "i"))],
    *[("__reduce_%s_sync" % op, "u", ["u", "u"], 80, "warp_reduce", ["mask", "value"])
      for op in ("and", "or", "xor")],

    # -- warp_match (sm_70) --------------------------------------------
    *[("__match_any_sync", "u", ["u", p], 70, "warp_match", ["mask", "value"])
      for p in ("i", "u", "f")],

    # -- warp_vote (sm_70) ---------------------------------------------
    ("__all_sync", "i", ["u", "i"], 70, "warp_vote", ["mask", "predicate"]),
    ("__any_sync", "i", ["u", "i"], 70, "warp_vote", ["mask", "predicate"]),
    ("__ballot_sync", "u", ["u", "i"], 70, "warp_vote", ["mask", "predicate"]),

    # -- warp_shuffle (sm_70) ------------------------------------------
    # delta/lane is `int` for sync/xor and `unsigned int` for up/down (CUDA).
    *[("%s" % op, p, ["u", p, "i"], 70, "warp_shuffle", ["mask", "value", "const_small"])
      for op in ("__shfl_sync", "__shfl_xor_sync") for p in ("i", "u", "f")],
    *[("%s" % op, p, ["u", p, "u"], 70, "warp_shuffle", ["mask", "value", "const_small"])
      for op in ("__shfl_up_sync", "__shfl_down_sync") for p in ("i", "u", "f")],

    # -- sync -----------------------------------------------------------
    ("__syncthreads_count", "i", ["i"], 0, "sync", ["predicate"]),
    ("__syncthreads_and", "i", ["i"], 0, "sync", ["predicate"]),
    ("__syncthreads_or", "i", ["i"], 0, "sync", ["predicate"]),
    ("__syncwarp", "v", ["u"], 70, "sync", ["mask"]),
    ("__activemask", "u", [], 70, "sync", []),

    # -- bit manipulation ------------------------------------------------
    ("__ffs", "i", ["i"], 0, "bit_manip", ["value"]),
    ("__clz", "i", ["i"], 0, "bit_manip", ["value"]),
    ("__popc", "i", ["i"], 0, "bit_manip", ["value"]),
    ("__popc", "i", ["u"], 0, "bit_manip", ["value"]),
    ("__brev", "u", ["u"], 0, "bit_manip", ["value"]),
    ("__byte_perm", "u", ["u", "u", "u"], 0, "bit_manip", ["value", "value", "const_small"]),
    *[("__funnelshift_%s" % s, "u", ["u", "u", "u"], 0, "bit_manip",
       ["value", "value", "const_small"]) for s in ("l", "r", "lc", "rc")],

    # -- integer math -----------------------------------------------------
    ("__mul24", "i", ["i", "i"], 0, "int_math", ["value", "value"]),
    ("__umul24", "u", ["u", "u"], 0, "int_math", ["value", "value"]),
    ("__mulhi", "i", ["i", "i"], 0, "int_math", ["value", "value"]),
    ("__umulhi", "u", ["u", "u"], 0, "int_math", ["value", "value"]),
    ("__sad", "u", ["i", "i", "u"], 0, "int_math", ["value", "value", "value"]),
    ("__usad", "u", ["u", "u", "u"], 0, "int_math", ["value", "value", "value"]),
    ("__dp4a", "i", ["u", "u", "i"], 61, "int_math", ["value", "value", "value"]),
    ("__dp2a_lo", "i", ["u", "u", "i"], 61, "int_math", ["value", "value", "value"]),
    ("__dp2a_hi", "i", ["u", "u", "i"], 61, "int_math", ["value", "value", "value"]),

    # -- fast math (single precision) -------------------------------------
    *[("%s" % n, "f", ["f"], 0, "fast_math", ["value"])
      for n in ("__sinf", "__cosf", "__tanf", "__expf", "__logf", "__log2f",
                "__log10f", "__rsqrtf", "__saturatef")],
    ("__powf", "f", ["f", "f"], 0, "fast_math", ["value", "value"]),
    ("__fdividef", "f", ["f", "f"], 0, "fast_math", ["value", "value"]),
    # float rounding-mode arithmetic
    *[("__%s_%s" % (base, rm), "f", ["f"] * arity, 0, "fast_math", ["value"] * arity)
      for rm in ("rn", "rd", "ru", "rz")
      for base, arity in (("fmaf", 3), ("fadd", 2), ("fsub", 2), ("fmul", 2),
                          ("fdiv", 2), ("frcp", 1), ("fsqrt", 1))],

    # -- std math (float) --------------------------------------------------
    *[("%s" % n, "f", ["f"], 0, "std_math", ["value"])
      for n in ("sinf", "cosf", "tanf", "expf", "exp2f", "logf", "log2f",
                "sqrtf", "rsqrtf", "cbrtf", "ceilf", "floorf", "truncf",
                "roundf", "rintf", "fabsf", "asinf", "acosf", "atanf",
                "sinhf", "coshf", "tanhf", "asinhf", "acoshf", "atanhf",
                "erff", "erfcf", "lgammaf", "tgammaf", "expm1f", "log1pf",
                "logbf", "j0f", "j1f", "y0f", "y1f")],
    *[("%s" % n, "f", ["f", "f"], 0, "std_math", ["value", "value"])
      for n in ("fmaxf", "fminf", "copysignf", "fmodf", "remainderf", "fdimf",
                "hypotf", "atan2f", "powf", "nextafterf")],
    ("fmaf", "f", ["f", "f", "f"], 0, "std_math", ["value", "value", "value"]),
    ("scalbnf", "f", ["f", "i"], 0, "std_math", ["value", "const_small"]),
    ("ldexpf", "f", ["f", "i"], 0, "std_math", ["value", "const_small"]),
    ("sincosf", "v", ["f", "P<f>", "P<f>"], 0, "std_math",
     ["value", "out_ptr_float", "out_ptr_float"]),

    # -- double math --------------------------------------------------------
    *[("%s" % n, "d", ["d"], 0, "double_math", ["value"])
      for n in ("sin", "cos", "tan", "exp", "exp2", "log", "log2", "log10",
                "sqrt", "rsqrt", "cbrt", "ceil", "floor", "trunc", "round",
                "rint", "fabs", "asin", "acos", "atan", "sinh", "cosh", "tanh",
                "erf", "erfc", "lgamma", "tgamma", "j0", "j1", "y0", "y1")],
    *[("%s" % n, "d", ["d", "d"], 0, "double_math", ["value", "value"])
      for n in ("fmax", "fmin", "fmod", "remainder", "fdim", "hypot",
                "atan2", "pow", "copysign", "nextafter")],
    ("fma", "d", ["d", "d", "d"], 0, "double_math", ["value", "value", "value"]),
    ("scalbn", "d", ["d", "i"], 0, "double_math", ["value", "const_small"]),
    ("ldexp", "d", ["d", "i"], 0, "double_math", ["value", "const_small"]),
    ("sincos", "v", ["d", "P<d>", "P<d>"], 0, "std_math",
     ["value", "out_ptr_double", "out_ptr_double"]),
    # double rounding-mode arithmetic
    *[("__%s_%s" % (base, rm), "d", ["d"] * arity, 0, "double_math", ["value"] * arity)
      for rm in ("rn", "rd", "ru", "rz")
      for base, arity in (("dadd", 2), ("dsub", 2), ("dmul", 2), ("ddiv", 2),
                          ("drcp", 1), ("dsqrt", 1), ("fma", 3))],
    *[("__double2float_%s" % rm, "f", ["d"], 0, "double_math", ["value"])
      for rm in ("rn", "rd", "ru", "rz")],

    # -- type casting ---------------------------------------------------------
    ("__float_as_int", "i", ["f"], 0, "type_cast", ["value"]),
    ("__int_as_float", "f", ["i"], 0, "type_cast", ["value"]),
    ("__float_as_uint", "u", ["f"], 0, "type_cast", ["value"]),
    ("__uint_as_float", "f", ["u"], 0, "type_cast", ["value"]),
    ("__double_as_longlong", "l", ["d"], 0, "type_cast", ["value"]),
    ("__longlong_as_double", "d", ["l"], 0, "type_cast", ["value"]),
    ("__hiloint2double", "d", ["i", "i"], 0, "type_cast", ["value", "value"]),
    ("__double2hiint", "i", ["d"], 0, "type_cast", ["value"]),
    ("__double2loint", "i", ["d"], 0, "type_cast", ["value"]),
    *[("__float2int_%s" % rm, "i", ["f"], 0, "type_cast", ["value"]) for rm in ("rn", "rd", "ru", "rz")],
    *[("__float2uint_%s" % rm, "u", ["f"], 0, "type_cast", ["value"]) for rm in ("rn", "rd", "ru", "rz")],
    *[("__int2float_%s" % rm, "f", ["i"], 0, "type_cast", ["value"]) for rm in ("rn", "rd", "ru", "rz")],
    *[("__uint2float_%s" % rm, "f", ["u"], 0, "type_cast", ["value"]) for rm in ("rn", "rd", "ru", "rz")],
    *[("__double2int_%s" % rm, "i", ["d"], 0, "type_cast", ["value"]) for rm in ("rn", "rd", "ru", "rz")],
    *[("__double2uint_%s" % rm, "u", ["d"], 0, "type_cast", ["value"]) for rm in ("rn", "rd", "ru", "rz")],
    *[("__double2ll_%s" % rm, "l", ["d"], 0, "type_cast", ["value"]) for rm in ("rn", "rd", "ru", "rz")],
    *[("__double2ull_%s" % rm, "L", ["d"], 0, "type_cast", ["value"]) for rm in ("rn", "rd", "ru", "rz")],
    *[("__int2double_%s" % rm, "d", ["i"], 0, "type_cast", ["value"]) for rm in ("rn", "rd", "ru", "rz")],
    *[("__uint2double_%s" % rm, "d", ["u"], 0, "type_cast", ["value"]) for rm in ("rn", "rd", "ru", "rz")],
    *[("__ll2float_%s" % rm, "f", ["l"], 0, "type_cast", ["value"]) for rm in ("rn", "rd", "ru", "rz")],
    *[("__ull2float_%s" % rm, "f", ["L"], 0, "type_cast", ["value"]) for rm in ("rn", "rd", "ru", "rz")],
    *[("__ll2double_%s" % rm, "d", ["l"], 0, "type_cast", ["value"]) for rm in ("rn", "rd", "ru", "rz")],
    *[("__ull2double_%s" % rm, "d", ["L"], 0, "type_cast", ["value"]) for rm in ("rn", "rd", "ru", "rz")],

    # -- cache load hints ------------------------------------------------------
    *[("__%s" % n, r, ["P<%s>" % r], 0 if n != "ldlu" else 80, "cache_load", ["ptr"])
      for n in ("ldg", "ldcg", "ldca", "ldcs", "ldlu") for r in ("u", "i", "f")],

    # -- cache store hints ------------------------------------------------------
    *[("__%s" % n, "v", ["P<%s>" % r, r], 0, "cache_store", ["ptr", "value"])
      for n in ("stwb", "stcg", "stcs", "stwt") for r in ("u", "i", "f")],

    # -- half math ---------------------------------------------------------------
    ("__float2half", "h", ["f"], 53, "half_math", ["value"]),
    ("__half2float", "f", ["h"], 53, "half_math", ["value"]),
    *[("__float2half_%s" % rm, "h", ["f"], 53, "half_math", ["value"]) for rm in ("rn", "rd", "ru", "rz")],
    *[("__%s" % n, "h", ["h", "h"], 53, "half_math", ["value", "value"])
      for n in ("hadd", "hsub", "hmul", "hdiv")],
    ("__hfma", "h", ["h", "h", "h"], 53, "half_math", ["value", "value", "value"]),
    ("__habs", "h", ["h"], 53, "half_math", ["value"]),
    ("__hneg", "h", ["h"], 53, "half_math", ["value"]),
    ("__hmax", "h", ["h", "h"], 80, "half_math", ["value", "value"]),
    ("__hmin", "h", ["h", "h"], 80, "half_math", ["value", "value"]),
    *[("%s" % n, "h", ["h"], 53, "half_math", ["value"])
      for n in ("hexp", "hexp2", "hexp10", "hlog", "hlog2", "hlog10", "hsin",
                "hcos", "hsqrt", "hrsqrt", "hceil", "hfloor", "htrunc", "hrint")],
    *[("__%s" % n, "b", ["h", "h"], 53, "half_math", ["value", "value"])
      for n in ("heq", "hne", "hlt", "hgt", "hle", "hge")],
    *[("__%s" % n, "h", ["h", "h"], 53, "half_math", ["value", "value"])
      for n in ("hadd_sat", "hsub_sat", "hmul_sat")],
    ("__hfma_sat", "h", ["h", "h", "h"], 53, "half_math", ["value", "value", "value"]),

    # -- half2 helpers ------------------------------------------------------------
    ("__float2half2_rn", "h", ["f"], 53, "half2_math", ["value"]),
    ("__half2half2", "h", ["h"], 53, "half2_math", ["value"]),
    ("__low2half", "h", ["h"], 53, "half2_math", ["value"]),
    ("__high2half", "h", ["h"], 53, "half2_math", ["value"]),

    # -- bf16 math ------------------------------------------------------------------
    ("__float2bfloat16", "x", ["f"], 80, "bf16_math", ["value"]),
    ("__bfloat162float", "f", ["x"], 80, "bf16_math", ["value"]),
    *[("__float2bfloat16_%s" % rm, "x", ["f"], 80, "bf16_math", ["value"]) for rm in ("rn", "rd", "ru", "rz")],
    *[("__%s" % n, "x", ["x", "x"], 80, "bf16_math", ["value", "value"])
      for n in ("hadd", "hsub", "hmul", "hdiv", "hmax", "hmin")],
    ("__hfma", "x", ["x", "x", "x"], 80, "bf16_math", ["value", "value", "value"]),
    ("__habs", "x", ["x"], 80, "bf16_math", ["value"]),
    ("__hneg", "x", ["x"], 80, "bf16_math", ["value"]),
    *[("__%s" % n, "b", ["x", "x"], 80, "bf16_math", ["value", "value"])
      for n in ("heq", "hne", "hlt", "hgt", "hle", "hge")],

    # -- DPX (sm_90) ------------------------------------------------------------------
    *[("__%s" % n, r, [r] * arity, 90, "dpx", ["value"] * arity)
      for n, arity, r in (
          ("vimax3_s32", 3, "i"), ("vimin3_s32", 3, "i"),
          ("vimax3_u32", 3, "u"), ("vimin3_u32", 3, "u"),
          ("viaddmax_s32", 3, "i"), ("viaddmin_s32", 3, "i"),
          ("viaddmax_u32", 3, "u"), ("viaddmin_u32", 3, "u"),
          # The plain 2-arg forms (vimax_s32/vimin_s32/vimax_u32/vimin_u32)
          # were removed in CUDA 13 -- only the _relu spellings remain.
          ("vimax_s32_relu", 2, "i"), ("vimin_s32_relu", 2, "i"),
          ("vimax3_s32_relu", 3, "i"), ("vimin3_s32_relu", 3, "i"),
          ("viaddmax_s32_relu", 3, "i"), ("viaddmin_s32_relu", 3, "i"),
          ("vimax3_s16x2", 3, "u"), ("vimin3_s16x2", 3, "u"),
          ("vimax3_u16x2", 3, "u"), ("vimin3_u16x2", 3, "u"),
          ("viaddmax_s16x2", 3, "u"), ("viaddmin_s16x2", 3, "u"),
          ("viaddmax_u16x2", 3, "u"), ("viaddmin_u16x2", 3, "u"),
          ("vimax_s16x2_relu", 2, "u"), ("vimin_s16x2_relu", 2, "u"),
          ("vimax3_s16x2_relu", 3, "u"), ("vimin3_s16x2_relu", 3, "u"),
          ("viaddmax_s16x2_relu", 3, "u"), ("viaddmin_s16x2_relu", 3, "u"),
      )],

    # -- float predicates ----------------------------------------------------------------
    ("__isnanf", "i", ["f"], 0, "float_predicate", ["value"]),
    ("__isinff", "i", ["f"], 0, "float_predicate", ["value"]),
    ("__signbitf", "i", ["f"], 0, "float_predicate", ["value"]),
    ("__isnan", "i", ["d"], 0, "float_predicate", ["value"]),
    ("__isinf", "i", ["d"], 0, "float_predicate", ["value"]),
    *[("%s" % n, "i", [t], 0, "float_predicate", ["value"])
      for n in ("isfinite", "isinf", "isnan", "signbit") for t in ("f", "d")],

    # -- atomics ---------------------------------------------------------------------------
    *[("atomicAdd", r, ["P<%s>" % r, r], arch, "atomic_base", ["ptr", "value"])
      for r, arch in (("i", 0), ("u", 0), ("f", 0), ("d", 60), ("l", 0), ("L", 0))],
    *[("atomicSub", r, ["P<%s>" % r, r], 0, "atomic_base", ["ptr", "value"]) for r in ("i", "u")],
    *[("atomic%s" % n, r, ["P<%s>" % r, r], 0, "atomic_base", ["ptr", "value"])
      for n in ("Min", "Max", "And", "Or", "Xor") for r in ("i", "u", "l", "L")],
    ("atomicInc", "u", ["P<u>", "u"], 0, "atomic_special", ["ptr", "value"]),
    ("atomicDec", "u", ["P<u>", "u"], 0, "atomic_special", ["ptr", "value"]),
    *[("atomicExch", r, ["P<%s>" % r, r], 0, "atomic_special", ["ptr", "value"])
      for r in ("u", "i", "f", "l", "L")],
    *[("atomicCAS", r, ["P<%s>" % r, r, r], 0, "atomic_special", ["ptr", "value", "value"])
      for r in ("u", "i", "L")],

    # -- scoped atomics (sm_60+) --------------------------------------------------------------
    *[("atomic%s_block" % n, r, ["P<%s>" % r, r], 60, "scoped_atomic", ["ptr", "value"])
      for n in ("Add", "Sub", "Min", "Max", "And", "Or", "Xor", "Exch") for r in ("u", "i")],
    *[("atomicCAS_block", r, ["P<%s>" % r, r, r], 60, "scoped_atomic", ["ptr", "value", "value"])
      for r in ("u", "i")],
    ("atomicAdd_block", "f", ["P<f>", "f"], 60, "scoped_atomic", ["ptr", "value"]),
    *[("atomicAdd_system", r, ["P<%s>" % r, r], 60, "scoped_atomic", ["ptr", "value"])
      for r in ("u", "i")],
    *[("atomic%s_system" % n, r, ["P<%s>" % r, r], 60, "scoped_atomic", ["ptr", "value"])
      for n in ("Sub", "Min", "Max", "And", "Or", "Xor", "Exch") for r in ("u", "i")],
    *[("atomicCAS_system", r, ["P<%s>" % r, r, r], 60, "scoped_atomic", ["ptr", "value", "value"])
      for r in ("u", "i")],
    ("atomicAdd_system", "f", ["P<f>", "f"], 60, "scoped_atomic", ["ptr", "value"]),

    # -- address space predicates + conversions (sm_90 for __isGridConstant) ----
    *[(fn, "u", ["P<u>"], 0, "addr_space", ["ptr"])
      for fn in ("__isGlobal", "__isShared", "__isConstant", "__isLocal")],
    ("__isGridConstant", "u", ["P<u>"], 90, "addr_space", ["ptr"]),
    *[(fn, "P<u>", ["P<u>"], 0, "addr_space", ["ptr"])
      for fn in ("__cvta_generic_to_global", "__cvta_generic_to_shared",
                 "__cvta_generic_to_local", "__cvta_generic_to_constant",
                 "__cvta_global_to_generic", "__cvta_shared_to_generic",
                 "__cvta_local_to_generic", "__cvta_constant_to_generic")],

    # -- 64-bit integer math ---------------------------------------------------------------------
    ("__mul64hi", "l", ["l", "l"], 0, "longlong_math", ["value", "value"]),
    ("__umul64hi", "L", ["L", "L"], 0, "longlong_math", ["value", "value"]),
    ("__clzll", "i", ["l"], 0, "longlong_math", ["value"]),
    ("__ffsll", "i", ["l"], 0, "longlong_math", ["value"]),
    ("__popcll", "i", ["l"], 0, "longlong_math", ["value"]),
    ("__brevll", "L", ["L"], 0, "longlong_math", ["value"]),

    # -- pipeline (sm_80) ---------------------------------------------------------------------------
    ("__pipeline_commit", "v", [], 80, "pipeline", []),
    ("__pipeline_wait_prior", "v", ["u"], 80, "pipeline", ["const_small"]),

    # -- dependent launch (sm_90) ----------------------------------------------------------------------
    ("cudaTriggerProgrammaticLaunchCompletion", "v", [], 90, "dependent_launch", []),
    ("cudaGridDependencySynchronize", "v", [], 90, "dependent_launch", []),

    # -- vector constructors ------------------------------------------------------------------------------
    *[("make_%s%d" % (base, n), r, [r] * n, 0, "vector_make", ["value"] * n)
      for n in (2, 3, 4) for base, r in (("int", "i"), ("uint", "u"), ("float", "f"),
                                         ("short", "i"), ("ushort", "u"),
                                         ("char", "i"), ("uchar", "u"))],
    ("make_double2", "d", ["d", "d"], 0, "vector_make", ["value", "value"]),
    ("make_longlong2", "l", ["l", "l"], 0, "vector_make", ["value", "value"]),
    ("make_ulonglong2", "L", ["L", "L"], 0, "vector_make", ["value", "value"]),

    # -- compiler hints ---------------------------------------------------------------------------------------
    ("__builtin_expect", "i", ["i", "i"], 0, "compiler_hint", ["value", "const_small"]),
    ("__builtin_unreachable", "v", [], 0, "compiler_hint", []),
    ("__assume", "v", ["i"], 0, "compiler_hint", ["predicate"]),

    # -- misc ---------------------------------------------------------------------------------------------------
    ("__nanosleep", "v", ["u"], 70, "misc", ["const_small"]),
    ("__threadfence", "v", [], 0, "misc", []),
    ("__threadfence_block", "v", [], 0, "misc", []),
    ("__threadfence_system", "v", [], 0, "misc", []),
    ("clock", "i", [], 0, "misc", []),
    ("clock64", "l", [], 0, "misc", []),
    ("__trap", "v", [], 0, "misc", []),
    ("__brkpt", "v", [], 0, "misc", []),
    *[("__lanemask_%s" % s, "u", [], 0, "misc", []) for s in ("eq", "lt", "le", "gt", "ge")],
]


class Catalog:
    """Query interface over the intrinsic table."""

    def __init__(self) -> None:
        self._all: List[Intrinsic] = [
            Intrinsic(name, _decode(res), tuple(_decode(p) for p in params),
                      arch, cat, tuple(roles))
            for name, res, params, arch, cat, roles in _ROWS
        ]
        self._by_cat: Dict[str, List[Intrinsic]] = {}
        for it in self._all:
            self._by_cat.setdefault(it.category, []).append(it)

    def available(self, arch_version: int) -> List[Intrinsic]:
        return [i for i in self._all if i.kx_min_arch <= arch_version]

    def in_category(self, category: str, arch_version: int) -> List[Intrinsic]:
        return [i for i in self._by_cat.get(category, [])
                if i.kx_min_arch <= arch_version]

    def categories(self, arch_version: int) -> List[str]:
        return sorted({i.category for i in self.available(arch_version)})

    def named(self, name: str, arch_version: int) -> List[Intrinsic]:
        return [i for i in self._all
                if i.name == name and i.kx_min_arch <= arch_version]

    def count(self, arch_version: int = 999) -> int:
        return len(self.available(arch_version))


CATALOG = Catalog()
