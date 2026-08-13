"""Branchy / special floating-point *library* weave blocks.

Port of perffuzzer/generator/codegen/gens_fp_lib.py.

Adds the math-library functions the fuzzer never emitted: the branchy /
special-value transcendentals (``asinf``/``tgammaf``/Bessel
``j0f``..``y1f``/``lgammaf``/...), the CUDA-libdevice-only specials
(``erfinvf``/``normcdff``/``rcbrtf``/``cospif``/...), a double special-value
*classification* family (``isinf``/``isnan``/``isfinite``/``signbit`` driving
an FP select), and a double division-refinement nudge (``a/b`` Newton path).

Why this is LOW-RISK / UB-FREE by construction (same contract as fp.py):
  * Every value produced here is folded through ``ctx.kx_observe_fp`` into the
    SEPARATE off-checksum accumulator (``output_fp_acc``), which the
    kx_translator EXCLUDES from the old==new integer-checksum comparison.  So a
    cross-compiler FP difference lands off-checksum and can NEVER become a
    false miscompile.
  * FP arithmetic is fully defined and deterministic within a single binary:
    div-by-zero -> +-inf, 0/0 -> nan, out-of-domain transcendentals -> nan.
  * The ONLY genuine FP UB is an out-of-range ``(int)float`` conversion.  We
    never convert FP->int here at all (the categorize family reads bits via
    ``__longlong_as_double`` / ``__ull2double_rn`` and keeps everything FP).

The integer variable pool holds NO floating-point vars, so each block SEEDS
its own FP inputs with ``_kx_fp_seed`` (copied from weave_blocks.fp per the
perffuzzer brief: copying is more robust than importing).

Backend gating:
  * ``fp_lib_transcendental_f`` / ``fp_lib_transcendental_d`` use standard
    libm math names available under both nvcc and hipcc -> both backends.
  * ``fp_lib_cuda_special`` uses CUDA-libdevice-only names (``erfinvf`` etc.)
    that hipcc/HIP-libm does not provide -> ``backends=("cuda",)``.
  * ``fp_double_classify`` / ``fp_double_div_loop`` are portable -> both.
"""

from __future__ import annotations

from generator_ir.ir.typedecl import T, GType, TDouble
from generator_ir.ir.operators import Arith
from generator_ir.ir.nodes import Expr, Lit, Var, BinOp, Call, Select
from generator_ir.gen.context import GenContext
from generator_ir.gen.weave_handlers import weave_block


# ---------------------------------------------------------------------------
# FP seeding helper -- the pool has no FP vars, so make our own.
# (Copied from weave_blocks/fp.py per the brief: copying is more robust than
# importing.)
# ---------------------------------------------------------------------------

_FLOAT_LITS = [0.5, 1.0, -1.0, 1.5, 2.0, 3.25, 0.125, 7.0, -0.75, 42.0]


def _kx_fp_seed(ctx: GenContext, t: GType) -> Var:
    """Declare and return a fresh FP variable of type *t* with a defined init.

    Half the time the seed is reinterpreted from a live integer pool variable
    (via ``__uint2float_rn`` / ``__ull2double_rn``) so the FP island depends
    on real runtime data and cannot be constant-folded away; otherwise it is
    a small literal.  Both inits are defined for any input.
    """
    is_double = isinstance(t, TDouble)
    if ctx.rnd_bool(0.5):
        if is_double:
            src = ctx.kx_random_var(T.u64, recency=0.5)
            init: Expr = Call(t, fn="__ull2double_rn", args=[src],
                              result_ty=t)
        else:
            src = ctx.kx_random_var(T.u32, recency=0.5)
            init = Call(t, fn="__uint2float_rn", args=[src], result_ty=t)
    else:
        lit = ctx.rnd_choice(_FLOAT_LITS)
        init = Lit(t, value=float(lit))
    return ctx.fresh(t, init=init)


# ---------------------------------------------------------------------------
# Math-name pools.
# ---------------------------------------------------------------------------

# Single-precision branchy / special transcendentals (portable libm).
_LIB_TRANSC_F1 = [
    "asinf", "acosf", "atanf", "sinhf", "coshf", "tanhf", "cbrtf",
    "tgammaf", "lgammaf", "j0f", "j1f", "y0f", "y1f", "log1pf", "expm1f",
    "acoshf", "asinhf", "atanhf",
]
_LIB_TRANSC_F2 = [
    "atan2f", "powf", "fmodf", "remainderf", "hypotf", "fdimf", "copysignf",
]

# Double-precision branchy / special transcendentals (portable libm).
_LIB_TRANSC_D1 = [
    "asin", "acos", "atan", "sinh", "cosh", "tanh", "cbrt",
    "tgamma", "lgamma", "j0", "j1", "log1p", "expm1", "erf", "erfc",
]
_LIB_TRANSC_D2 = [
    "atan2", "pow", "fmod", "remainder", "hypot",
]

# CUDA-libdevice-only single/double specials (NOT available under hipcc).
_LIB_CUDA_SPECIAL_F = [
    "erfinvf", "erfcinvf", "normcdff", "normcdfinvf", "rcbrtf",
    "cospif", "sinpif",
]
_LIB_CUDA_SPECIAL_D = [
    "erfinv", "erfcinv", "normcdf", "normcdfinv", "rcbrt",
]

# Cheap single-arg specials used as the "branch payloads" for the categorize
# family (kept double, defined for any finite/inf/nan input).
_CLASSIFY_PAYLOAD_D = ["cbrt", "tgamma", "erf", "tanh", "atan", "expm1"]
_CLASSIFY_PRED = ["isinf", "isnan", "isfinite", "signbit"]


# ---------------------------------------------------------------------------
# 1) Single-precision branchy / special transcendentals -- portable.
# ---------------------------------------------------------------------------

@weave_block("fp_lib_transcendental_f", cost=2)
def _blk_fp_lib_transcendental_f(ctx: GenContext) -> None:
    """v = f(seed) | f(seed_a, seed_b) | g(f(x)+bias) -- SINGLE-precision
    branchy/special library calls (inverse-trig, hyperbolic, gamma, Bessel
    j0f/j1f/y0f/y1f, log1pf/expm1f, ...).

    These have different polynomial approximations / branch structure across
    nvcc versions.  The result is folded OFF-CHECKSUM via kx_observe_fp, so any
    FP divergence is harmless.  UB-free: transcendentals of any value (incl.
    out-of-domain -> nan) are defined and deterministic per binary; no FP->int
    conversion occurs.
    """
    t = T.f32
    if ctx.rnd_bool(0.45):
        # Two-argument special.
        a = _kx_fp_seed(ctx, t)
        b = _kx_fp_seed(ctx, t)
        name = ctx.rnd_choice(_LIB_TRANSC_F2)
        val: Expr = Call(t, fn=name, args=[a, b], result_ty=t)
    else:
        # One-argument special, optionally chained g(f(x) + bias).
        x = _kx_fp_seed(ctx, t)
        f = ctx.rnd_choice(_LIB_TRANSC_F1)
        val = Call(t, fn=f, args=[x], result_ty=t)
        if ctx.rnd_bool(0.5):
            bias = Lit(t, value=float(ctx.rnd_choice(_FLOAT_LITS)))
            summed = BinOp(t, op=Arith.ADD, lhs=val, rhs=bias)
            g = ctx.rnd_choice(_LIB_TRANSC_F1)
            val = Call(t, fn=g, args=[summed], result_ty=t)
    ctx.kx_observe_fp(ctx.fresh(t, init=val))


# ---------------------------------------------------------------------------
# 2) Double-precision branchy / special transcendentals -- portable.
# ---------------------------------------------------------------------------

@weave_block("fp_lib_transcendental_d", cost=2)
def _blk_fp_lib_transcendental_d(ctx: GenContext) -> None:
    """v = f(seed) | f(seed_a, seed_b) | g(f(x)+bias) -- DOUBLE-precision
    branchy/special library calls (inverse-trig, hyperbolic, gamma, Bessel
    j0/j1, log1p/expm1, erf/erfc, ...).

    Same rationale as the single-precision variant: cross-version FP-divergent
    by design, folded OFF-CHECKSUM via kx_observe_fp, UB-free (no FP->int).
    """
    t = T.f64
    if ctx.rnd_bool(0.45):
        a = _kx_fp_seed(ctx, t)
        b = _kx_fp_seed(ctx, t)
        name = ctx.rnd_choice(_LIB_TRANSC_D2)
        val: Expr = Call(t, fn=name, args=[a, b], result_ty=t)
    else:
        x = _kx_fp_seed(ctx, t)
        f = ctx.rnd_choice(_LIB_TRANSC_D1)
        val = Call(t, fn=f, args=[x], result_ty=t)
        if ctx.rnd_bool(0.5):
            bias = Lit(t, value=float(ctx.rnd_choice(_FLOAT_LITS)))
            summed = BinOp(t, op=Arith.ADD, lhs=val, rhs=bias)
            g = ctx.rnd_choice(_LIB_TRANSC_D1)
            val = Call(t, fn=g, args=[summed], result_ty=t)
    ctx.kx_observe_fp(ctx.fresh(t, init=val))


# ---------------------------------------------------------------------------
# 3) CUDA-libdevice-only specials -- backends=("cuda",) ONLY.
# ---------------------------------------------------------------------------

@weave_block("fp_lib_cuda_special", cost=2, backends=("cuda",))
def _blk_fp_lib_cuda_special(ctx: GenContext) -> None:
    """v = f(seed) -- CUDA-libdevice-specific specials: inverse error
    functions (erfinvf/erfcinvf), normal CDF + inverse
    (normcdff/normcdfinvf), reciprocal cube root (rcbrtf), and exact +-pi
    scaling (cospif/sinpif), plus their double versions
    (erfinv/erfcinv/normcdf/normcdfinv/rcbrt).

    Gated CUDA-only because hipcc / HIP-libm does not provide these names.
    The result folds OFF-CHECKSUM via kx_observe_fp; UB-free (defined for any
    input, no FP->int).
    """
    if ctx.rnd_bool(0.5):
        t = T.f32
        name = ctx.rnd_choice(_LIB_CUDA_SPECIAL_F)
    else:
        t = T.f64
        name = ctx.rnd_choice(_LIB_CUDA_SPECIAL_D)
    x = _kx_fp_seed(ctx, t)
    val = Call(t, fn=name, args=[x], result_ty=t)
    ctx.kx_observe_fp(ctx.fresh(t, init=val))


# ---------------------------------------------------------------------------
# 4) Double special-value CLASSIFICATION select (CLASS K) -- portable.
# ---------------------------------------------------------------------------

def _categorize_seed_double(ctx: GenContext) -> Var:
    """Seed a double that can REACH inf/nan/denormal special values.

    Half the time we reinterpret an arbitrary 64-bit pattern from the pool
    via ``__longlong_as_double`` (so the value can be inf/nan/denormal/
    anything); otherwise it is an ordinary value (literal or
    ``__ull2double_rn`` of a pool integer).  Both are defined for every input
    bit pattern -- reinterpretation is never UB, and the result stays FP.
    """
    t = T.f64
    if ctx.rnd_bool(0.5):
        src = ctx.kx_random_var(T.u64, recency=0.5)
        init = Call(t, fn="__longlong_as_double", args=[src], result_ty=t)
        return ctx.fresh(t, init=init)
    # Ordinary value path: reuse the standard seed helper.
    return _kx_fp_seed(ctx, t)


@weave_block("fp_double_classify", cost=2)
def _blk_fp_double_classify(ctx: GenContext) -> None:
    """v = combine of 2-3 categorize-driven FP selects (CLASS K).

    Seeds a double ``x`` that can hold inf/nan/denormal (via bit-reinterpret),
    then builds nested ternaries keyed on ``isinf`` / ``isnan`` / ``isfinite``
    / ``signbit`` whose branch payloads are cheap specials
    (cbrt/tgamma/erf/...) or scaled literals.

    nvcc versions lower the categorize predicates + the resulting select
    differently.  The whole thing is KEPT FP and folded OFF-CHECKSUM via
    kx_observe_fp, so even though ``x`` may be inf/nan there is ZERO correctness
    risk -- no value reaches the integer checksum, and no FP->int conversion
    occurs.
    """
    t = T.f64

    def _kx_payload(x: Expr) -> Expr:
        """A defined double payload: a cheap special of x, or x*const."""
        if ctx.rnd_bool(0.5):
            name = ctx.rnd_choice(_CLASSIFY_PAYLOAD_D)
            return Call(t, fn=name, args=[x], result_ty=t)
        k = Lit(t, value=float(ctx.rnd_choice(_FLOAT_LITS)))
        return BinOp(t, op=Arith.MUL, lhs=x, rhs=k)

    x = _categorize_seed_double(ctx)

    # Innermost value: another categorize select, so we combine 2-3 predicates.
    pred_name = ctx.rnd_choice(_CLASSIFY_PRED)
    cond = Call(T.boolean, fn=pred_name, args=[x], result_ty=T.boolean)
    val: Expr = Select(t, cond=cond, when_true=_kx_payload(x),
                       when_false=_kx_payload(x))

    n_extra = ctx.rnd_int(1, 2)  # total 2-3 classifies
    for _ in range(n_extra):
        pred_name = ctx.rnd_choice(_CLASSIFY_PRED)
        cond = Call(T.boolean, fn=pred_name, args=[x], result_ty=T.boolean)
        # Fold the running select into one branch so all predicates contribute.
        val = Select(t, cond=cond, when_true=val, when_false=_kx_payload(x))

    ctx.kx_observe_fp(ctx.fresh(t, init=val))


# ---------------------------------------------------------------------------
# 5) Double division-refinement nudge (CLASS J) -- portable.
# ---------------------------------------------------------------------------

@weave_block("fp_double_div_loop", cost=2)
def _blk_fp_double_div_loop(ctx: GenContext) -> None:
    """q = a / b ; v = q*c + a -- a DOUBLE divide feeding a multiply-add.

    Nudges toward CLASS J: the double ``a/b`` Newton-refinement path differs
    across nvcc versions (RCP refinement / FMA scheduling).  FP division by
    zero is DEFINED (-> +-inf, 0/0 -> nan), so this is UB-free and
    deterministic per binary.  Folded OFF-CHECKSUM via kx_observe_fp.
    """
    t = T.f64
    a = _kx_fp_seed(ctx, t)
    b = _kx_fp_seed(ctx, t)
    c = _kx_fp_seed(ctx, t)
    q = BinOp(t, op=Arith.DIV, lhs=a, rhs=b)
    qc = BinOp(t, op=Arith.MUL, lhs=q, rhs=c)
    val = BinOp(t, op=Arith.ADD, lhs=qc, rhs=a)
    ctx.kx_observe_fp(ctx.fresh(t, init=val))
