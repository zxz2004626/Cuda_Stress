"""Floating-point "island" weave blocks + the FP-aware WeaveTemplate.

Port of perffuzzer/generator/codegen/gens_fp.py.

These blocks compute *rich* floating-point values (binary arith, FMA,
transcendentals, rounding-mode intrinsics, FP16/BF16 bridges, packed half2)
and fold the result through ``ctx.kx_observe_fp`` -- a SEPARATE off-checksum
accumulator (``output_fp_acc``).  Because the FP result never reaches the
integer ``_fv`` checksum, we may use FP-divergent intrinsics here FREELY:
any cross-compiler floating-point difference lands in ``output_fp`` which the
kx_translator EXCLUDES from the old==new comparison.

Determinism / UB-by-construction:
  * FP arithmetic (incl. div-by-zero -> inf/nan, sqrt of negatives -> nan) is
    fully defined and deterministic within a single binary.
  * The one genuine FP UB is an out-of-range ``(int)float`` conversion; we
    NEVER use a C cast for FP->int.  Where a block narrows FP to an integer it
    goes through ``__float2int_rz`` / ``__double2ll_rz`` (saturating, defined)
    and then reinterprets back to FP so the value still folds via kx_observe_fp.

The integer variable pool (``ctx.kx_random_var``) holds NO floating-point vars,
so each block SEEDS its own FP inputs with ``_kx_fp_seed`` -- a fresh FP var
initialised either from a small literal or by reinterpreting a live pool
integer through ``__uint2float_rn`` / ``__ull2double_rn`` (data-dependent,
blocks DCE).

Backends: the portable blocks (binary/fma/transcendental/divmix/fp2int)
target both cuda + hip.  The CUDA-only intrinsic families (explicit rounding
modes ``__fadd_rn`` etc., packed ``half2``, and the FP16/BF16 bridge built
from ``__float2half_rn``) are gated ``backends=("cuda",)``.
"""

from __future__ import annotations

from generator_ir.ir.typedecl import T, GType, TFloat, TDouble, TQualified
from generator_ir.ir.operators import Arith
from generator_ir.ir.nodes import Expr, Lit, Var, BinOp, Call, Fma
from generator_ir.gen.context import GenContext
from generator_ir.gen.expr_builder import ExprGen
from generator_ir.gen.weave_handlers import weave_block, REGISTRY
from generator_ir.gen.weave_templates import WeaveTemplate, weave_template


# ---------------------------------------------------------------------------
# FP seeding helper -- the pool has no FP vars, so make our own.
# ---------------------------------------------------------------------------

_FLOAT_LITS = [0.5, 1.0, -1.0, 1.5, 2.0, 3.25, 0.125, 7.0, -0.75, 42.0]


def _kx_fp_seed(ctx: GenContext, t: GType) -> Var:
    """Declare and return a fresh FP variable of type *t* with a defined init.

    Half the time the seed is reinterpreted from a live integer pool variable
    (via ``__uint2float_rn`` / ``__ull2double_rn``) so the FP island depends on
    real runtime data and cannot be constant-folded away; otherwise it is a
    small literal.  Both inits are defined for any input.
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


def _kx_fp_target(ctx: GenContext) -> GType:
    """Pick a float-or-double island target type."""
    return T.f64 if ctx.rnd_bool(0.35) else T.f32


def _kx_as_uint_island(value: Expr) -> Expr:
    """Reinterpret an FP/uint value as a float for kx_observe_fp.

    Returns the value unchanged when it is already FP -- kx_observe_fp itself
    handles the FP->bits reinterpret -- but for uint-typed intrinsic results
    we wrap in ``__uint_as_float`` so the folded value stays a genuine FP bit
    pattern.
    """
    vt = value.ty
    base = vt.unqualified() if isinstance(vt, TQualified) else vt
    if isinstance(base, (TFloat, TDouble)):
        return value
    # uint-typed intrinsic result (rounding_mode / half2 already __*_as_uint):
    # reinterpret to float so kx_observe_fp folds an FP bit pattern.
    return Call(T.f32, fn="__uint_as_float", args=[value], result_ty=T.f32)


# ---------------------------------------------------------------------------
# Portable FP binary / mixed expression (wraps the full ExprGen FP path)
# ---------------------------------------------------------------------------

@weave_block("fp_binary", cost=2)
def _blk_fp_binary(ctx: GenContext) -> None:
    """v = ExprGen(ctx).kx_generate(float/double) -- wraps ALL existing FP expr
    paths (binary / unary / ternary / fma / casts) at a float target, folded
    off-checksum via kx_observe_fp."""
    t = _kx_fp_target(ctx)
    _kx_fp_seed(ctx, t)
    _kx_fp_seed(ctx, t)
    val = ExprGen(ctx).kx_generate(t)
    ctx.kx_observe_fp(ctx.fresh(t, init=val))


# ---------------------------------------------------------------------------
# FMA chain (fmaf / fma) -- portable
# ---------------------------------------------------------------------------

@weave_block("fp_fma", cost=2)
def _blk_fp_fma(ctx: GenContext) -> None:
    """v = fma(a, b, fma(c, d, e)) -- a nested fused-multiply-add chain built
    via the existing ExprGen FMA builder, folded off-checksum."""
    t = _kx_fp_target(ctx)
    _kx_fp_seed(ctx, t)
    _kx_fp_seed(ctx, t)
    eg = ExprGen(ctx)
    inner = eg._fma(t, 1)
    a = _kx_fp_seed(ctx, t)
    b = _kx_fp_seed(ctx, t)
    node = Fma(t, a=a, b=b, c=inner)
    ctx.kx_observe_fp(ctx.fresh(t, init=node))


# ---------------------------------------------------------------------------
# Transcendentals -- portable (sinf/cosf/expf/sqrtf/logf/tanhf ...)
# ---------------------------------------------------------------------------

_TRANSC_F = ["sinf", "cosf", "expf", "sqrtf", "logf", "tanhf", "rsqrtf",
             "__expf", "__logf", "__sinf", "__cosf"]
_TRANSC_D = ["sin", "cos", "exp", "sqrt", "log", "tanh"]


@weave_block("fp_transcendental", cost=2)
def _blk_fp_transcendental(ctx: GenContext) -> None:
    """v = f(g(x) + c) -- a chain of transcendental math intrinsics.

    These are FP-divergent across compiler versions (different polynomial
    approximations / fast-math choices) which is exactly what we WANT to
    exercise; the result is off-checksum so no false miscompile.  Inputs are
    bounded/defined (transcendentals of any finite value, incl. inf/nan, are
    defined and deterministic per binary)."""
    is_double = ctx.rnd_bool(0.35)
    t = T.f64 if is_double else T.f32
    fns = _TRANSC_D if is_double else _TRANSC_F
    x = _kx_fp_seed(ctx, t)
    g = ctx.rnd_choice(fns)
    inner = Call(t, fn=g, args=[x], result_ty=t)
    bias = Lit(t, value=float(ctx.rnd_choice(_FLOAT_LITS)))
    summed = BinOp(t, op=Arith.ADD, lhs=inner, rhs=bias)
    f = ctx.rnd_choice(fns)
    outer = Call(t, fn=f, args=[summed], result_ty=t)
    ctx.kx_observe_fp(ctx.fresh(t, init=outer))


# ---------------------------------------------------------------------------
# Division / reciprocal mix -- portable (FP div-by-zero is defined -> inf)
# ---------------------------------------------------------------------------

@weave_block("fp_divmix", cost=2)
def _blk_fp_divmix(ctx: GenContext) -> None:
    """v = (a / b) + (c * d) -- FP divide + multiply mix.

    FP division (including by zero -> +-inf, 0/0 -> nan) is fully defined, so
    this is UB-free and deterministic.  The divide path differs across nvcc
    versions (MUFU.RCP refinement) -- desirable, and off-checksum."""
    t = _kx_fp_target(ctx)
    a = _kx_fp_seed(ctx, t)
    b = _kx_fp_seed(ctx, t)
    c = _kx_fp_seed(ctx, t)
    d = _kx_fp_seed(ctx, t)
    quot = BinOp(t, op=Arith.DIV, lhs=a, rhs=b)
    prod = BinOp(t, op=Arith.MUL, lhs=c, rhs=d)
    val = BinOp(t, op=Arith.ADD, lhs=quot, rhs=prod)
    ctx.kx_observe_fp(ctx.fresh(t, init=val))


# ---------------------------------------------------------------------------
# FP -> int -> FP round-trip -- portable, demonstrates SAFE rz conversion
# ---------------------------------------------------------------------------

@weave_block("fp_to_int_roundtrip", cost=2)
def _blk_fp_to_int_roundtrip(ctx: GenContext) -> None:
    """i = __float2int_rz(x*scale) ; v = __int2float_rn(i) + frac.

    Demonstrates the UB-FREE FP->int conversion rule: out-of-range (int)float
    is UB, so we use the saturating round-toward-zero intrinsic
    (__float2int_rz / __double2ll_rz) and reinterpret back to FP, folding the
    FP value off-checksum."""
    is_double = ctx.rnd_bool(0.35)
    t = T.f64 if is_double else T.f32
    x = _kx_fp_seed(ctx, t)
    scale = Lit(t, value=float(ctx.rnd_choice([4.0, 16.0, 64.0, 1.5])))
    scaled = BinOp(t, op=Arith.MUL, lhs=x, rhs=scale)
    if is_double:
        as_int = Call(T.u64, fn="__double2ll_rz", args=[scaled],
                      result_ty=T.u64)
        back = Call(t, fn="__ll2double_rn", args=[as_int], result_ty=t)
    else:
        as_int = Call(T.u32, fn="__float2int_rz", args=[scaled],
                      result_ty=T.u32)
        back = Call(t, fn="__int2float_rn", args=[as_int], result_ty=t)
    frac = BinOp(t, op=Arith.SUB, lhs=x, rhs=back)
    val = BinOp(t, op=Arith.ADD, lhs=back, rhs=frac)
    ctx.kx_observe_fp(ctx.fresh(t, init=val))


# ---------------------------------------------------------------------------
# CUDA-only: explicit rounding-mode arithmetic (__fadd_rn / __fmul_rd / ...)
# ---------------------------------------------------------------------------

@weave_block("fp_rounding_mode", cost=2, backends=("cuda",))
def _blk_fp_rounding_mode(ctx: GenContext) -> None:
    """Wrap the existing rounding-mode intrinsic builder (__fadd_rn/rd/ru/rz,
    __fmul_*, __fmaf_*, __fdiv_*, __fsqrt_*).  Each rounding mode maps to a
    distinct PTX variant -- prime cross-version FP divergence, off-checksum."""
    _kx_fp_seed(ctx, T.f32)
    _kx_fp_seed(ctx, T.f32)
    # returns uint (__float_as_uint), like perffuzzer's _gen_rounding_mode_arith
    val = ExprGen(ctx)._rounding_mode(T.u32, 1)
    ctx.kx_observe_fp(_kx_as_uint_island(val))


# ---------------------------------------------------------------------------
# CUDA-only: FP16 / BF16 bridge chains
# ---------------------------------------------------------------------------

@weave_block("fp_bridge", cost=3, backends=("cuda",))
def _blk_fp_bridge(ctx: GenContext) -> None:
    """Wrap the existing FP16/BF16 conversion+math bridge (__float2half_rn /
    __half2float / __hfma / __float2bfloat16_rn / ...).  Half-precision
    rounding differs across toolkits -- exercised off-checksum."""
    _kx_fp_seed(ctx, T.f32)
    val = ExprGen(ctx)._cuda_fp_bridge(T.f32, 1)
    ctx.kx_observe_fp(_kx_as_uint_island(val))


# ---------------------------------------------------------------------------
# CUDA-only: packed half2 arithmetic
# ---------------------------------------------------------------------------

@weave_block("fp_half2", cost=3, backends=("cuda",))
def _blk_fp_half2(ctx: GenContext) -> None:
    """Wrap the existing packed-half2 builder (__halves2half2 / __hadd2 /
    __hmul2 / __low2half).  Stresses the packed half ALU; folded off-checksum."""
    _kx_fp_seed(ctx, T.f32)
    # returns uint (__float_as_uint), like perffuzzer's _gen_half2_packed
    val = ExprGen(ctx)._half2_packed(T.u32, 1)
    ctx.kx_observe_fp(_kx_as_uint_island(val))


# ===========================================================================
# FP-aware WeaveTemplate -- exercises the output_fp kx_translator plumbing.
# ===========================================================================

@weave_template
class FpIslandTemplate(WeaveTemplate):
    """Like pure_composable, but FP-first: forces at least one FP island block
    so ``kx_observe_fp`` definitely fires, then weaves the rest of the budget.

    Port of perffuzzer's FpIslandTemplate (gens_fp.py).  The whole-kernel
    plumbing perffuzzer builds by hand -- the extra ``output_fp`` float* param
    and the guarded ``output_fp[tid] = __uint_as_float(output_fp_acc)`` write
    -- is already owned by gpuperf's weaver epilogue (generate_weave_kernel
    adds both whenever ``ctx.used_fp()``), so this template only supplies the
    compute phase:

      * sets ``ctx.use_float_types`` (perffuzzer kernel_gen does this when the
        fp_island template is chosen; the gpuperf weaver does not kx_pick weave
        templates yet, so the template sets it itself);
      * force-runs the ``fp_binary`` block (perffuzzer calls
        ``_gen_fp_binary.run(ctx)``) so the FP accumulator is guaranteed
        declared even if swarm-weighted weaving never picks an fp_* block;
      * weaves the complexity budget, during which the fp_* blocks above can
        fire and fold into ``output_fp_acc``.
    """

    name = "fp_island"

    def render_body(self, ctx: GenContext) -> None:
        ctx.use_float_types = True
        blk = REGISTRY.get("fp_binary")
        if blk is not None:
            ctx.fired_blocks[blk.name] += 1
            blk.run(ctx)
            ctx.features_used.add("forced_template_fragment:fp_binary")
        ctx.weave(ctx.kx_budget_for_complexity())
