"""Data-dependent branches whose ARMS are floating-point computations.

Port of perffuzzer/generator/codegen/gens_fp_branch.py.

This family supplies the if-conversion substrate: an
``if (cond) { FP arm A } else { FP arm B }`` where a single live FP
accumulator is assigned a DIFFERENT floating-point expression on each path,
and is then folded OFF-CHECKSUM via ``ctx.kx_observe_fp``.

Why it is correctness-safe (old == new always holds)
----------------------------------------------------
  * **Off-checksum.**  Every accumulator produced here is folded through
    ``ctx.kx_observe_fp`` into ``output_fp_acc``, which the kx_translator EXCLUDES
    from the old==new comparison.  So even though the two arms compute
    divergent transcendentals, they can NEVER cause a false miscompile.
  * **Deterministic branch condition.**  The basic blocks select on an
    INTEGER condition built from live unsigned pool values
    (``_kx_det_int_cond``), which is bit-identical across compilers.  The
    library-call-guard block may additionally use an FP magnitude compare
    (``fabsf(x) > 1.0f``) -- still safe because the assigned value is
    off-checksum.
  * **UB-free by construction.**  FP arithmetic is fully defined: div-by-zero
    -> +-inf, 0/0 -> nan, transcendentals of any finite (or inf/nan) input ->
    defined.  We never do a C-cast FP->int.  All FP seeds come from
    ``_kx_fp_seed`` (weave_blocks.fp) -- always defined, never uninitialised.

Context discipline
------------------
Each IF arm is woven inside ``_kx_fp_branch_scope`` -- a try/finally context
manager that bumps ``if_depth`` / ``in_nested_control_flow`` (so body
statements record ``ctx.snapshot.uniform == False``) and ALWAYS restores the
emission stack + kx_scope + depth, even on exception.

``requires_uniform=False``: these CREATE divergence; they do not need it.
Backends default to ("cuda", "hip") -- only portable math is used
(tanhf/expf/logf/atanf/sinhf/coshf/sqrtf and the double forms), which both
nvcc and hipcc accept.
"""

from __future__ import annotations

from contextlib import contextmanager

from generator_ir.ir.typedecl import T, GType, TDouble
from generator_ir.ir.operators import Arith, Cmp
from generator_ir.ir.nodes import Expr, Lit, Var, BinOp, RelOp, Call, Fma
from generator_ir.ir.statements import Assign, Block, If
from generator_ir.gen.context import GenContext
from generator_ir.gen.weave_handlers import weave_block

# Reuse the EXISTING FP-island seeding + literal pool so the FP arms depend on
# real runtime data (cannot be constant-folded away) exactly like weave fp.
from generator_ir.gen.weave_blocks.fp import _kx_fp_seed, _FLOAT_LITS


# ---------------------------------------------------------------------------
# Portable transcendental name tables (only single-argument functions both
# nvcc and hipcc accept -- no CUDA-only intrinsics, so backends stay
# ("cuda","hip")).
# ---------------------------------------------------------------------------

_EXP_SAT_F = ["tanhf", "expf", "logf", "sinhf", "coshf", "atanf", "erff"]
_EXP_SAT_D = ["tanh", "exp", "log", "sinh", "cosh", "atan", "erf"]


def _kx_fp_target(ctx: GenContext) -> GType:
    """Pick a float-or-double island target type (double ~ a third of the
    time), mirroring weave_blocks.fp._kx_fp_target."""
    return T.f64 if ctx.rnd_bool(0.35) else T.f32


def _kx_transc_names(t: GType):
    return _EXP_SAT_D if isinstance(t, TDouble) else _EXP_SAT_F


def _rnd_lit(ctx: GenContext, t: GType) -> Lit:
    return Lit(t, value=float(ctx.rnd_choice(_FLOAT_LITS)))


# ---------------------------------------------------------------------------
# Branch-kx_scope context manager (kx_open_block/kx_close_block do NOT touch the depth
# counters -- we manage them, balanced and exception-safe).
# ---------------------------------------------------------------------------

@contextmanager
def _kx_fp_branch_scope(ctx: GenContext):
    """Open a nested emission block as the body of an IF arm.

    Bumps ``if_depth`` + sets ``in_nested_control_flow`` (so statements woven
    in the arm record uniform=False), and restores the emission stack / kx_scope
    / depth / flag on exit -- even on exception -- so the emission stack stays
    balanced.  Yields the arm's statement list.
    """
    ctx.if_depth += 1
    prev_nested = ctx.in_nested_control_flow
    ctx.in_nested_control_flow = True
    ctx.kx_open_block()
    stmts_box = []
    try:
        yield stmts_box
    finally:
        stmts_box.extend(ctx.kx_close_block())
        ctx.in_nested_control_flow = prev_nested
        ctx.if_depth -= 1


# ---------------------------------------------------------------------------
# Deterministic per-thread INTEGER condition (pure read of live unsigned vars
# compared with a random op, so it is bit-identical across compilers and
# per-thread divergent).
# ---------------------------------------------------------------------------

def _kx_det_int_cond(ctx: GenContext) -> RelOp:
    """Build a deterministic, per-thread, UB-free boolean from live ints."""
    uint_t = T.u32
    a = ctx.kx_random_var(uint_t, recency=0.6)
    if ctx.rnd_bool(0.5):
        b = ctx.kx_random_var(uint_t, recency=0.4)
    else:
        b = BinOp(uint_t, op=Arith.BAND,
                  lhs=ctx.kx_random_var(uint_t, recency=0.3),
                  rhs=Lit(uint_t, value=ctx.rnd_choice([1, 3, 7, 15])))
    op = ctx.rnd_choice(list(Cmp))
    return RelOp(T.boolean, op=op, lhs=a, rhs=b)


# ---------------------------------------------------------------------------
# Small FP arm builders -- kept to depth 1-2 so kernels assemble fast.  Each
# returns an FP expression of type *t* over the supplied seed variable(s).
# ---------------------------------------------------------------------------

def _kx_arm_rational(ctx: GenContext, t: GType, x: Expr) -> Expr:
    """A rational-polynomial arm: x*x*c2 + x*c1 + c0, built as an fma chain.

    fma(x, fma(x, c2, c1), c0).  Mirrors the polynomial (rational) branch of
    a piecewise approximation like tanhf's small-|x| path -- the arm the
    compiler keeps cheap and tends to if-convert against the saturating
    branch.
    """
    c2 = _rnd_lit(ctx, t)
    c1 = _rnd_lit(ctx, t)
    c0 = _rnd_lit(ctx, t)
    inner = Fma(t, a=x, b=c2, c=c1)          # x*c2 + c1
    return Fma(t, a=x, b=inner, c=c0)        # x*(x*c2+c1) + c0


def _kx_arm_saturating(ctx: GenContext, t: GType, x: Expr) -> Expr:
    """A transcendental/saturation arm: f(x) * c  (or f(x*s)).

    e.g. tanhf(x)*c / expf(x)*c -- the exp-saturation branch that pairs with
    the rational branch in tanhf/erff-style two-region approximations.
    """
    f = ctx.rnd_choice(_kx_transc_names(t))
    if ctx.rnd_bool(0.5):
        arg: Expr = BinOp(t, op=Arith.MUL, lhs=x, rhs=_rnd_lit(ctx, t))
    else:
        arg = x
    call = Call(t, fn=f, args=[arg], result_ty=t)
    return BinOp(t, op=Arith.MUL, lhs=call, rhs=_rnd_lit(ctx, t))


def _kx_seed_accum(ctx: GenContext, t: GType) -> Var:
    """Declare the live FP accumulator (seeded like an FP island value) that
    both arms will assign and that is kx_observe_fp'd after the if."""
    return _kx_fp_seed(ctx, t)


# ===========================================================================
# 1. fp_branch_twoarm -- the basic data-dependent two-arm if/else.
# ===========================================================================

@weave_block("fp_branch_twoarm", cost=3)
def _blk_fp_branch_twoarm(ctx: GenContext) -> None:
    """if (int_cond) acc = <rational poly>; else acc = <tanhf/expf-sat>;
    kx_observe_fp(acc).

    The if-conversion substrate in its simplest form: a deterministic INTEGER
    condition guards two DIFFERENT floating-point arms assigned to one live
    accumulator -- arm A a rational polynomial (fma chain), arm B a
    transcendental saturation (tanhf/expf * c).  Integer condition =>
    bit-identical branch direction across compilers; the accumulator is
    folded OFF-CHECKSUM (kx_observe_fp) so the divergent transcendentals can
    never be a false miscompile.
    """
    t = _kx_fp_target(ctx)
    # Two seeds so each arm can read a real runtime FP value.
    xa = _kx_fp_seed(ctx, t)
    xb = _kx_fp_seed(ctx, t)
    acc = _kx_seed_accum(ctx, t)

    cond = _kx_det_int_cond(ctx)

    with _kx_fp_branch_scope(ctx) as then_stmts:
        arm_a = _kx_arm_rational(ctx, t, xa)
        ctx.emit(Assign(Var(t, name=acc.name), arm_a))
    then_block = Block(then_stmts)

    with _kx_fp_branch_scope(ctx) as else_stmts:
        arm_b = _kx_arm_saturating(ctx, t, xb)
        ctx.emit(Assign(Var(t, name=acc.name), arm_b))
    else_block = Block(else_stmts)

    ctx.emit(If(cond=cond, then=then_block, otherwise=else_block))
    ctx.kx_observe_fp(Var(t, name=acc.name))


# ===========================================================================
# 2. fp_branch_multiregion -- a 3-way nested if / else-if / else region split.
# ===========================================================================

@weave_block("fp_branch_multiregion", cost=4)
def _blk_fp_branch_multiregion(ctx: GenContext) -> None:
    """if (c0) acc=polyA; else if (c1) acc=polyB; else acc=satC;
    kx_observe_fp(acc).

    A 3-region piecewise split mirroring erfinvf/tgammaf-style multi-region
    polynomial approximations.  The region selectors are deterministic
    INTEGER comparisons (bit-identical across compilers); each region assigns
    a DIFFERENT FP form (two rational polys + a transcendental tail) to one
    live accumulator, folded OFF-CHECKSUM.
    """
    t = _kx_fp_target(ctx)
    xa = _kx_fp_seed(ctx, t)
    xb = _kx_fp_seed(ctx, t)
    xc = _kx_fp_seed(ctx, t)
    acc = _kx_seed_accum(ctx, t)

    c0 = _kx_det_int_cond(ctx)
    c1 = _kx_det_int_cond(ctx)

    with _kx_fp_branch_scope(ctx) as a_stmts:
        ctx.emit(Assign(Var(t, name=acc.name), _kx_arm_rational(ctx, t, xa)))
    a_block = Block(a_stmts)

    # else-if region: a nested IF inside the else arm == `else if`.
    with _kx_fp_branch_scope(ctx) as elif_stmts:
        with _kx_fp_branch_scope(ctx) as b_stmts:
            ctx.emit(Assign(Var(t, name=acc.name),
                            _kx_arm_rational(ctx, t, xb)))
        b_block = Block(b_stmts)
        with _kx_fp_branch_scope(ctx) as c_stmts:
            # transcendental tail region (the "outer" asymptotic arm).
            ctx.emit(Assign(Var(t, name=acc.name),
                            _kx_arm_saturating(ctx, t, xc)))
        c_block = Block(c_stmts)
        ctx.emit(If(cond=c1, then=b_block, otherwise=c_block))
    elif_block = Block(elif_stmts)

    ctx.emit(If(cond=c0, then=a_block, otherwise=elif_block))
    ctx.kx_observe_fp(Var(t, name=acc.name))


# ===========================================================================
# 3. fp_branch_libcall_guard -- the range-guarded library-call shape.
# ===========================================================================

@weave_block("fp_branch_libcall_guard", cost=3)
def _blk_fp_branch_libcall_guard(ctx: GenContext) -> None:
    """if (fabsf(x) > 1.0f) acc = atanf(1.0f/x); else acc = <poly>;
    kx_observe_fp(acc).

    The atanf reciprocal-fold structure: a range-GUARDED library call inside
    one arm (``atanf(1/x)`` for the large-|x| asymptotic region) versus a
    cheap polynomial in the other.  The guard is an FP magnitude compare
    (``fabsf(x) > 1``); that is fine here because the assigned accumulator is
    OFF-CHECKSUM, and half the time we instead use a rock-solid INTEGER guard
    so the family always contributes a bit-identical-direction branch too.
    FP div-by-zero is defined (``1.0f/x`` -> +-inf) so the reciprocal is
    UB-free.
    """
    # float only: fabsf / atanf / the reciprocal-fold idiom is the float
    # shape; keeps the guard portable & simple.
    t = T.f32
    x = _kx_fp_seed(ctx, t)
    acc = _kx_seed_accum(ctx, t)

    if ctx.rnd_bool(0.5):
        # FP magnitude guard: if (fabsf(x) > threshold)
        thr = float(ctx.rnd_choice([1.0, 0.5, 2.0]))
        absx = Call(t, fn="fabsf", args=[x], result_ty=t)
        cond: Expr = RelOp(T.boolean, op=Cmp.GT, lhs=absx,
                           rhs=Lit(t, value=thr))
    else:
        # Rock-solid integer guard.
        cond = _kx_det_int_cond(ctx)

    # THEN arm: guarded library call on the reciprocal -> asymptotic region.
    with _kx_fp_branch_scope(ctx) as then_stmts:
        recip = BinOp(t, op=Arith.DIV, lhs=Lit(t, value=1.0), rhs=x)
        libf = ctx.rnd_choice(["atanf", "logf", "tanhf"])
        call = Call(t, fn=libf, args=[recip], result_ty=t)
        ctx.emit(Assign(Var(t, name=acc.name), call))
    then_block = Block(then_stmts)

    # ELSE arm: cheap polynomial on x (the near-zero region).
    with _kx_fp_branch_scope(ctx) as else_stmts:
        ctx.emit(Assign(Var(t, name=acc.name), _kx_arm_rational(ctx, t, x)))
    else_block = Block(else_stmts)

    ctx.emit(If(cond=cond, then=then_block, otherwise=else_block))
    ctx.kx_observe_fp(Var(t, name=acc.name))
