"""Arithmetic / bitwise / mixed weaving blocks (port of perffuzzer gens_arith).

Each block reads inputs from the shared variable pool (``ctx.kx_random_var``),
builds a value by wrapping ``ExprGen(ctx).kx_generate(t)`` and/or composing
existing IR ops (``BinOp``/``SafeBin``/``Cast``/``Select``), materializes the
result with ``ctx.fresh`` and folds it into the running checksum with
``ctx.observe``.

Determinism-by-construction:
  * UNSIGNED types use plain ``BinOp`` arith (overflow is defined -- wraps).
  * SIGNED / narrow types route through ``SafeOp.SADD/SSUB/SMUL`` (cast ->
    unsigned width, op, cast back) so there is NO signed-overflow UB.
  * div/mod/shift always go through ``SafeOp.SDIV/SMOD/SSHL/SSHR``.
The full integer type mix is used.  All are pure-arith so the default
("cuda", "hip") backend tuple is correct.
"""

from __future__ import annotations

from generator_ir.ir.typedecl import T, GType, TBool, TPointer
from generator_ir.ir.operators import Arith, Cmp, SafeOp, UnaOp
from generator_ir.ir.nodes import (
    BinOp, RelOp, Cast, SafeBin, Select, UnOp, Var, Lit, Subscript,
)
from generator_ir.gen.expr_builder import ExprGen
from generator_ir.gen.weave_handlers import weave_block


# Full integer type mix.
_UNSIGNED_TYPES = [T.u32, T.ushort, T.uchar, T.u64]
_SIGNED_TYPES = [T.i32, T.short, T.schar, T.i64]
_ALL_INT_TYPES = _UNSIGNED_TYPES + _SIGNED_TYPES + [T.boolean]


def _rnd_type(ctx, pool) -> GType:
    return ctx.rnd_choice(pool)


# ---------------------------------------------------------------------------
# Plain unsigned arithmetic (overflow defined -> no UB)
# ---------------------------------------------------------------------------

@weave_block("arith_unsigned", inputs=(T.u32,), cost=2)
def _blk_arith_unsigned(ctx) -> None:
    """v = a <op> b on an unsigned type (ADD/SUB/MUL/AND/OR/XOR), observed.

    Unsigned overflow is defined, so plain BinOp is UB-free."""
    t = _rnd_type(ctx, _UNSIGNED_TYPES)
    a = ctx.kx_random_var(t, recency=0.6)
    b = ctx.kx_random_var(t, recency=0.3)
    op = ctx.rnd_choice(ctx.kx_safe_arith_ops())
    val = BinOp(t, op=op, lhs=a, rhs=b)
    ctx.observe(ctx.fresh(t, init=val))


# ---------------------------------------------------------------------------
# Signed / narrow arithmetic via S* (unsigned-wrap) -> no signed-overflow UB
# ---------------------------------------------------------------------------

@weave_block("arith_signed_safe", inputs=(T.i32,), cost=2)
def _blk_arith_signed_safe(ctx) -> None:
    """v = SADD/SSUB/SMUL(a, b) on a SIGNED type.  The S* ops compute in
    unsigned width and cast back, so signed overflow never occurs."""
    t = _rnd_type(ctx, _SIGNED_TYPES)
    a = ctx.kx_random_var(t, recency=0.6)
    b = ctx.kx_random_var(t, recency=0.4)
    op = ctx.rnd_choice([SafeOp.SADD, SafeOp.SSUB, SafeOp.SMUL])
    val = SafeBin(t, op=op, lhs=a, rhs=b)
    ctx.observe(ctx.fresh(t, init=val))


# ---------------------------------------------------------------------------
# Bit-mix / xorshift hash (UB-free: SSHR + unsigned mul)
# ---------------------------------------------------------------------------

@weave_block("bitmix_xorshift", inputs=(T.u32,), cost=1)
def _blk_bitmix_xorshift(ctx) -> None:
    """v = (a ^ (a >> k)) * C  -- xorshift-style avalanche hash, observed."""
    uint_t = T.u32
    a = ctx.kx_random_var(uint_t, recency=0.7)
    k = ctx.rnd_choice([5, 11, 13, 16, 17])
    shifted = SafeBin(uint_t, op=SafeOp.SSHR, lhs=a, rhs=Lit(uint_t, value=k))
    xored = BinOp(uint_t, op=Arith.BXOR, lhs=a, rhs=shifted)
    C = ctx.rnd_choice([0x9E3779B9, 0x85EBCA6B, 0xC2B2AE35])
    mixed = BinOp(uint_t, op=Arith.MUL, lhs=xored, rhs=Lit(uint_t, value=C))
    ctx.observe(ctx.fresh(uint_t, init=mixed))


@weave_block("bitmix_64", inputs=(T.u64,), cost=2)
def _blk_bitmix_64(ctx) -> None:
    """64-bit xorshift fold: v = (a ^ (a >> k)) * C at ull width, observed."""
    ull = T.u64
    a = ctx.kx_random_var(ull, recency=0.7)
    k = ctx.rnd_choice([29, 31, 33, 27])
    shifted = SafeBin(ull, op=SafeOp.SSHR, lhs=a, rhs=Lit(ull, value=k))
    xored = BinOp(ull, op=Arith.BXOR, lhs=a, rhs=shifted)
    C = ctx.rnd_choice([0xBF58476D1CE4E5B9, 0x94D049BB133111EB])
    mixed = BinOp(ull, op=Arith.MUL, lhs=xored, rhs=Lit(ull, value=C))
    ctx.observe(ctx.fresh(ull, init=mixed))


# ---------------------------------------------------------------------------
# Safe div / mod
# ---------------------------------------------------------------------------

@weave_block("safe_divmod", inputs=(T.u32,), cost=2)
def _blk_safe_divmod(ctx) -> None:
    """v = SDIV/SMOD(a, b) -- divide-by-zero and INT_MIN/-1 guarded."""
    t = _rnd_type(ctx, _UNSIGNED_TYPES + _SIGNED_TYPES)
    a = ctx.kx_random_var(t, recency=0.6)
    b = ctx.kx_random_var(t, recency=0.4)
    op = ctx.rnd_choice([SafeOp.SDIV, SafeOp.SMOD])
    val = SafeBin(t, op=op, lhs=a, rhs=b)
    ctx.observe(ctx.fresh(t, init=val))


# ---------------------------------------------------------------------------
# Shift chains (SSHL/SSHR -> no out-of-range / negative shift UB)
# ---------------------------------------------------------------------------

@weave_block("shift_chain", inputs=(T.u32,), cost=2)
def _blk_shift_chain(ctx) -> None:
    """v = (a << s) ^ (b >> s) with a data-dependent, masked shift amount.

    SSHL/SSHR clamp the shift to [0, bitwidth) so there is no shift UB."""
    uint_t = T.u32
    a = ctx.kx_random_var(uint_t, recency=0.6)
    b = ctx.kx_random_var(uint_t, recency=0.5)
    s = ctx.kx_random_var(uint_t, recency=0.4)
    ls = SafeBin(uint_t, op=SafeOp.SSHL, lhs=a, rhs=s)
    rs = SafeBin(uint_t, op=SafeOp.SSHR, lhs=b, rhs=s)
    val = BinOp(uint_t, op=Arith.BXOR, lhs=ls, rhs=rs)
    ctx.observe(ctx.fresh(uint_t, init=val))


# ---------------------------------------------------------------------------
# Cast chains (stress sign-extension / truncation / promotion)
# ---------------------------------------------------------------------------

@weave_block("cast_chain", inputs=(T.u32,), cost=2)
def _blk_cast_chain(ctx) -> None:
    """v = (T_out)(T_mid2)(T_mid1) a  -- multi-step width/sign cast chain.

    Pure casts of an existing integer value: always defined."""
    src_t = _rnd_type(ctx, _ALL_INT_TYPES)
    cur = ctx.kx_random_var(src_t, recency=0.6)
    cur_t = src_t
    for _ in range(ctx.rnd_int(2, 3)):
        mid_t = _rnd_type(ctx, _ALL_INT_TYPES)
        cur = Cast(mid_t, to=mid_t, arg=cur)
        cur_t = mid_t
    out_t = _rnd_type(ctx, _ALL_INT_TYPES)
    if type(cur_t) is not type(out_t):
        cur = Cast(out_t, to=out_t, arg=cur)
    ctx.observe(ctx.fresh(out_t, init=cur))


# ---------------------------------------------------------------------------
# Ternary / select
# ---------------------------------------------------------------------------

@weave_block("ternary_select", inputs=(T.u32,), cost=2)
def _blk_ternary_select(ctx) -> None:
    """v = (a <cmp> b) ? x : y  -- branchless select; stresses predication."""
    t = _rnd_type(ctx, _UNSIGNED_TYPES + _SIGNED_TYPES)
    uint_t = T.u32
    a = ctx.kx_random_var(uint_t, recency=0.5)
    b = ctx.kx_random_var(uint_t, recency=0.5)
    op = ctx.rnd_choice(list(Cmp))
    cond = RelOp(T.boolean, op=op, lhs=a, rhs=b)
    x = ctx.kx_random_var(t, recency=0.6)
    y = ctx.kx_random_var(t, recency=0.4)
    val = Select(t, cond=cond, when_true=x, when_false=y)
    ctx.observe(ctx.fresh(t, init=val))


# ---------------------------------------------------------------------------
# Mixed-width fold: wrap the full ExprGen so the whole expr machinery fires
# ---------------------------------------------------------------------------

@weave_block("expr_full", inputs=(T.u32,), cost=3)
def _blk_expr_full(ctx) -> None:
    """v = ExprGen(ctx).kx_generate(t) -- wraps ALL existing expr paths (binary/
    unary/ternary/cast/fma/safe/shift/intrinsic/...) at an integer target so the
    produced value is integer/UB-free and folds cleanly into the checksum."""
    t = _rnd_type(ctx, _UNSIGNED_TYPES + _SIGNED_TYPES)
    val = ExprGen(ctx).kx_generate(t)
    ctx.observe(ctx.fresh(t, init=val))


@weave_block("expr_condition", inputs=(T.u32,), cost=2)
def _blk_expr_condition(ctx) -> None:
    """v = (unsigned)generate_condition() -- a boolean predicate folded in as
    0/1, exercising the comparison/logical expr paths."""
    uint_t = T.u32
    cond = _generate_condition(ExprGen(ctx))
    val = Cast(uint_t, to=uint_t, arg=cond)
    ctx.observe(ctx.fresh(uint_t, init=val))


def _generate_condition(eg: ExprGen, depth: int = 0):
    """Port of perffuzzer ExprGen.generate_condition (no gpuperf equivalent):
    a boolean RelOp over two generated operands."""
    ctx = eg.ctx
    if getattr(ctx.config, "use_diverse_types", False):
        # Integer-comparable types only: an FP comparison's 0/1 result depends
        # on FP operands that can diverge across compilers (FMA contraction,
        # libdevice revisions), so folding it into _fv would force a false
        # checksum mismatch.
        non_bool = [t for t in ctx.kx_computation_types()
                    if t.is_int and not isinstance(t, TBool)]
        if non_bool:
            left_type = ctx.rnd_choice(non_bool)
            right_type = ctx.rnd_choice(non_bool)
            left = eg.kx_generate(left_type, depth + 1)
            right = eg.kx_generate(right_type, depth + 1)
            op = ctx.rnd_choice(list(Cmp))
            return RelOp(T.boolean, op=op, lhs=left, rhs=right)
    left = eg.kx_generate(T.u32, depth + 1)
    right = eg.kx_generate(T.u32, depth + 1)
    op = ctx.rnd_choice(list(Cmp))
    return RelOp(T.boolean, op=op, lhs=left, rhs=right)


# ---------------------------------------------------------------------------
# Runtime input load (data-dependent -> not DCE'd; subsumes s0_load_input)
# ---------------------------------------------------------------------------

@weave_block("load_input", inputs=(T.u32,), cost=2)
def _blk_load_input(ctx) -> None:
    """v = input[SMOD(idx, n)]  -- a runtime-data-dependent load.

    SMOD keeps the index in [0, n) so there is no out-of-bounds access.
    Folding this in makes the kernel's value depend on real device data, which
    blocks DCE.  Backend-portable (plain pointer index)."""
    uint_t = T.u32
    idx = ctx.kx_random_var(uint_t, recency=0.5)
    n_uint = Cast(uint_t, to=uint_t, arg=Var(T.i32, name="n"))
    safe_idx = SafeBin(uint_t, op=SafeOp.SMOD, lhs=idx, rhs=n_uint)
    load = Subscript(uint_t, array=Var(TPointer(uint_t), name="input"),
                     index=safe_idx)
    ctx.observe(ctx.fresh(uint_t, init=load))


# ---------------------------------------------------------------------------
# Unary transforms (NOT / NEG via safe routing)
# ---------------------------------------------------------------------------

@weave_block("unary_xform", inputs=(T.u32,), cost=1)
def _blk_unary_xform(ctx) -> None:
    """v = ~a (bitwise NOT, always defined) or, for unsigned, a wrapping negate
    via SSUB(0, a) -- both UB-free."""
    t = _rnd_type(ctx, _UNSIGNED_TYPES)
    a = ctx.kx_random_var(t, recency=0.7)
    if ctx.rnd_bool(0.5):
        val = UnOp(t, op=UnaOp.BNOT, arg=a)
    else:
        # Wrapping unary minus on unsigned == SSUB(0, a); avoids signed NEG UB.
        val = SafeBin(t, op=SafeOp.SSUB, lhs=Lit(t, value=0), rhs=a)
    ctx.observe(ctx.fresh(t, init=val))
