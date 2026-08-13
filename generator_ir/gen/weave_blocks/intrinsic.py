"""Weave-mode GPU-intrinsic blocks (thin wrappers of ExprGen).

Port of perffuzzer/generator/codegen/gens_intrinsic.py.  These blocks
exercise GPU integer intrinsics by directly calling ExprGen's internal
intrinsic builders:

  * ``ExprGen._intrinsic``          -- the hardcoded portable integer
    intrinsics (__clz/__popc/__ffs/__brev + ll variants), available on BOTH
    CUDA and HIP -> default ("cuda","hip") backends.
  * ``ExprGen._catalog_intrinsic``  -- the data-driven, ARCH-GATED catalog
    (gpuperf/ir/intrinsics.py CATALOG: bit_manip/int_math/longlong_math/dpx).
    It internally filters by the target SM version and returns a terminal on
    a non-CUDA backend, so the catalog blocks are registered CUDA-only and
    rely on ExprGen's own arch checks for sm-version gating.

Both builders return an INTEGER-typed expression (or a terminal of the
requested integer target), so the produced value casts cleanly into
``ctx.observe`` and the integer checksum stays UB-free.

``intrinsic_dp4a`` and ``inline_ptx_int`` are feature-coverage gens:
perffuzzer disables them by default (weight=0 unless --enable-feature-gens),
so they are registered here with weight=0.0.
"""

from __future__ import annotations

from generator_ir.ir.typedecl import T, GType
from generator_ir.ir.operators import Arith
from generator_ir.ir.nodes import Expr, Var, Lit, BinOp, Cast, Call
from generator_ir.ir.statements import RawCode
from generator_ir.gen.expr_builder import ExprGen
from generator_ir.gen.weave_handlers import weave_block
from generator_ir.gen.weave_blocks.memory import _c_text


_INT_TARGETS = [T.u32, T.i32, T.u64, T.ushort]


def _kx_observe_int(ctx, val: Expr, t: GType) -> None:
    """Materialize *val* as an integer var of type *t* and fold into _fv.

    ``ctx.observe`` already casts to unsigned int, but we additionally type
    the fresh var as *t* so the value is concrete-integer in the IR."""
    ctx.observe(ctx.fresh(t, init=val))


# ---------------------------------------------------------------------------
# Portable integer intrinsics (CUDA + HIP): __clz / __popc / __ffs / __brev ...
# ---------------------------------------------------------------------------

@weave_block("intrinsic_portable", inputs=(T.u32,), cost=2)
def _blk_intrinsic_portable(ctx) -> None:
    """v = ExprGen._intrinsic(t) -- chains __clz/__popc/__ffs/__brev (and ll
    variants), data-dependent-latency ops that stress instruction selection.
    Available on both CUDA and HIP."""
    t = ctx.rnd_choice(_INT_TARGETS)
    val = ExprGen(ctx)._intrinsic(t, 0)
    _kx_observe_int(ctx, val, t)


# ---------------------------------------------------------------------------
# Catalog intrinsics (CUDA-only, arch-gated by ExprGen)
# ---------------------------------------------------------------------------

@weave_block("intrinsic_catalog", inputs=(T.u32,), cost=2,
             backends=("cuda",))
def _blk_intrinsic_catalog(ctx) -> None:
    """v = ExprGen._catalog_intrinsic(t) -- a deterministic integer/bit/dpx
    intrinsic from ir/intrinsics.CATALOG, gated to the target SM by ExprGen's
    own arch-version filter.  CUDA-only; on an arch where nothing fits, the
    builder returns a terminal of *t* (still integer / UB-free)."""
    t = ctx.rnd_choice(_INT_TARGETS)
    val = ExprGen(ctx)._catalog_intrinsic(t, 0)
    _kx_observe_int(ctx, val, t)


@weave_block("intrinsic_dp4a", inputs=(T.u32,), weight=0.0, cost=2,
             backends=("cuda",))
def _blk_intrinsic_dp4a(ctx) -> None:
    """Unsigned 4x int8 dot-product-accumulate: __dp4a(uA, uB, uC) (sm_61+).

    Returns ``uC + sum_i uA.byte[i]*uB.byte[i]`` over the four uint8 lanes.
    Explicit unsigned-int casts on all three operands select the unsigned
    overload ``unsigned __dp4a(unsigned,unsigned,unsigned)`` unambiguously --
    avoiding both the overload ambiguity the data-driven catalog hits and the
    signed-`__dp4a` cross-nvcc divergence (the catalog blocks the signed
    form).  Deterministic integer op; available on Turing (sm_75).

    Feature-coverage gen: registered with weight=0.0 (perffuzzer disables it
    unless --enable-feature-gens)."""
    def _u(v: Expr) -> Cast:
        return Cast(T.u32, to=T.u32, arg=v)

    a = _u(ctx.kx_random_var(T.u32, loose=True, recency=0.6))
    b = _u(ctx.kx_random_var(T.u32, loose=True, recency=0.6))
    c = _u(ctx.kx_random_var(T.u32, loose=True, recency=0.6))
    val = Call(T.u32, fn="__dp4a", args=[a, b, c], result_ty=T.u32)
    _kx_observe_int(ctx, val, T.u32)


@weave_block("inline_ptx_int", inputs=(T.u32,), weight=0.0, cost=2,
             backends=("cuda",))
def _blk_inline_ptx_int(ctx) -> None:
    """A deterministic integer op expressed as inline PTX (``asm volatile``),
    folded into _fv.

    The ``volatile`` asm is opaque to the compiler (it cannot fold/DCE it),
    so it forces real instruction emission -- a high-value compiler-codegen
    surface -- yet the computed value is a fixed function of its register
    inputs, identical across nvcc versions and stable across runs.  All ops
    are pure register PTX (no memory, no shift-amount UB).  CUDA/PTX only.

    Feature-coverage gen: registered with weight=0.0 (perffuzzer disables it
    unless --enable-feature-gens)."""
    a = _c_text(ctx.kx_random_var(T.u32, loose=True, recency=0.6))
    b = _c_text(ctx.kx_random_var(T.u32, loose=True, recency=0.6))
    out = ctx.kx_fresh_name("ptx_out")
    mnemonic, operands, nin = ctx.rnd_choice([
        ("add.u32",    "%0, %1, %2", 2),
        ("mul.lo.u32", "%0, %1, %2", 2),
        ("xor.b32",    "%0, %1, %2", 2),
        ("or.b32",     "%0, %1, %2", 2),
        ("and.b32",    "%0, %1, %2", 2),
        ("min.u32",    "%0, %1, %2", 2),
        ("max.u32",    "%0, %1, %2", 2),
        ("popc.b32",   "%0, %1",     1),
        ("brev.b32",   "%0, %1",     1),
    ])
    if nin == 2:
        ins = f'"r"((unsigned int)({a})), "r"((unsigned int)({b}))'
    else:
        ins = f'"r"((unsigned int)({a}))'
    code = (
        "{ // inline PTX integer op (deterministic, opaque to the compiler)\n"
        f"    unsigned int {out};\n"
        f'    asm volatile("{mnemonic} {operands};" : "=r"({out}) : {ins});\n'
        f"    _fv = (((_fv) * 2654435761u) ^ {out});\n"
        "}"
    )
    ctx.emit(RawCode(text=code))


@weave_block("intrinsic_catalog_mixed", inputs=(T.u32,), cost=3,
             backends=("cuda",))
def _blk_intrinsic_catalog_mixed(ctx) -> None:
    """Mix a catalog intrinsic with a pool value: v = catalog(t) ^ a.

    Couples the intrinsic result back into the variable pool (Fuzzilli-style)
    so later generators can consume it.  Stays integer/UB-free (BIT_XOR on
    unsigned)."""
    intr = ExprGen(ctx)._catalog_intrinsic(T.u32, 0)
    intr_u = Cast(T.u32, to=T.u32, arg=intr)
    a = ctx.kx_random_var(T.u32, loose=True, recency=0.6)
    val = BinOp(T.u32, op=Arith.BXOR, lhs=intr_u, rhs=a)
    _kx_observe_int(ctx, val, T.u32)
