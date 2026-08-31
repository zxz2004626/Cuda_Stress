"""Narrow-integer division / byte-extraction weaving blocks (port of
perffuzzer gens_narrowint).

These three blocks emit INTEGER idioms the rest of codegen never produces,
each of which is known to make newer nvcc emit slower SASS than older nvcc on
sm_75 (a pure codegen/perf regression -- the integer *values* are
bit-identical across compilers, so the deterministic checksum oracle stays
green and only the timing oracle fires).  They are deliberately built from RAW
``BinOp`` nodes (NOT the ``SafeOp.SDIV/SMOD`` macros): the S* guard-ternary
``(b == 0 ? 1 : b)`` would hide the narrow / constant divisor from the
compiler's value-range analysis and KILL the trigger.  Raw division is UB-free
here only because every divisor is *provably nonzero* and every type is
UNSIGNED (so there is no signed INT_MIN/-1 overflow):

  * class M (``narrow_const_div``): divisor is a hard COMPILE-TIME CONSTANT
    >= 2.
  * class H (``narrow_runtime_div``): divisor is a RUNTIME ``(v & mask) + 1``
    in [1, mask+1] -- structurally >= 1, never zero.
  * class L (``byte_extract``): no division at all; only constant in-range
    shifts [0,31] on a 32-bit uint (well-defined) and value-preserving casts.

Each value is folded into the running checksum via ``ctx.observe`` so it is
not DCE'd.  All three are pure-arith / portable, so the default
("cuda", "hip") backend tuple is correct.
"""

from __future__ import annotations

from generator_ir.ir.typedecl import T
from generator_ir.ir.operators import Arith
from generator_ir.ir.nodes import BinOp, Cast, Lit
from generator_ir.gen.weave_handlers import weave_block


# Hard constants whose reciprocal lowering is the one newer nvcc regresses
# (NOT easy powers-of-two / 3 / 5 the strength-reducer handles cheaply on both
# compilers).
_HARD_CONSTS = [7, 23, 25, 11, 13, 37, 41, 53, 100, 1000]
# Hard byte-range constants for the uchar (& 0xff) variant.
_HARD_BYTE_CONSTS = [7, 23, 25, 11, 13, 37, 41, 53]
# Byte/half lanes for the extraction idiom.
_BYTE_SHIFTS = [0, 8, 16, 24]


# ---------------------------------------------------------------------------
# Class M -- narrow CONSTANT division.
#   (x & 0xffff) / C   (or % C), C a HARD constant.
# Newer nvcc lowers the range-narrowed (<=16-bit) dividend / hard-constant
# divisor to an inexact mul.wide.u16 reciprocal + Newton add-back correction
# where older nvcc used one exact mul.wide.u32.  Both the `& 0xffff` mask AND
# the hard divisor are load-bearing.
# ---------------------------------------------------------------------------

@weave_block("narrow_const_div", inputs=(T.u32,), weight=3.0, cost=2)
def _blk_narrow_const_div(ctx) -> None:
    """v = (x & 0xffff) / C   or   (x & 0xff) / C, with C a HARD constant.

    UB proof: unsigned type (no signed INT_MIN/-1), and C is a assemble-time
    constant >= 7 (always nonzero) -- raw BinOp DIV/MOD is well-defined.
    The mask narrows the dividend so the compiler picks the inexact 16-bit
    reciprocal."""
    uint_t = T.u32
    x = ctx.kx_random_var(uint_t, recency=0.6)
    op = ctx.rnd_choice([Arith.DIV, Arith.MOD])

    if ctx.rnd_bool(0.5):
        # 16-bit-narrowed dividend, hard constant divisor.
        mask = 0xffff
        C = ctx.rnd_choice(_HARD_CONSTS)
    else:
        # 8-bit-narrowed (uchar-range) dividend, hard byte constant divisor.
        mask = 0xff
        C = ctx.rnd_choice(_HARD_BYTE_CONSTS)

    masked = BinOp(uint_t, op=Arith.BAND, lhs=x, rhs=Lit(uint_t, value=mask))
    # RAW division: C is a nonzero assemble-time constant, dividend unsigned.
    val = BinOp(uint_t, op=op, lhs=masked, rhs=Lit(uint_t, value=C))
    ctx.observe(ctx.fresh(uint_t, init=val))


# ---------------------------------------------------------------------------
# Class H -- narrow RUNTIME division -> software subroutine.
#   (x & 0x7fff) % ((y & 7) + 1)   (or DIV).
# When the compiler proves BOTH operands narrow (<=16-bit), newer nvcc emits a
# CALL to the software remainder subroutine instead of the inline
# I2F / MUFU.RCP / F2I sequence older nvcc used.
# ---------------------------------------------------------------------------

@weave_block("narrow_runtime_div", inputs=(T.u32,), weight=3.0, cost=2)
def _blk_narrow_runtime_div(ctx) -> None:
    """v = (x & dmask) % ((y & qmask) + 1)  or DIV -- both operands proven narrow.

    UB proof: unsigned type (no signed overflow), and the divisor is
    ``(y & qmask) + 1`` which lies in [1, qmask+1] -- structurally >= 1, so it
    can NEVER be zero.  Raw BinOp DIV/MOD is therefore well-defined.  The
    masks let the compiler prove both operands <=16-bit -> the software path.
    """
    uint_t = T.u32
    x = ctx.kx_random_var(uint_t, recency=0.6)
    y = ctx.kx_random_var(uint_t, recency=0.4)
    op = ctx.rnd_choice([Arith.MOD, Arith.DIV])

    if ctx.rnd_bool(0.5):
        dmask, qmask = 0x7fff, 0x7   # dividend <=15-bit, divisor in [1, 8]
    else:
        dmask, qmask = 0xff, 0x3     # dividend <=8-bit,  divisor in [1, 4]

    dividend = BinOp(uint_t, op=Arith.BAND, lhs=x, rhs=Lit(uint_t, value=dmask))
    # divisor = (y & qmask) + 1u  -> always in [1, qmask+1], provably nonzero.
    divisor = BinOp(uint_t, op=Arith.ADD,
                    lhs=BinOp(uint_t, op=Arith.BAND,
                              lhs=y, rhs=Lit(uint_t, value=qmask)),
                    rhs=Lit(uint_t, value=1))
    # RAW division: divisor structurally >= 1, dividend unsigned.
    val = BinOp(uint_t, op=op, lhs=dividend, rhs=divisor)
    ctx.observe(ctx.fresh(uint_t, init=val))


# ---------------------------------------------------------------------------
# Class L -- repeated byte extraction.
#   sum of (unsigned)(unsigned char)(x >> k) for k in {0,8,16,24}.
# Newer nvcc changes the PRMT-based byte unpack into SHF + LOP3.
# ---------------------------------------------------------------------------

@weave_block("byte_extract", inputs=(T.u32,), weight=3.0, cost=2)
def _blk_byte_extract(ctx) -> None:
    """v = sum_k (unsigned)(unsigned char)(x >> k), k a permutation of
    {0,8,16,24}.

    UB proof: shifts are CONSTANT amounts in {0,8,16,24} (all < 32) on a
    32-bit unsigned value -> well-defined; casts to unsigned char / unsigned
    are value-preserving conversions.  No division, no indexing -- the only
    risk (shift range) is excluded by construction."""
    uint_t = T.u32
    char_t = T.uchar
    x = ctx.kx_random_var(uint_t, recency=0.7)

    # Random subset (>=2 lanes) in random order -> the repeated byte unpack.
    shifts = list(_BYTE_SHIFTS)
    ctx.rng.shuffle(shifts)
    k_count = ctx.rnd_int(2, len(shifts))
    shifts = shifts[:k_count]

    acc = None
    for k in shifts:
        # (x >> k): RAW right shift, constant amount in [0,31] -> defined.
        shifted = BinOp(uint_t, op=Arith.SHR, lhs=x, rhs=Lit(uint_t, value=k))
        # (unsigned)(unsigned char)(...) -- truncate to a byte, widen back.
        byte = Cast(uint_t, to=uint_t,
                    arg=Cast(char_t, to=char_t, arg=shifted))
        acc = byte if acc is None else BinOp(uint_t, op=Arith.ADD,
                                             lhs=acc, rhs=byte)

    # Optional ushort half-word extraction variant, summed in too.
    if ctx.rnd_bool(0.4):
        ushort_t = T.ushort
        hk = ctx.rnd_choice([0, 16])
        hshift = BinOp(uint_t, op=Arith.SHR, lhs=x, rhs=Lit(uint_t, value=hk))
        half = Cast(uint_t, to=uint_t,
                    arg=Cast(ushort_t, to=ushort_t, arg=hshift))
        acc = BinOp(uint_t, op=Arith.ADD, lhs=acc, rhs=half)

    ctx.observe(ctx.fresh(uint_t, init=acc))
