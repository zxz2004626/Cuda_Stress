"""inline_ptx strategy.

Generates CUDA kernels whose core computation is a *randomly generated inline PTX
block* (asm volatile).  The block is assembled from a small, empirically verified PTX
instruction set (every op was validated against its Python mirror on a real GPU), with:

  * many simultaneously-live 32/64-bit registers (register pressure / spills),
  * long dependency chains plus independent chains (scheduling pressure),
  * predicate registers (setp / and.or.not.pred / selp) live across the block,
  * rare backend-relevant ops (prmt, shf.r, bfe, bfi, lop3, sad, mul.hi, popc, clz, brev),
  * runtime inputs and immediate constants feeding constant banks.

All semantics are unsigned modular; every shift count is masked to a legal range first,
so there is no undefined behaviour.  The CPU oracle simulates the exact same instruction
sequence, so GPU-vs-oracle equality is a true correctness check of the compiled block.

Register naming: nvcc inline-asm substitutes %N operands, and the compiler itself owns
%r/%rd/%p names -- so internal PTX registers use the %%xr/%%xd/%%xp escape (literal %).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..base import GeneratedCase, Strategy
from .helpers import MASK32, MASK64, make_input_words, pack_words, unpack_words

M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF


def _signed(value: int, bits: int) -> int:
    """Interpret the low *bits* of value as a two's-complement integer."""
    value &= (1 << bits) - 1
    sign = 1 << (bits - 1)
    return value - (1 << bits) if value & sign else value


def _shl(value: int, count: int, bits: int) -> int:
    """PTX shl: counts greater than the register width clamp to the width."""
    count = min(count & M32, bits)
    return (value << count) & ((1 << bits) - 1)


def _shr(value: int, count: int, bits: int, signed: bool = False) -> int:
    """PTX shr: clamp the u32 count, then zero-fill or sign-fill."""
    count = min(count & M32, bits)
    mask = (1 << bits) - 1
    source = _signed(value, bits) if signed else value & mask
    return (source >> count) & mask


def _bfe(value: int, pos: int, length: int, bits: int, signed: bool) -> int:
    """PTX bfe semantics, including zero length and fields crossing the MSB."""
    pos &= 0xFF
    length &= 0xFF
    mask = (1 << bits) - 1
    if not signed or length == 0:
        sign_bit = 0
    else:
        sign_at = min(pos + length - 1, bits - 1)
        sign_bit = (value >> sign_at) & 1
    result = 0
    for i in range(bits):
        bit = ((value >> (pos + i)) & 1) if i < length and pos + i < bits else sign_bit
        result |= bit << i
    return result & mask


def _bfi_bits(a: int, b: int, pos: int, length: int, bits: int) -> int:
    """PTX bfi semantics: excess inserted bits are dropped, never wrapped."""
    pos &= 0xFF
    length &= 0xFF
    mask = (1 << bits) - 1
    result = b & mask
    for i in range(length):
        if pos + i >= bits:
            break
        result = (result & ~(1 << (pos + i))) | (((a >> i) & 1) << (pos + i))
    return result & mask


def _bmsk(pos: int, width: int, mode: str) -> int:
    """PTX bmsk.{clamp,wrap}.b32 semantics from ISA 9.3 section 9.7.1.23."""
    pos &= M32
    width &= M32
    if mode == "clamp":
        if pos >= 32 or width == 0:
            return 0
        width = min(width, 32)
        return (((1 << width) - 1) << pos) & M32
    pos &= 31
    width &= 31
    if width == 0:
        return 0
    return (((1 << width) - 1) << pos) & M32


def _bfind(value: int, bits: int, signed: bool, shiftamt: bool) -> int:
    value &= (1 << bits) - 1
    if signed and value & (1 << (bits - 1)):
        value = (~value) & ((1 << bits) - 1)
    if value == 0:
        return M32
    found = value.bit_length() - 1
    return bits - 1 - found if shiftamt else found


def _fns(mask: int, base: int, offset: int) -> int:
    """PTX fns.b32; callers constrain base to the documented 0..31 range."""
    base &= M32
    offset = _signed(offset, 32)
    if offset == 0:
        return base if (mask >> base) & 1 else M32
    pos = base
    count = abs(offset) - 1
    increment = 1 if offset > 0 else -1
    while 0 <= pos < 32:
        if (mask >> pos) & 1:
            if count == 0:
                return pos
            count -= 1
        pos += increment
    return M32


def _shf(a: int, b: int, count: int, direction: str, mode: str) -> int:
    n = min(count & M32, 32) if mode == "clamp" else count & 31
    pair = ((b & M32) << 32) | (a & M32)
    if direction == "r":
        return (pair >> n) & M32
    return ((pair << n) >> 32) & M32


def _narrow24(value: int, signed: bool) -> int:
    value &= 0xFFFFFF
    return value - (1 << 24) if signed and value & 0x800000 else value


def _mul24(a: int, b: int, signed: bool, high: bool) -> int:
    product = _narrow24(a, signed) * _narrow24(b, signed)
    return ((product >> 16) if high else product) & M32


def _trunc_div(a: int, b: int) -> int:
    """Integer division rounded toward zero without converting through float."""
    quotient = abs(a) // abs(b)
    return -quotient if (a < 0) != (b < 0) else quotient


def _szext(value: int, count: int, signed: bool, mode: str) -> int:
    count &= M32
    if mode == "clamp" and count >= 32:
        return value & M32
    count &= 31
    if count == 0:
        return 0
    low = value & ((1 << count) - 1)
    return _signed(low, count) & M32 if signed else low


# ---------------------------------------------------------------------------
# Verified PTX instruction table.
# Each op knows how to emit itself and how to evaluate itself.
# ---------------------------------------------------------------------------
def _imm32(x: int) -> str:
    return f"{x & M32}"

def _imm64(x: int) -> str:
    return f"0x{(x & M64):016x}"


def _op(name: str, dst: str, srcs: List[str], imm: Optional[str],
        emit: Callable[[List[str], int], str],
        mirror: Callable[[List[int], int], int], *,
        min_ptx: Tuple[int, int] = (1, 0), min_sm: int = 10) -> Dict[str, Any]:
    return {"name": name, "dst": dst, "srcs": srcs, "imm": imm,
            "emit": emit, "mirror": mirror, "min_ptx": min_ptx, "min_sm": min_sm}


def _build_op_table() -> List[Dict[str, Any]]:
    O = []
    # ---- b32 binary (reg, reg) -------------------------------------------------
    for name in ("add.u32", "sub.u32", "mul.lo.u32", "and.b32", "or.b32", "xor.b32",
                 "min.u32", "max.u32"):
        O.append(_op(name, "r32", ["r32", "r32"], None,
                     lambda d, s, n=name: f"{n} {d}, {s[0]}, {s[1]};",
                     lambda v, n=name: _bin32(v, n)))
    # Carry-flag variants: register result identical to add/sub, but they set
    # CC, exercising the distinct IADD3.COUT / ISUB3.COUT SASS path.
    for name in ("add.cc.u32", "sub.cc.u32"):
        base = name.split(".")[0] + ".u32"
        O.append(_op(name, "r32", ["r32", "r32"], None,
                     lambda d, s, n=name: f"{n} {d}, {s[0]}, {s[1]};",
                     lambda v, n=base: _bin32(v, n)))
    O.append(_op("mul.hi.u32", "r32", ["r32", "r32"], None,
                 lambda d, s: f"mul.hi.u32 {d}, {s[0]}, {s[1]};",
                 lambda v: ((v[0] * v[1]) >> 32) & M32))
    O.append(_op("mad.lo.u32", "r32", ["r32", "r32", "r32"], None,
                 lambda d, s: f"mad.lo.u32 {d}, {s[0]}, {s[1]}, {s[2]};",
                 lambda v: (v[0] * v[1] + v[2]) & M32))
    for signed in (False, True):
        typ = "s32" if signed else "u32"
        for high in (False, True):
            mode = "hi" if high else "lo"
            name = f"mul24.{mode}.{typ}"
            O.append(_op(name, "r32", ["r32", "r32"], None,
                         lambda d, s, n=name: f"{n} {d}, {s[0]}, {s[1]};",
                         lambda v, sg=signed, hi=high: _mul24(v[0], v[1], sg, hi)))
            name = f"mad24.{mode}.{typ}"
            O.append(_op(name, "r32", ["r32", "r32", "r32"], None,
                         lambda d, s, n=name: f"{n} {d}, {s[0]}, {s[1]}, {s[2]};",
                         lambda v, sg=signed, hi=high: (_mul24(v[0], v[1], sg, hi) + v[2]) & M32))
    O.append(_op("div.u32_i", "r32", ["r32"], "divisor32",
                 lambda d, s, i: f"div.u32 {d}, {s[0]}, {i};",
                 lambda v, i: (v[0] & M32) // i))
    O.append(_op("rem.u32_i", "r32", ["r32"], "divisor32",
                 lambda d, s, i: f"rem.u32 {d}, {s[0]}, {i};",
                 lambda v, i: (v[0] & M32) % i))
    for name in ("min.s32", "max.s32"):
        O.append(_op(name, "r32", ["r32", "r32"], None,
                     lambda d, s, n=name: f"{n} {d}, {s[0]}, {s[1]};",
                     lambda v, n=name: (_signed_minmax(v, 32, n))))
    # ---- b32 unary --------------------------------------------------------------
    O.append(_op("popc.b32", "r32", ["r32"], None,
                 lambda d, s: f"popc.b32 {d}, {s[0]};",
                 lambda v: bin(v[0]).count("1")))
    O.append(_op("clz.b32", "r32", ["r32"], None,
                 lambda d, s: f"clz.b32 {d}, {s[0]};",
                 lambda v: 32 - v[0].bit_length() if v[0] else 32))
    O.append(_op("brev.b32", "r32", ["r32"], None,
                 lambda d, s: f"brev.b32 {d}, {s[0]};",
                 lambda v: int(f"{v[0] & M32:032b}"[::-1], 2)))
    O.append(_op("abs.s32", "r32", ["r32"], None,
                 lambda d, s: f"abs.s32 {d}, {s[0]};",
                 lambda v: abs(_signed(v[0], 32)) & M32))
    O.append(_op("neg.s32", "r32", ["r32"], None,
                 lambda d, s: f"neg.s32 {d}, {s[0]};",
                 lambda v: (-_signed(v[0], 32)) & M32))
    O.append(_op("not.b32", "r32", ["r32"], None,
                 lambda d, s: f"not.b32 {d}, {s[0]};",
                 lambda v: (~v[0]) & M32))
    O.append(_op("cnot.b32", "r32", ["r32"], None,
                 lambda d, s: f"cnot.b32 {d}, {s[0]};",
                 lambda v: 1 if (v[0] & M32) == 0 else 0))
    # ---- b32 shifts: PTX clamps counts to 32; the count operand is always u32. ---
    O.append(_op("shl.b32", "r32", ["r32", "r32"], "shift32",
                 lambda d, s: f"shl.b32 {d}, {s[0]}, {s[1]};",
                 lambda v: _shl(v[0], v[1], 32)))
    O.append(_op("shr.u32", "r32", ["r32", "r32"], "shift32",
                 lambda d, s: f"shr.u32 {d}, {s[0]}, {s[1]};",
                 lambda v: _shr(v[0], v[1], 32)))
    O.append(_op("shr.s32", "r32", ["r32", "r32"], "shift32",
                 lambda d, s: f"shr.s32 {d}, {s[0]}, {s[1]};",
                 lambda v: _shr(v[0], v[1], 32, signed=True)))
    O.append(_op("shl.b32_i", "r32", ["r32"], "shift32_imm",
                 lambda d, s, i: f"shl.b32 {d}, {s[0]}, {i};",
                 lambda v, i: _shl(v[0], i, 32)))
    O.append(_op("shr.u32_i", "r32", ["r32"], "shift32_imm",
                 lambda d, s, i: f"shr.u32 {d}, {s[0]}, {i};",
                 lambda v, i: _shr(v[0], i, 32)))
    O.append(_op("shr.s32_i", "r32", ["r32"], "shift32_imm",
                 lambda d, s, i: f"shr.s32 {d}, {s[0]}, {i};",
                 lambda v, i: _shr(v[0], i, 32, signed=True)))
    # ---- b32 rare / special ------------------------------------------------------
    O.append(_op("bfe.u32", "r32", ["r32"], "bfe",
                 lambda d, s, i: f"bfe.u32 {d}, {s[0]}, {i[0]}, {i[1]};",
                 lambda v, i: _bfe(v[0], i[0], i[1], 32, False)))
    O.append(_op("bfe.s32", "r32", ["r32"], "bfe",
                 lambda d, s, i: f"bfe.s32 {d}, {s[0]}, {i[0]}, {i[1]};",
                 lambda v, i: _bfe(v[0], i[0], i[1], 32, True)))
    O.append(_op("bfi.b32", "r32", ["r32", "r32"], "bfi",
                 lambda d, s, i: f"bfi.b32 {d}, {s[0]}, {s[1]}, {i[0]}, {i[1]};",
                 lambda v, i: _bfi_bits(v[0], v[1], i[0], i[1], 32)))
    for mode in ("clamp", "wrap"):
        name = f"bmsk.{mode}.b32"
        O.append(_op(name, "r32", [], "bmsk",
                     lambda d, s, i, n=name: f"{n} {d}, {i[0]}, {i[1]};",
                     lambda v, i, m=mode: _bmsk(i[0], i[1], m),
                     min_ptx=(7, 6), min_sm=70))
    for signed in (False, True):
        typ = "s32" if signed else "u32"
        for shiftamt in (False, True):
            suffix = ".shiftamt" if shiftamt else ""
            name = f"bfind{suffix}.{typ}"
            O.append(_op(name, "r32", ["r32"], None,
                         lambda d, s, n=name: f"{n} {d}, {s[0]};",
                         lambda v, sg=signed, sh=shiftamt: _bfind(v[0], 32, sg, sh)))
    O.append(_op("fns.b32", "r32", ["r32"], "fns",
                 lambda d, s, i: f"fns.b32 {d}, {s[0]}, {i[0]}, {i[1]};",
                 lambda v, i: _fns(v[0], i[0], i[1]), min_ptx=(6, 0), min_sm=30))
    O.append(_op("prmt.b32", "r32", ["r32", "r32"], "prmt",
                 lambda d, s, i: f"prmt.b32 {d}, {s[0]}, {s[1]}, 0x{i:04x};",
                 lambda v, i: _prmt(v[0], v[1], i)))
    O.append(_op("lop3.b32", "r32", ["r32", "r32", "r32"], "lop3",
                 lambda d, s, i: f"lop3.b32 {d}, {s[0]}, {s[1]}, {s[2]}, 0x{i:02x};",
                 lambda v, i: _lop3(v[0], v[1], v[2], i)))
    O.append(_op("sad.u32", "r32", ["r32", "r32", "r32"], None,
                 lambda d, s: f"sad.u32 {d}, {s[0]}, {s[1]}, {s[2]};",
                 lambda v: (v[2] + (v[0] - v[1] if v[0] >= v[1] else v[1] - v[0])) & M32))
    for direction in ("l", "r"):
        for mode in ("clamp", "wrap"):
            name = f"shf.{direction}.{mode}.b32"
            O.append(_op(name, "r32", ["r32", "r32"], "shf_count",
                         lambda d, s, i, n=name: f"{n} {d}, {s[0]}, {s[1]}, {i};",
                         lambda v, i, dr=direction, md=mode: _shf(v[0], v[1], i, dr, md),
                         min_ptx=(3, 1), min_sm=32))
    O.append(_op("selp.b32", "r32", ["r32", "r32", "pred"], None,
                 lambda d, s: f"selp.b32 {d}, {s[0]}, {s[1]}, {s[2]};",
                 lambda v: v[0] if v[2] else v[1]))
    O.append(_op("slct.b32.s32", "r32", ["r32", "r32", "r32"], None,
                 lambda d, s: f"slct.b32.s32 {d}, {s[0]}, {s[1]}, {s[2]};",
                 lambda v: v[0] if _signed(v[2], 32) >= 0 else v[1]))
    for signed in (False, True):
        typ = "s32" if signed else "u32"
        for cmp in ("eq", "ne", "lt", "le", "gt", "ge"):
            name = f"set.{cmp}.u32.{typ}"
            O.append(_op(name, "r32", ["r32", "r32"], None,
                         lambda d, s, n=name: f"{n} {d}, {s[0]}, {s[1]};",
                         lambda v, c=cmp, sg=signed: M32 if _cmp_typed(v[0], v[1], c, 32, sg) else 0))
    for mode in ("clamp", "wrap"):
        for signed in (False, True):
            typ = "s32" if signed else "u32"
            name = f"szext.{mode}.{typ}"
            O.append(_op(name, "r32", ["r32"], "szext_count",
                         lambda d, s, i, n=name: f"{n} {d}, {s[0]}, {i};",
                         lambda v, i, sg=signed, md=mode: _szext(v[0], i, sg, md),
                         min_ptx=(7, 6), min_sm=70))
    # ---- b64 binary ---------------------------------------------------------------
    for name in ("add.u64", "sub.u64", "mul.lo.u64", "and.b64", "or.b64", "xor.b64",
                 "min.u64", "max.u64"):
        O.append(_op(name, "r64", ["r64", "r64"], None,
                     lambda d, s, n=name: f"{n} {d}, {s[0]}, {s[1]};",
                     lambda v, n=name: _bin64(v, n)))
    O.append(_op("mul.hi.u64", "r64", ["r64", "r64"], None,
                 lambda d, s: f"mul.hi.u64 {d}, {s[0]}, {s[1]};",
                 lambda v: (v[0] * v[1]) >> 64))
    O.append(_op("mad.lo.u64", "r64", ["r64", "r64", "r64"], None,
                 lambda d, s: f"mad.lo.u64 {d}, {s[0]}, {s[1]}, {s[2]};",
                 lambda v: (v[0] * v[1] + v[2]) & M64))
    O.append(_op("div.u64_i", "r64", ["r64"], "divisor64",
                 lambda d, s, i: f"div.u64 {d}, {s[0]}, {i};",
                 lambda v, i: (v[0] & M64) // i))
    O.append(_op("rem.u64_i", "r64", ["r64"], "divisor64",
                 lambda d, s, i: f"rem.u64 {d}, {s[0]}, {i};",
                 lambda v, i: (v[0] & M64) % i))
    O.append(_op("mul.wide.u32", "r64", ["r32", "r32"], None,
                 lambda d, s: f"mul.wide.u32 {d}, {s[0]}, {s[1]};",
                 lambda v: (v[0] & M32) * (v[1] & M32)))
    O.append(_op("mul.wide.s32", "r64", ["r32", "r32"], None,
                 lambda d, s: f"mul.wide.s32 {d}, {s[0]}, {s[1]};",
                 lambda v: (_signed(v[0], 32) * _signed(v[1], 32)) & M64))
    for name in ("min.s64", "max.s64"):
        O.append(_op(name, "r64", ["r64", "r64"], None,
                     lambda d, s, n=name: f"{n} {d}, {s[0]}, {s[1]};",
                     lambda v, n=name: _signed_minmax(v, 64, n)))
    # The PTX count operand is u32 even for a 64-bit destination/source.
    O.append(_op("shl.b64", "r64", ["r64", "r32"], "shift64",
                 lambda d, s: f"shl.b64 {d}, {s[0]}, {s[1]};",
                 lambda v: _shl(v[0], v[1], 64)))
    O.append(_op("shr.u64", "r64", ["r64", "r32"], "shift64",
                 lambda d, s: f"shr.u64 {d}, {s[0]}, {s[1]};",
                 lambda v: _shr(v[0], v[1], 64)))
    O.append(_op("shr.s64", "r64", ["r64", "r32"], "shift64",
                 lambda d, s: f"shr.s64 {d}, {s[0]}, {s[1]};",
                 lambda v: _shr(v[0], v[1], 64, signed=True)))
    O.append(_op("shl.b64_i", "r64", ["r64"], "shift64_imm",
                 lambda d, s, i: f"shl.b64 {d}, {s[0]}, {i};",
                 lambda v, i: _shl(v[0], i, 64)))
    O.append(_op("shr.u64_i", "r64", ["r64"], "shift64_imm",
                 lambda d, s, i: f"shr.u64 {d}, {s[0]}, {i};",
                 lambda v, i: _shr(v[0], i, 64)))
    O.append(_op("shr.s64_i", "r64", ["r64"], "shift64_imm",
                 lambda d, s, i: f"shr.s64 {d}, {s[0]}, {i};",
                 lambda v, i: _shr(v[0], i, 64, signed=True)))
    # ---- b64 unary / cvt ----------------------------------------------------------
    O.append(_op("popc.b64", "r32", ["r64"], None,
                 lambda d, s: f"popc.b64 {d}, {s[0]};",
                 lambda v: bin(v[0]).count("1")))
    O.append(_op("clz.b64", "r32", ["r64"], None,
                 lambda d, s: f"clz.b64 {d}, {s[0]};",
                 lambda v: 64 - v[0].bit_length() if v[0] else 64))
    O.append(_op("brev.b64", "r64", ["r64"], None,
                 lambda d, s: f"brev.b64 {d}, {s[0]};",
                 lambda v: int(f"{v[0] & M64:064b}"[::-1], 2)))
    O.append(_op("abs.s64", "r64", ["r64"], None,
                 lambda d, s: f"abs.s64 {d}, {s[0]};",
                 lambda v: abs(_signed(v[0], 64)) & M64))
    O.append(_op("neg.s64", "r64", ["r64"], None,
                 lambda d, s: f"neg.s64 {d}, {s[0]};",
                 lambda v: (-_signed(v[0], 64)) & M64))
    O.append(_op("not.b64", "r64", ["r64"], None,
                 lambda d, s: f"not.b64 {d}, {s[0]};",
                 lambda v: (~v[0]) & M64))
    O.append(_op("cnot.b64", "r64", ["r64"], None,
                 lambda d, s: f"cnot.b64 {d}, {s[0]};",
                 lambda v: 1 if (v[0] & M64) == 0 else 0))
    O.append(_op("bfe.u64", "r64", ["r64"], "bfe",
                 lambda d, s, i: f"bfe.u64 {d}, {s[0]}, {i[0]}, {i[1]};",
                 lambda v, i: _bfe(v[0], i[0], i[1], 64, False)))
    O.append(_op("bfe.s64", "r64", ["r64"], "bfe",
                 lambda d, s, i: f"bfe.s64 {d}, {s[0]}, {i[0]}, {i[1]};",
                 lambda v, i: _bfe(v[0], i[0], i[1], 64, True)))
    O.append(_op("bfi.b64", "r64", ["r64", "r64"], "bfi",
                 lambda d, s, i: f"bfi.b64 {d}, {s[0]}, {s[1]}, {i[0]}, {i[1]};",
                 lambda v, i: _bfi_bits(v[0], v[1], i[0], i[1], 64)))
    for signed in (False, True):
        typ = "s64" if signed else "u64"
        for shiftamt in (False, True):
            suffix = ".shiftamt" if shiftamt else ""
            name = f"bfind{suffix}.{typ}"
            O.append(_op(name, "r32", ["r64"], None,
                         lambda d, s, n=name: f"{n} {d}, {s[0]};",
                         lambda v, sg=signed, sh=shiftamt: _bfind(v[0], 64, sg, sh)))
        for cmp in ("eq", "ne", "lt", "le", "gt", "ge"):
            name = f"set.{cmp}.u32.{typ}"
            O.append(_op(name, "r32", ["r64", "r64"], None,
                         lambda d, s, n=name: f"{n} {d}, {s[0]}, {s[1]};",
                         lambda v, c=cmp, sg=signed: M32 if _cmp_typed(v[0], v[1], c, 64, sg) else 0))
    O.append(_op("cvt.u64.u32", "r64", ["r32"], None,
                 lambda d, s: f"cvt.u64.u32 {d}, {s[0]};",
                 lambda v: v[0] & M64))
    O.append(_op("cvt.u32.u64", "r32", ["r64"], None,
                 lambda d, s: f"cvt.u32.u64 {d}, {s[0]};",
                 lambda v: v[0] & M32))
    O.append(_op("selp.b64", "r64", ["r64", "r64", "pred"], None,
                 lambda d, s: f"selp.b64 {d}, {s[0]}, {s[1]}, {s[2]};",
                 lambda v: v[0] if v[2] else v[1]))
    O.append(_op("slct.b64.s32", "r64", ["r64", "r64", "r32"], None,
                 lambda d, s: f"slct.b64.s32 {d}, {s[0]}, {s[1]}, {s[2]};",
                 lambda v: v[0] if _signed(v[2], 32) >= 0 else v[1]))
    # ---- predicates ---------------------------------------------------------------
    for bits, kind in ((32, "r32"), (64, "r64")):
        for typ, signed in ((f"u{bits}", False), (f"s{bits}", True)):
            for cmp in ("eq", "ne", "lt", "le", "gt", "ge"):
                name = f"setp.{cmp}.{typ}"
                O.append(_op(name, "pred", [kind, kind], None,
                             lambda d, s, n=name: f"{n} {d}, {s[0]}, {s[1]};",
                             lambda v, c=cmp, b=bits, sg=signed: _cmp_typed(v[0], v[1], c, b, sg)))
    O.append(_op("and.pred", "pred", ["pred", "pred"], None,
                 lambda d, s: f"and.pred {d}, {s[0]}, {s[1]};",
                 lambda v: 1 if (v[0] and v[1]) else 0))
    O.append(_op("or.pred", "pred", ["pred", "pred"], None,
                 lambda d, s: f"or.pred {d}, {s[0]}, {s[1]};",
                 lambda v: 1 if (v[0] or v[1]) else 0))
    O.append(_op("xor.pred", "pred", ["pred", "pred"], None,
                 lambda d, s: f"xor.pred {d}, {s[0]}, {s[1]};",
                 lambda v: 1 if bool(v[0]) != bool(v[1]) else 0))
    O.append(_op("not.pred", "pred", ["pred"], None,
                 lambda d, s: f"not.pred {d}, {s[0]};",
                 lambda v: 0 if v[0] else 1))
    # ---- mov (reg / immediate) ------------------------------------------------------
    O.append(_op("mov.b32", "r32", ["r32"], None,
                 lambda d, s: f"mov.b32 {d}, {s[0]};",
                 lambda v: v[0] & M32))
    O.append(_op("mov.b32_i", "r32", [], "u32",
                 lambda d, s, i: f"mov.b32 {d}, {i & M32};",
                 lambda v, i: i & M32))
    O.append(_op("mov.u64", "r64", ["r64"], None,
                 lambda d, s: f"mov.u64 {d}, {s[0]};",
                 lambda v: v[0] & M64))
    O.append(_op("mov.u64_i", "r64", [], "u64",
                 lambda d, s, i: f"mov.u64 {d}, {_imm64(i)};",
                 lambda v, i: i & M64))

    # Requirements are part of the constraint model, not merely documentation:
    # _gen_program filters variants against max_ptx_isa and target_sm.
    requirements = (
        (("popc.", "clz.", "bfind.", "bfind.shiftamt.", "brev.", "bfe.", "bfi.", "prmt."),
         (2, 0), 20),
        (("shf.",), (3, 1), 32),
        (("lop3.",), (4, 3), 50),
        (("fns.",), (6, 0), 30),
        (("bmsk.",), (7, 6), 70),
    )
    for op in O:
        for prefixes, min_ptx, min_sm in requirements:
            if op["name"].startswith(prefixes):
                op["min_ptx"] = max(op["min_ptx"], min_ptx)
                op["min_sm"] = max(op["min_sm"], min_sm)
    return O


OPS = _build_op_table()
_OPS_BY_DST = {
    "r32": [o for o in OPS if o["dst"] == "r32"],
    "r64": [o for o in OPS if o["dst"] == "r64"],
    "pred": [o for o in OPS if o["dst"] == "pred"],
}


def _bin32(v, name):
    a, b = v[0] & M32, v[1] & M32
    return {"add.u32": a + b, "sub.u32": a - b, "mul.lo.u32": a * b,
            "and.b32": a & b, "or.b32": a | b, "xor.b32": a ^ b,
            "min.u32": min(a, b), "max.u32": max(a, b)}[name] & M32


def _bin64(v, name):
    a, b = v[0] & M64, v[1] & M64
    return {"add.u64": a + b, "sub.u64": a - b, "mul.lo.u64": a * b,
            "and.b64": a & b, "or.b64": a | b, "xor.b64": a ^ b,
            "min.u64": min(a, b), "max.u64": max(a, b)}[name] & M64


def _signed_minmax(v, bits, name):
    a, b = _signed(v[0], bits), _signed(v[1], bits)
    return (min(a, b) if name.startswith("min") else max(a, b)) & ((1 << bits) - 1)


def _prmt(a, b, sel):
    src = [(a >> (8 * k)) & 0xFF for k in range(4)] + [(b >> (8 * k)) & 0xFF for k in range(4)]
    res = 0
    for k in range(4):
        selector = (sel >> (4 * k)) & 0xF
        byte = src[selector & 7]
        if selector & 8:
            byte = 0xFF if byte & 0x80 else 0
        res |= byte << (8 * k)
    return res & M32


def _lop3(a, b, c, table):
    a, b, c = a & M32, b & M32, c & M32
    # PTX truth-table bit order: index = a<<2 | b<<1 | c  (a is the MSB) -- verified on GPU.
    res = 0
    for bit in range(32):
        idx = ((a >> bit) & 1) << 2 | ((b >> bit) & 1) << 1 | ((c >> bit) & 1)
        if (table >> idx) & 1:
            res |= 1 << bit
    return res


def _cmp_typed(a, b, op, bits, signed):
    mask = (1 << bits) - 1
    a = _signed(a, bits) if signed else a & mask
    b = _signed(b, bits) if signed else b & mask
    return 1 if {"lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b,
                 "eq": a == b, "ne": a != b}[op] else 0


# ---------------------------------------------------------------------------
# Program generation (deterministic: seed -> program)
# ---------------------------------------------------------------------------
class _Program:
    def __init__(self, nr32: int, nr64: int, npred: int) -> None:
        self.nr32, self.nr64, self.npred = nr32, nr64, npred
        self.steps: List[Dict[str, Any]] = []   # each: op, dst, srcs, imm, extra
        self._last_def: Dict[str, int] = {}     # reg key -> step index
        self._next_dst_r32 = 0
        self._next_dst_r64 = 0

    # -- register helpers ----------------------------------------------------
    @staticmethod
    def r32(i): return ("r32", i)

    @staticmethod
    def r64(i): return ("r64", i)

    @staticmethod
    def pred(i): return ("pred", i)

    def add(self, op, dst, srcs, imm=None, note=""):
        self.steps.append({"op": op, "dst": dst, "srcs": srcs, "imm": imm, "note": note})
        self._last_def[dst] = len(self.steps) - 1

    def _pick_src(self, kind: str, rng: random.Random, chain_prev: Any) -> Any:
        n = {"r32": self.nr32, "r64": self.nr64, "pred": self.npred}[kind]
        # bias: 40% most-recently-defined register (long chain), 30% oldest, 30% random
        if n == 0:
            raise ValueError(f"no {kind} registers")
        r = rng.random()
        if r < 0.40 and chain_prev is not None and chain_prev[0] == kind:
            return chain_prev
        if r < 0.70 and self._last_def:
            oldest = min(self._last_def, key=self._last_def.get)
            if oldest[0] == kind:
                return oldest
        return (kind, rng.randrange(n))

    def _pick_dst32(self, rng) -> Tuple[str, int]:
        if self._next_dst_r32 < self.nr32:
            d = self.r32(self._next_dst_r32)
            self._next_dst_r32 += 1
            return d
        # overwrite a random reg (keeps pressure high)
        return self.r32(rng.randrange(self.nr32))

    def _pick_dst64(self, rng) -> Tuple[str, int]:
        if self._next_dst_r64 < self.nr64:
            d = self.r64(self._next_dst_r64)
            self._next_dst_r64 += 1
            return d
        return self.r64(rng.randrange(self.nr64))


_RARE_PREFIXES = (
    "bfe.", "bfi.", "bfind.", "bfind.shiftamt.", "bmsk.", "fns.", "prmt.",
    "lop3.", "sad.", "shf.", "shl.", "shr.", "popc.", "clz.", "brev.",
    "mul.hi.", "mul.wide.",
)


def _mnemonic(internal_name: str) -> str:
    """Return the PTX mnemonic from an internal variant name."""
    name = internal_name.split(".", 1)[0]
    return name


def _is_rare(internal_name: str) -> bool:
    return internal_name.startswith(_RARE_PREFIXES)


def _sample_immediate(op: Dict[str, Any], occurrence: int, rng: random.Random) -> Any:
    """Sample ISA-valid immediates with deterministic boundary-value coverage."""
    kind = op["imm"]
    name = op["name"]
    if kind == "u32":
        values = [0, 1, 0x7FFFFFFF, 0x80000000, M32]
        return values[occurrence % len(values)] if occurrence < len(values) else rng.getrandbits(32)
    if kind == "u64":
        values = [0, 1, 0x7FFFFFFFFFFFFFFF, 0x8000000000000000, M64]
        return values[occurrence % len(values)] if occurrence < len(values) else rng.getrandbits(64)
    if kind in ("shift32_imm", "shift64_imm", "shf_count"):
        width = 64 if kind == "shift64_imm" else 32
        values = [0, 1, width - 1, width, width + 1, 2 * width - 1, 2 * width, 255]
        variant_offset = sum((i + 1) * ord(ch) for i, ch in enumerate(name))
        return values[(variant_offset + occurrence) % len(values)]
    if kind in ("divisor32", "divisor64"):
        values = [1, 2, 3, 7, 31, 0x7FFFFFFF]
        return values[occurrence % len(values)]
    if kind == "szext_count":
        values = [0, 1, 31, 32, 33, 63, 255]
        offset = 3 if ".clamp." in name else 4
        return values[(offset + occurrence) % len(values)]
    if kind in ("bfe", "bfi"):
        width = 64 if name.endswith("64") else 32
        if name == "bfi.b32":
            return [(7, 39), (0, 0), (31, 2), (32, 1), (255, 255)][occurrence % 5]
        if name == "bfi.b64":
            return [(13, 71), (0, 0), (63, 2), (64, 1), (255, 255)][occurrence % 5]
        pairs = [
            (0, 0), (0, 1), (0, width), (width - 1, 1),
            (width - 1, width), (width, 1), (width + 1, width + 5),
            (7 if width == 32 else 13, 39 if width == 32 else 71),
            (255, 255),
        ]
        variant_offset = 3 if ".s" in name else 0
        return pairs[(variant_offset + occurrence) % len(pairs)]
    if kind == "bmsk":
        pairs = [(0, 0), (0, 1), (0, 31), (0, 32), (0, 33),
                 (31, 1), (31, 2), (32, 1), (33, 39), (255, 255)]
        offset = 7 if ".clamp." in name else 8
        return pairs[(offset + occurrence) % len(pairs)]
    if kind == "fns":
        pairs = [(0, 0), (31, 0), (0, 1), (31, -1), (15, 2), (15, -2)]
        return pairs[occurrence % len(pairs)]
    if kind == "prmt":
        # Generic selectors are four full nibbles.  Values 8..15 request sign
        # replication, a documented path the previous generator never reached.
        values = [0x89AB, 0xCDEF, 0xFFFF, 0x8888, 0x0123, 0x4567]
        if occurrence < len(values):
            return values[occurrence]
        return sum(rng.randrange(16) << (4 * k) for k in range(4))
    if kind == "lop3":
        values = [0xE2, 0x96, 0x00, 0x01, 0x40, 0x80, 0xFE, 0xFF]
        return values[occurrence % len(values)]
    return None


def _gen_program(seed: int, params: Dict[str, Any]) -> _Program:
    rng = random.Random(seed)
    nr32 = max(4, int(params.get("n_regs32", 10)))
    nr64 = max(3, int(params.get("n_regs64", 6)))
    npred = max(2, int(params.get("n_preds", 4)))
    n_instr = int(params.get("n_instr", 150))
    use_rare = bool(params.get("use_rare_ops", True))
    use_imm = bool(params.get("use_immediates", True))
    coverage_mode = str(params.get("coverage_mode", "balanced"))
    max_ptx_isa = tuple(int(x) for x in str(params.get("max_ptx_isa", "7.3")).split(".", 1))
    target_sm_text = str(params.get("target_sm", "sm_86")).removeprefix("sm_")
    target_sm = int("".join(ch for ch in target_sm_text if ch.isdigit()))
    if coverage_mode not in ("balanced", "random"):
        raise ValueError(f"unsupported inline PTX coverage_mode {coverage_mode!r}")

    prog = _Program(nr32, nr64, npred)

    # ---- setup: define all data registers (input + immediates) ---------------
    for i in range(nr32):
        if i < 4:
            prog.add("mov.b32", prog.r32(i), [("in32", i)])
        else:
            prog.add("mov.b32_i", prog.r32(i), [], rng.getrandbits(32))
    for i in range(nr64):
        if i < 2:
            prog.add("mov.u64", prog.r64(i), [("in64", i)])
        else:
            prog.add("mov.u64_i", prog.r64(i), [], rng.getrandbits(64))
    # initial predicates
    for i in range(npred):
        a = prog.r32(rng.randrange(nr32))
        b = prog.r32(rng.randrange(nr32))
        op = rng.choice(["setp.lt.u32", "setp.gt.u32", "setp.eq.u32", "setp.ne.u32"])
        prog.add(op, prog.pred(i), [a, b])

    # ---- compute phase: every instruction's result is folded into one of two
    # accumulator chains (rd0 / rd1) IMMEDIATELY, so ptxas cannot dead-code it away
    # at any -O level.  Sources are biased toward older registers so values stay live
    # long (register pressure), while the two accumulators give scheduling pressure.
    def include(o: Dict[str, Any]) -> bool:
        if o["min_ptx"] > max_ptx_isa or o["min_sm"] > target_sm:
            return False
        if o["name"] in ("mov.b32", "mov.u64"):
            return False
        if o["name"] in ("mov.b32_i", "mov.u64_i"):
            return use_imm
        return use_rare or not _is_rare(o["name"])

    r32_ops = [o for o in _OPS_BY_DST["r32"] if include(o)]
    r64_ops = [o for o in _OPS_BY_DST["r64"] if include(o)]
    pred_ops = [o for o in _OPS_BY_DST["pred"] if include(o)]
    if use_imm:
        # Immediate moves are already included once above.  Keep the flag explicit
        # because older code accidentally duplicated these variants in the pool.
        assert any(o["name"] == "mov.b32_i" for o in r32_ops)
        assert any(o["name"] == "mov.u64_i" for o in r64_ops)

    coverage_queue: List[Dict[str, Any]] = []
    if coverage_mode == "balanced":
        coverage_queue = r32_ops + r64_ops + pred_ops
        rng.shuffle(coverage_queue)
    immediate_occurrences: Dict[str, int] = {}
    shift_occurrences: Dict[str, int] = {}

    rd0, rd1 = prog.r64(0), prog.r64(1)
    scratch = prog.r64(2)
    fold_r32 = prog.r32(0)
    chain_prev = None
    for k in range(n_instr):
        if coverage_mode == "balanced":
            if not coverage_queue:
                coverage_queue = r32_ops + r64_ops + pred_ops
                rng.shuffle(coverage_queue)
            op = coverage_queue.pop()
            kind = op["dst"]
        else:
            kind = rng.choices(["r32", "r64", "pred"], weights=[0.55, 0.25, 0.20])[0]
            pool = {"r32": r32_ops, "r64": r64_ops, "pred": pred_ops}[kind]
            op = rng.choice(pool)
        if op["imm"] in ("shift32", "shift64"):
            value_kind = op["srcs"][0]
            value = prog._pick_src(value_kind, rng, chain_prev)
            tmp = prog.r32(rng.randrange(nr32))
            occurrence = shift_occurrences.get(op["name"], 0)
            shift_occurrences[op["name"]] = occurrence + 1
            width = 64 if op["imm"] == "shift64" else 32
            boundaries = [0, 1, width - 1, width, width + 1, 255]
            if occurrence < len(boundaries):
                variant_offset = sum((i + 1) * ord(ch) for i, ch in enumerate(op["name"]))
                count = boundaries[(variant_offset + occurrence) % len(boundaries)]
                prog.add("mov.b32_i", tmp, [], count, note="shift-boundary")
            else:
                cnt_src = prog._pick_src("r32", rng, chain_prev)
                prog.add("mov.b32", tmp, [cnt_src], note="shift-runtime-u32")
            srcs = [value, tmp]
        else:
            srcs = []
            for stype in op["srcs"]:
                if stype == "pred":
                    srcs.append(prog.pred(rng.randrange(npred)))
                else:
                    srcs.append(prog._pick_src(stype, rng, chain_prev))
        # destination: round-robin (no random overwrite -> fewer dead writes)
        if kind == "r32":
            dst = prog.r32(k % nr32)
        elif kind == "r64":
            dst = prog.r64(k % nr64)
        else:
            dst = prog.pred(k % npred)
        # immediate
        imm = None
        it = op["imm"]
        if it is not None and it not in ("shift32", "shift64"):
            occurrence = immediate_occurrences.get(op["name"], 0)
            immediate_occurrences[op["name"]] = occurrence + 1
            imm = _sample_immediate(op, occurrence, rng)
        prog.add(op["name"], dst, srcs, imm)
        if dst[0] in ("r32", "r64"):
            chain_prev = dst

        # --- immediate fold into the accumulator (all instructions essential) ----
        if kind == "r32":
            prog.add("cvt.u64.u32", scratch, [dst])
            prog.add("xor.b64", rd0 if k % 2 == 0 else rd1, [rd0 if k % 2 == 0 else rd1, scratch])
        elif kind == "r64":
            prog.add("xor.b64", rd0 if k % 2 == 0 else rd1, [rd0 if k % 2 == 0 else rd1, dst])
        else:  # predicate -> fold via selp
            prog.add("selp.b32", fold_r32, [prog.r32(1 % nr32), prog.r32(2 % nr32), dst])
            prog.add("cvt.u64.u32", scratch, [fold_r32])
            prog.add("xor.b64", rd0 if k % 2 == 0 else rd1, [rd0 if k % 2 == 0 else rd1, scratch])
        # keep the other accumulator mixed in occasionally (independent path)
        if k % 7 == 0:
            prog.add("add.u64", rd0, [rd0, rd1])

    # Final runtime anchors guarantee input sensitivity even when a balanced
    # schedule ends in value-destroying operations such as cnot or a zero-width
    # bit-field.  They are deliberately emitted in PTX rather than C++ so the
    # tested block remains the complete data-flow path.
    prog.add("mov.b32", fold_r32, [("in32", 0)], note="input-anchor32")
    prog.add("cvt.u64.u32", scratch, [fold_r32], note="input-anchor32")
    prog.add("xor.b64", rd0, [rd0, scratch], note="input-anchor32")
    prog.add("mov.u64", scratch, [("in64", 0)], note="input-anchor64")
    prog.add("xor.b64", rd1, [rd1, scratch], note="input-anchor64")

    # final outputs
    prog.add("out", rd0, [])
    prog.add("out", rd1, [])
    return prog


def _const_kind(v: int) -> Tuple[str, int]:
    return ("imm32_const", v)


# ---------------------------------------------------------------------------
# Emit the CUDA kernel from a program.
# ---------------------------------------------------------------------------
def _reg_name(reg) -> str:
    kind, idx = reg
    return {"r32": "%%xr", "r64": "%%xd", "pred": "%%xp"}[kind] + str(idx)


def _src_name(src) -> str:
    kind, idx = src
    if kind == "in32":
        return f"%{2 + idx}"
    if kind == "in64":
        return f"%{6 + idx}"
    if kind == "imm32_const":
        return f"{idx}"
    return _reg_name(src)


def _emit_asm(prog: _Program) -> str:
    lines = [
        f".reg .b32 %%xr<{prog.nr32}>;",
        f".reg .b64 %%xd<{prog.nr64}>;",
        f".reg .pred %%xp<{prog.npred}>;",
    ]
    for step in prog.steps:
        if step["op"] == "out":
            lines.append(f"mov.u64 %{0 if step['dst'][1] == 0 else 1}, {_reg_name(step['dst'])};")
            continue
        op = next(o for o in OPS if o["name"] == step["op"])
        dst = _reg_name(step["dst"])
        srcs = [_src_name(s) for s in step["srcs"]]
        imm = step["imm"]
        if op["imm"] == "bfe" or op["imm"] == "bfi":
            line = op["emit"](dst, srcs, imm)
        else:
            line = op["emit"](dst, srcs) if imm is None else op["emit"](dst, srcs, imm)
        lines.append(line)
    # escape newlines/tabs so the asm string stays a single C string literal
    return "\\n\\t".join(lines)


def _emit_cuda(prog: _Program) -> str:
    asm_body = _emit_asm(prog)
    return f"""\
// strategy=inline_ptx  (generated; do not edit)
extern "C" __global__ void test_kernel(
    const unsigned long long* __restrict__ input,
    unsigned long long* __restrict__ output,
    unsigned int input_words,
    unsigned long long seed)
{{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // runtime operands feeding the inline PTX block
    unsigned long long w0 = input[(tid * 3u) % input_words] ^ seed;
    unsigned long long w1 = input[(tid * 5u + 1u) % input_words] ^ (seed >> 11u);
    unsigned long long w2 = input[(tid * 7u + 2u) % input_words] ^ (seed >> 29u);
    unsigned long long w3 = input[(tid * 11u + 3u) % input_words] ^ (seed >> 37u);
    unsigned int a0 = (unsigned int)(w0 ^ (w0 >> 32u));
    unsigned int a1 = (unsigned int)(w1 ^ (w1 >> 32u));
    unsigned int a2 = (unsigned int)(w2) ^ (unsigned int)(seed >> 47u);
    unsigned int a3 = (unsigned int)(w3);
    unsigned long long b0 = input[(tid * 13u + 4u) % input_words] ^ (seed * 3u);
    unsigned long long b1 = input[(tid * 17u + 5u) % input_words] ^ (seed >> 13u);

    unsigned long long o0 = 0ULL, o1 = 0ULL;
    asm volatile(
        "{asm_body}"
        : "=l"(o0), "=l"(o1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(b0), "l"(b1)
        : "memory");

    output[tid * 2u] = o0;
    output[tid * 2u + 1u] = o1;
}}
"""


# ---------------------------------------------------------------------------
# CPU oracle: simulate the same program.
# ---------------------------------------------------------------------------
def _simulate(prog: _Program, a0, a1, a2, a3, b0, b1) -> Tuple[int, int]:
    r32 = [0] * prog.nr32
    r64 = [0] * prog.nr64
    preds = [0] * prog.npred
    for step in prog.steps:
        op = step["op"]
        if op == "out":
            v = r64[step["dst"][1]]
            if step["dst"][1] == 0:
                o0 = v
            else:
                o1 = v
            continue
        src_vals = []
        for s in step["srcs"]:
            kind, idx = s
            if kind == "in32":
                src_vals.append([a0, a1, a2, a3][idx])
            elif kind == "in64":
                src_vals.append([b0, b1][idx])
            elif kind == "imm32_const":
                src_vals.append(idx)
            elif kind == "r32":
                src_vals.append(r32[idx])
            elif kind == "r64":
                src_vals.append(r64[idx])
            elif kind == "pred":
                src_vals.append(preds[idx])
        o = next(x for x in OPS if x["name"] == op)
        result = o["mirror"](src_vals, step["imm"]) if step["imm"] is not None else o["mirror"](src_vals)
        dk, di = step["dst"]
        if dk == "r32":
            r32[di] = result & M32
        elif dk == "r64":
            r64[di] = result & M64
        else:
            preds[di] = 1 if result else 0
    return o0, o1


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------
class InlinePtx(Strategy):
    name = "inline_ptx"
    target_opcodes = ["prmt", "shf", "bfe", "bfi", "lop3", "sad", "mul.hi",
                      "popc", "clz", "brev", "selp", "setp"]

    def _fmt_params(self, p: Dict[str, Any], target_sm: Optional[str] = None) -> Dict[str, Any]:
        return {
            "n_instr": int(p.get("n_instr", 150)),
            "n_regs32": int(p.get("n_regs32", 10)),
            "n_regs64": int(p.get("n_regs64", 6)),
            "n_preds": int(p.get("n_preds", 4)),
            "use_rare_ops": bool(p.get("use_rare_ops", True)),
            "use_immediates": bool(p.get("use_immediates", True)),
            "coverage_mode": str(p.get("coverage_mode", "balanced")),
            # CUDA 11.3, the oldest configured comparator, supports PTX 7.3.
            # Newer ISA variants remain available when explicitly requested.
            "max_ptx_isa": str(p.get("max_ptx_isa", "7.3")),
            "target_sm": str(target_sm or p.get("target_sm", "sm_86")),
        }

    def _program(self, seed: int, params: Dict[str, Any]) -> _Program:
        return _gen_program(seed, params)

    def generate(self, config_seed: int, test_seed: int, params: Dict[str, Any],
                 gen_version: str, target_sm: str) -> GeneratedCase:
        p = self._fmt_params(params, target_sm)
        rng = self._new_rng(config_seed, test_seed)
        input_words = 256
        words = make_input_words(rng, input_words, 8)
        seed = rng.getrandbits(64)
        prog = self._program(seed, p)
        source = _emit_cuda(prog)
        grid, block = [1], [64]
        owpt = 2
        output_words = grid[0] * block[0] * owpt
        input_bytes = pack_words(words)
        expected = self.cpu_oracle(input_words, seed, p, input_bytes, grid, block, owpt)
        params_out = dict(p)
        params_out["covered_ptx_variants"] = sorted({
            step["op"] for step in prog.steps
            if step["op"] != "out" and not step["op"].startswith("mov.")
        })
        params_out["covered_ptx_mnemonics"] = sorted({
            _mnemonic(name) for name in params_out["covered_ptx_variants"]
        })
        return GeneratedCase(
            test_id=f"inline_ptx__{config_seed:x}__{test_seed:x}",
            config_seed=config_seed, test_seed=test_seed, strategy=self.name, params=params_out,
            generator_version=gen_version, kernel_seed=seed, source=source,
            input_bytes=input_bytes, expected_bytes=expected,
            grid=grid, block=block, input_words=input_words, output_words=output_words,
            output_words_per_thread=owpt, target_sm=target_sm,
            cpu_oracle=self.name, comparison_policy="exact_u64",
            cuda_features=[target_sm, "inline_ptx"],
        )

    def cpu_oracle(self, input_words: int, seed: int, params: Dict[str, Any],
                   input_bytes: bytes, grid: List[int], block: List[int],
                   output_words_per_thread: int) -> bytes:
        p = self._fmt_params(params)
        prog = self._program(seed, p)
        words = unpack_words(input_bytes)
        total = grid[0] * block[0]
        out: List[int] = []
        for tid in range(total):
            w0 = words[(tid * 3) % input_words] ^ seed
            w1 = words[(tid * 5 + 1) % input_words] ^ (seed >> 11)
            w2 = words[(tid * 7 + 2) % input_words] ^ (seed >> 29)
            w3 = words[(tid * 11 + 3) % input_words] ^ (seed >> 37)
            a0 = (w0 ^ (w0 >> 32)) & M32
            a1 = (w1 ^ (w1 >> 32)) & M32
            a2 = (w2 & M32) ^ ((seed >> 47) & M32)
            a3 = w3 & M32
            b0 = words[(tid * 13 + 4) % input_words] ^ (seed * 3)
            b1 = words[(tid * 17 + 5) % input_words] ^ (seed >> 13)
            o0, o1 = _simulate(prog, a0, a1, a2, a3, b0 & M64, b1 & M64)
            out.append(o0 & M64)
            out.append(o1 & M64)
        return pack_words(out)

    def describe_pressure(self, p: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "instructions": p.get("n_instr"),
            "regs32": p.get("n_regs32"), "regs64": p.get("n_regs64"),
            "preds": p.get("n_preds"), "rare_ops": p.get("use_rare_ops"),
        }
