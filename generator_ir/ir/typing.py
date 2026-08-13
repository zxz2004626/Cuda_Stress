"""Type operations for the gpuperf IR.

C/CUDA type promotion, compatibility, and arithmetic result-type rules
(port of perffuzzer ``ir_utils/type_ops.py``).  All functions accept IR
GType nodes (``TQualified`` wrappers are unwrapped) and return GType
nodes or booleans.

Promotion rank ordering (lowest to highest), following the C "usual
arithmetic conversions" extended with CUDA ``__half``/``__nv_bfloat16``:

  bool < schar < uchar < short < ushort < int < uint
  < long long < ulong long < half < bfloat16 < float < double
"""

from __future__ import annotations

from typing import Any, Optional

from generator_ir.ir.operators import Arith
from generator_ir.ir.typedecl import (
    GType, T, TBool, TSChar, TChar, TShort, TUShort, TInt, TUInt, TLong,
    TULong, THalf, TBF16, TFloat, TDouble, TVoid, TArray, TVec, TPointer,
    TStruct,
)

# ---------------------------------------------------------------------------
# Promotion rank table -- higher rank wins in promotions
# ---------------------------------------------------------------------------

_RANK_MAP = {
    TBool: 0,
    TSChar: 1,
    TChar: 2,
    TShort: 3,
    TUShort: 4,
    TInt: 5,
    TUInt: 6,
    TLong: 7,
    TULong: 8,
    THalf: 9,
    TBF16: 10,
    TFloat: 11,
    TDouble: 12,
}


def _kx_unwrap(t: Any) -> GType:
    """Strip TQualified wrappers to get the underlying type."""
    if isinstance(t, GType):
        return t.unqualified()
    return t


def _grade(t: Any) -> int:
    """Promotion rank for a type (higher wins); -1 for non-scalars."""
    t = _kx_unwrap(t)
    for cls, rank in _RANK_MAP.items():
        if isinstance(t, cls):
            return rank
    return -1


# ---------------------------------------------------------------------------
# Type classification helpers
# ---------------------------------------------------------------------------

_INT_TYPES = (TBool, TInt, TUInt, TShort, TUShort, TChar, TSChar,
              TLong, TULong)
_SIGNED_INT_TYPES = (TInt, TShort, TSChar, TLong)
_UNSIGNED_INT_TYPES = (TBool, TUInt, TUShort, TChar, TULong)
_FLOAT_TYPES = (TFloat, TDouble, THalf, TBF16)


def _kx_is_integer(t: Any) -> bool:
    return isinstance(_kx_unwrap(t), _INT_TYPES)


def _kx_is_signed_integer(t: Any) -> bool:
    return isinstance(_kx_unwrap(t), _SIGNED_INT_TYPES)


def _kx_is_unsigned_integer(t: Any) -> bool:
    return isinstance(_kx_unwrap(t), _UNSIGNED_INT_TYPES)


def _kx_is_floating(t: Any) -> bool:
    return isinstance(_kx_unwrap(t), _FLOAT_TYPES)


def _kx_is_scalar(t: Any) -> bool:
    return _kx_is_integer(t) or _kx_is_floating(t)


def _kx_is_bool(t: Any) -> bool:
    return isinstance(_kx_unwrap(t), TBool)


def _kx_is_void(t: Any) -> bool:
    return isinstance(_kx_unwrap(t), TVoid)


# ---------------------------------------------------------------------------
# Integer promotion (C11 6.3.1.1)
# ---------------------------------------------------------------------------

def kx_integer_promote(t: Any) -> GType:
    """Types narrower than ``int`` promote to ``int``; the rest stay."""
    t = _kx_unwrap(t)
    if isinstance(t, (TBool, TSChar, TChar, TShort, TUShort)):
        return T.i32
    return t


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def kx_can_promote(from_type: Any, to_type: Any) -> bool:
    """True if *from_type* widens to *to_type* without data loss (usual
    arithmetic conversion direction)."""
    fr = _grade(from_type)
    tr = _grade(to_type)
    if fr < 0 or tr < 0:
        return False
    return fr <= tr


def kx_common_type(type_a: Any, type_b: Any) -> GType:
    """Common type of a binary operation, per C's usual arithmetic
    conversions extended for CUDA half/bfloat16."""
    a = _kx_unwrap(type_a)
    b = _kx_unwrap(type_b)

    if type(a) is type(b):
        return a

    # Rule 1: double wins.
    if isinstance(a, TDouble) or isinstance(b, TDouble):
        return T.f64
    # Rule 2: then float.
    if isinstance(a, TFloat) or isinstance(b, TFloat):
        return T.f32
    # Rule 3: mixed bfloat16 promotes to float.
    if isinstance(a, TBF16) or isinstance(b, TBF16):
        return T.f32
    # Rule 4: half (integer other side promotes to float).
    if isinstance(a, THalf) or isinstance(b, THalf):
        other = b if isinstance(a, THalf) else a
        if _kx_is_integer(other):
            return T.f32
        return T.f16

    # Rule 5: both integers -- integer-promote first.
    a = kx_integer_promote(a)
    b = kx_integer_promote(b)

    if type(a) is type(b):
        return a

    ra, rb = _grade(a), _grade(b)

    # Same signedness -> wider type.
    if _kx_is_signed_integer(a) == _kx_is_signed_integer(b):
        return a if ra >= rb else b

    # Mixed signedness.
    signed, unsigned = (a, b) if _kx_is_signed_integer(a) else (b, a)
    rs, ru = _grade(signed), _grade(unsigned)

    if ru >= rs:
        return unsigned
    if rs > ru:
        return signed

    _to_unsigned = {
        TInt: T.u32,
        TLong: T.u64,
        TShort: T.ushort,
        TSChar: T.uchar,
    }
    return _to_unsigned.get(type(signed), unsigned)


def kx_is_compatible(type_a: Any, type_b: Any) -> bool:
    """True when *type_a* can be assigned to a location of *type_b* under
    C/CUDA implicit conversion rules (narrowing allowed, void excluded)."""
    a = _kx_unwrap(type_a)
    b = _kx_unwrap(type_b)

    if type(a) is type(b):
        return True
    if _kx_is_void(a) or _kx_is_void(b):
        return False
    if _kx_is_scalar(a) and _kx_is_scalar(b):
        return True
    if _kx_is_bool(a) and _kx_is_integer(b):
        return True
    if _kx_is_integer(a) and _kx_is_bool(b):
        return True
    if isinstance(a, TPointer) and isinstance(b, TPointer):
        if isinstance(a.base, TVoid) or isinstance(b.base, TVoid):
            return True
        return kx_is_compatible(a.base, b.base)
    return False


def kx_implicit_cast_needed(from_type: Any, to_type: Any) -> bool:
    """True when converting *from_type* to *to_type* needs a cast (types
    differ but are compatible)."""
    a = _kx_unwrap(from_type)
    b = _kx_unwrap(to_type)
    if type(a) is type(b):
        return False
    return kx_is_compatible(a, b)


def kx_arithmetic_result_type(op: Any, left_type: Any, right_type: Any) -> GType:
    """Result type of an arithmetic binary op.  Shifts yield the promoted
    left operand type (C11 6.5.7); everything else is kx_common_type."""
    if isinstance(op, Arith) and op in (Arith.SHL, Arith.SHR):
        return kx_integer_promote(left_type)
    return kx_common_type(left_type, right_type)


def diff_result_type() -> GType:
    """Comparison operators always produce ``bool``."""
    return T.boolean


def kx_logical_result_type() -> GType:
    """Logical operators (&&, ||) always produce ``bool``."""
    return T.boolean


# ---------------------------------------------------------------------------
# Convenience predicates
# ---------------------------------------------------------------------------

def kx_is_integer_type(t: Any) -> bool:
    return _kx_is_integer(t)


def kx_is_signed_integer_type(t: Any) -> bool:
    return _kx_is_signed_integer(t)


def kx_is_unsigned_integer_type(t: Any) -> bool:
    return _kx_is_unsigned_integer(t)


def kx_is_floating_type(t: Any) -> bool:
    return _kx_is_floating(t)


def kx_is_scalar_type(t: Any) -> bool:
    return _kx_is_scalar(t)


def kx_is_bool_type(t: Any) -> bool:
    return _kx_is_bool(t)


def kx_is_void_type(t: Any) -> bool:
    return _kx_is_void(t)


def kx_is_pointer_type(t: Any) -> bool:
    return isinstance(_kx_unwrap(t), TPointer)


def kx_is_array_type(t: Any) -> bool:
    return isinstance(_kx_unwrap(t), TArray)


def kx_is_vector_type(t: Any) -> bool:
    return isinstance(_kx_unwrap(t), TVec)


def kx_is_struct_type(t: Any) -> bool:
    return isinstance(_kx_unwrap(t), TStruct)


def kx_element_type(t: Any) -> Optional[GType]:
    """Element type of an array, vector, or pointer; None otherwise."""
    t = _kx_unwrap(t)
    if isinstance(t, (TArray, TVec)):
        return t.elem
    if isinstance(t, TPointer):
        return t.base
    return None


def kx_type_size_bits(t: Any) -> int:
    """Bit-width for scalar types; 0 for aggregates."""
    t = _kx_unwrap(t)
    if _kx_is_scalar(t):
        return t.width * 8
    return 0
