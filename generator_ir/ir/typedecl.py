"""Type system for gpuperf GPU kernels.

A small structural type universe covering the CUDA/HIP C subset that the
generator emits: scalar numerics, fixed-size vectors, arrays, structs,
pointers, and a qualifier/memory-space wrapper.

Every type exposes:
  * ``width``          -- size in bytes
  * ``is_int`` / ``is_fp`` / ``kx_is_signed``
  * ``kx_min_arch``       -- lowest SM/GFX version providing the type
  * ``spelling()``     -- canonical C spelling used by the emitters
"""

from __future__ import annotations

from enum import Enum, auto
from typing import List, Optional, Sequence, Tuple


class MemSpace(Enum):
    """GPU address spaces."""
    GENERIC = auto()      # unqualified
    GLOBAL = auto()       # device memory
    SHARED = auto()       # block-kx_scope shared memory
    LOCAL = auto()        # thread-private spill
    REGISTER = auto()     # register hint
    CONSTANT = auto()     # read-only constant memory


class Qual(Enum):
    """Source-level qualifiers understood by the emitters."""
    CONST = auto()
    VOLATILE = auto()
    RESTRICT = auto()
    DEVICE = auto()          # __device__
    KERNEL = auto()          # __global__
    SHARED_MEM = auto()      # __shared__
    CONSTANT_MEM = auto()    # __constant__
    MANAGED = auto()         # __managed__
    GRID_CONSTANT = auto()   # __grid_constant__


class Uniformity(Enum):
    """Whether a value may differ between lanes of a warp."""
    UNIFORM = auto()
    VARYING = auto()
    UNKNOWN = auto()


class GType:
    """Base class for all IR types."""

    @property
    def width(self) -> int:
        raise NotImplementedError(type(self).__name__)

    @property
    def is_int(self) -> bool:
        return False

    @property
    def is_fp(self) -> bool:
        return False

    @property
    def kx_is_signed(self) -> bool:
        return False

    @property
    def kx_min_arch(self) -> int:
        return 0

    def unqualified(self) -> "GType":
        return self

    def spelling(self) -> str:
        raise NotImplementedError(type(self).__name__)

    def __str__(self) -> str:
        return self.spelling()

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# ---------------------------------------------------------------------------
# Scalar types
# ---------------------------------------------------------------------------

class _Scalar(GType):
    """Shared implementation for scalar numeric types."""

    _name = "?"
    _width = 0
    _int = False
    _fp = False
    _signed = False
    _min_arch = 0
    _limits: Tuple = (0, 0)

    @property
    def width(self) -> int:
        return self._width

    @property
    def is_int(self) -> bool:
        return self._int

    @property
    def is_fp(self) -> bool:
        return self._fp

    @property
    def kx_is_signed(self) -> bool:
        return self._signed

    @property
    def kx_min_arch(self) -> int:
        return self._min_arch

    @property
    def limits(self) -> Tuple:
        return self._limits

    def spelling(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class TVoid(_Scalar):
    _name = "void"
    _width = 0


class TBool(_Scalar):
    _name = "bool"
    _width = 1
    _int = True
    _limits = (0, 1)


class TChar(_Scalar):
    """unsigned char."""
    _name = "unsigned char"
    _width = 1
    _int = True
    _limits = (0, 255)


class TSChar(_Scalar):
    _name = "signed char"
    _width = 1
    _int = True
    _signed = True
    _limits = (-128, 127)


class TShort(_Scalar):
    _name = "short"
    _width = 2
    _int = True
    _signed = True
    _limits = (-32768, 32767)


class TUShort(_Scalar):
    _name = "unsigned short"
    _width = 2
    _int = True
    _limits = (0, 65535)


class TInt(_Scalar):
    _name = "int"
    _width = 4
    _int = True
    _signed = True
    _limits = (-2147483648, 2147483647)


class TUInt(_Scalar):
    _name = "unsigned int"
    _width = 4
    _int = True
    _limits = (0, 4294967295)


class TLong(_Scalar):
    """long long (i64)."""
    _name = "long long"
    _width = 8
    _int = True
    _signed = True
    _limits = (-9223372036854775808, 9223372036854775807)


class TULong(_Scalar):
    """unsigned long long (u64)."""
    _name = "unsigned long long"
    _width = 8
    _int = True
    _limits = (0, 18446744073709551615)


class THalf(_Scalar):
    """IEEE FP16 (CUDA __half); sm_53+."""
    _name = "__half"
    _width = 2
    _fp = True
    _signed = True
    _min_arch = 53
    _limits = (-65504.0, 65504.0)


class TBF16(_Scalar):
    """bfloat16 (CUDA __nv_bfloat16); sm_80+."""
    _name = "__nv_bfloat16"
    _width = 2
    _fp = True
    _signed = True
    _min_arch = 80
    _limits = (-3.39e38, 3.39e38)


class TFloat(_Scalar):
    _name = "float"
    _width = 4
    _fp = True
    _signed = True


class TDouble(_Scalar):
    _name = "double"
    _width = 8
    _fp = True
    _signed = True


# ---------------------------------------------------------------------------
# Composite types
# ---------------------------------------------------------------------------

_VEC_BASE_NAMES = {
    "int": "int",
    "unsigned int": "uint",
    "float": "float",
    "double": "double",
    "short": "short",
    "unsigned short": "ushort",
    "long long": "longlong",
    "unsigned long long": "ulonglong",
    "__half": "half",
    "unsigned char": "uchar",
    "signed char": "char",
}


class TVec(GType):
    """CUDA-style vector: int2, float4, ..."""

    def __init__(self, elem: GType, lanes: int) -> None:
        if lanes not in (1, 2, 3, 4):
            raise ValueError(f"vector lane count must be 1..4, got {lanes}")
        self.elem = elem
        self.lanes = lanes

    @property
    def width(self) -> int:
        return self.elem.width * self.lanes

    @property
    def is_int(self) -> bool:
        return self.elem.is_int

    @property
    def is_fp(self) -> bool:
        return self.elem.is_fp

    @property
    def kx_min_arch(self) -> int:
        return self.elem.kx_min_arch

    def spelling(self) -> str:
        base = _VEC_BASE_NAMES.get(self.elem.spelling(), self.elem.spelling())
        return f"{base}{self.lanes}"

    def __repr__(self) -> str:
        return f"TVec({self.elem!r}, {self.lanes})"


class TArray(GType):
    """Fixed-size array (possibly nested for multi-dim)."""

    def __init__(self, elem: GType, count: int) -> None:
        if count <= 0:
            raise ValueError(f"array size must be positive, got {count}")
        self.elem = elem
        self.count = count

    @property
    def width(self) -> int:
        return self.elem.width * self.count

    @property
    def kx_min_arch(self) -> int:
        return self.elem.kx_min_arch

    def dims(self) -> List[int]:
        """Outermost-first dimension sizes (unwraps nested arrays)."""
        out: List[int] = []
        t: GType = self
        while isinstance(t, TArray):
            out.append(t.count)
            t = t.elem
        return out

    def scalar_elem(self) -> GType:
        t: GType = self
        while isinstance(t, TArray):
            t = t.elem
        return t

    def spelling(self) -> str:
        return f"{self.scalar_elem().spelling()}[{']['.join(str(d) for d in self.dims())}]"

    def __repr__(self) -> str:
        return f"TArray({self.elem!r}, {self.count})"


class TStruct(GType):
    """Named struct with ordered typed fields."""

    def __init__(self, tag: str, fields: Sequence[Tuple[str, GType]]) -> None:
        self.tag = tag
        self.fields = list(fields)

    @property
    def width(self) -> int:
        return sum(ft.width for _, ft in self.fields)  # ignores padding

    def kx_field_type(self, name: str) -> Optional[GType]:
        for fn, ft in self.fields:
            if fn == name:
                return ft
        return None

    def spelling(self) -> str:
        return self.tag

    def __repr__(self) -> str:
        return f"TStruct({self.tag!r}, {self.fields!r})"


class TPointer(GType):
    """Pointer to a base type in a memory space."""

    def __init__(self, base: GType, space: MemSpace = MemSpace.GENERIC) -> None:
        self.base = base
        self.space = space

    @property
    def width(self) -> int:
        return 8

    @property
    def kx_min_arch(self) -> int:
        return self.base.kx_min_arch

    def spelling(self) -> str:
        return f"{self.base.spelling()}*"

    def __repr__(self) -> str:
        return f"TPointer({self.base!r}, {self.space!r})"


class TQualified(GType):
    """Wraps a base type with qualifiers and an explicit memory space."""

    def __init__(self, base: GType,
                 quals: Optional[Sequence[Qual]] = None,
                 space: MemSpace = MemSpace.GENERIC) -> None:
        self.base = base
        self.quals: List[Qual] = list(quals or [])
        self.space = space

    @property
    def width(self) -> int:
        return self.base.width

    @property
    def is_int(self) -> bool:
        return self.base.is_int

    @property
    def is_fp(self) -> bool:
        return self.base.is_fp

    @property
    def kx_is_signed(self) -> bool:
        return self.base.kx_is_signed

    @property
    def kx_min_arch(self) -> int:
        return self.base.kx_min_arch

    def has(self, q: Qual) -> bool:
        return q in self.quals

    def unqualified(self) -> GType:
        return self.base

    def spelling(self) -> str:
        parts = [q.name.lower() for q in self.quals]
        parts.append(self.base.spelling())
        return " ".join(parts)

    def __repr__(self) -> str:
        return f"TQualified({self.base!r}, {self.quals!r}, {self.space!r})"


# ---------------------------------------------------------------------------
# Canonical singleton instances
# ---------------------------------------------------------------------------

class T:
    """Shared type singletons (the 'T' namespace keeps call-sites terse)."""

    void = TVoid()
    boolean = TBool()
    uchar = TChar()
    schar = TSChar()
    short = TShort()
    ushort = TUShort()
    i32 = TInt()
    u32 = TUInt()
    i64 = TLong()
    u64 = TULong()
    f16 = THalf()
    bf16 = TBF16()
    f32 = TFloat()
    f64 = TDouble()

    # Frequent vectors
    int2 = TVec(i32, 2)
    int4 = TVec(i32, 4)
    uint2 = TVec(u32, 2)
    uint4 = TVec(u32, 4)
    float2 = TVec(f32, 2)
    float4 = TVec(f32, 4)

    @staticmethod
    def scalars() -> List[GType]:
        return [T.boolean, T.uchar, T.schar, T.short, T.ushort,
                T.i32, T.u32, T.i64, T.u64, T.f16, T.bf16, T.f32, T.f64]

    @staticmethod
    def integers() -> List[GType]:
        return [T.uchar, T.schar, T.short, T.ushort,
                T.i32, T.u32, T.i64, T.u64]

    @staticmethod
    def floats() -> List[GType]:
        return [T.f16, T.bf16, T.f32, T.f64]
