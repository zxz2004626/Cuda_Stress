"""Expression nodes for the gpuperf IR.

All nodes are dataclasses whose first field is the result type (``ty``).
A ``uniformity`` annotation records whether the value is known-equal across
the lanes of a warp (used by divergence-aware mutations).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union

from generator_ir.ir.operators import Arith, Cmp, Logic, UnaOp, SafeOp
from generator_ir.ir.typedecl import GType, MemSpace, Uniformity, T


@dataclass(eq=False)
class Expr:
    """Base class for every expression node."""
    ty: GType
    uniformity: Uniformity = field(default=Uniformity.UNKNOWN, repr=False)

    def __hash__(self) -> int:
        return id(self)


# ---------------------------------------------------------------------------
# Leaves
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class Lit(Expr):
    """Literal constant."""
    value: Union[bool, int, float] = 0


@dataclass(eq=False)
class Var(Expr):
    """Reference to a named variable."""
    name: str = ""
    space: MemSpace = field(default=MemSpace.GENERIC)

    @staticmethod
    def loop_names() -> List[str]:
        return ["i", "j", "k", "l"]

    def kx_is_loop_var(self) -> bool:
        return self.name in Var.loop_names()


@dataclass(eq=False)
class ThreadBuiltin(Expr):
    """GPU grid geometry builtin.

    source: threadIdx | blockIdx | blockDim | gridDim | warpSize
    axis:   x | y | z (ignored for warpSize)
    """
    axis: str = "x"
    source: str = "threadIdx"


# ---------------------------------------------------------------------------
# Arithmetic / logic
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class BinOp(Expr):
    op: Arith = Arith.ADD
    lhs: Optional[Expr] = None
    rhs: Optional[Expr] = None


@dataclass(eq=False)
class RelOp(Expr):
    """Comparison producing a boolean."""
    op: Cmp = Cmp.EQ
    lhs: Optional[Expr] = None
    rhs: Optional[Expr] = None


@dataclass(eq=False)
class LogicOp(Expr):
    op: Logic = Logic.AND
    lhs: Optional[Expr] = None
    rhs: Optional[Expr] = None


@dataclass(eq=False)
class UnOp(Expr):
    op: UnaOp = UnaOp.NEG
    arg: Optional[Expr] = None


@dataclass(eq=False)
class Select(Expr):
    """cond ? when_true : when_false."""
    cond: Optional[Expr] = None
    when_true: Optional[Expr] = None
    when_false: Optional[Expr] = None


# ---------------------------------------------------------------------------
# Type manipulation
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class Cast(Expr):
    """(to)(arg).  When soft=True the emitter omits the cast syntax and lets
    C implicit conversion do the work (stresses type-lowering passes)."""
    to: Optional[GType] = None
    arg: Optional[Expr] = None
    soft: bool = False


# ---------------------------------------------------------------------------
# GPU-flavoured compound expressions
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class Fma(Expr):
    """fused multiply-add a*b+c (emitters kx_pick fmaf/__fma_rn/fma)."""
    a: Optional[Expr] = None
    b: Optional[Expr] = None
    c: Optional[Expr] = None


@dataclass(eq=False)
class SafeBin(Expr):
    """UB-guarded div/mod/shift/add/sub/mul (expanded by the emitters)."""
    op: SafeOp = SafeOp.SDIV
    lhs: Optional[Expr] = None
    rhs: Optional[Expr] = None


# ---------------------------------------------------------------------------
# Memory access
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class Subscript(Expr):
    """array[index]; ``stride`` records the intended access pattern and
    ``locked`` tells stride-rewriting mutations to leave it alone."""
    array: Optional[Expr] = None
    index: Optional[Expr] = None
    stride: int = 1
    locked: bool = False


@dataclass(eq=False)
class Member(Expr):
    """obj.field."""
    obj: Optional[Expr] = None
    field: str = ""


@dataclass(eq=False)
class AddrOf(Expr):
    arg: Optional[Expr] = None


@dataclass(eq=False)
class Deref(Expr):
    arg: Optional[Expr] = None


# ---------------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class Call(Expr):
    fn: str = ""
    args: List[Expr] = field(default_factory=list)
    result_ty: Optional[GType] = None


# ---------------------------------------------------------------------------
# Atomics and warp collectives
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class Atomic(Expr):
    """atomic<what><kx_scope>(addr, operand)."""
    what: str = "Add"           # Add Sub Min Max And Or Xor Exch CAS Inc Dec
    addr: Optional[Expr] = None
    operand: Optional[Expr] = None
    kx_scope: str = ""             # "" | "_block" | "_system"


@dataclass(eq=False)
class WarpShuffle(Expr):
    """__shfl_*_sync(mask, value, delta_or_lane[, width])."""
    kind: str = "shfl_sync"     # shfl_sync/up_sync/down_sync/xor_sync
    mask: Optional[Expr] = None
    value: Optional[Expr] = None
    delta: Optional[Expr] = None
    width: int = 32


@dataclass(eq=False)
class WarpVote(Expr):
    """__ballot_sync / __all_sync / __any_sync (mask, predicate)."""
    kind: str = "ballot_sync"
    mask: Optional[Expr] = None
    predicate: Optional[Expr] = None


# ---------------------------------------------------------------------------
# Small convenience constructors (used across generators/mutations)
# ---------------------------------------------------------------------------

def uint_lit(v: int) -> Lit:
    return Lit(T.u32, value=int(v))


def int_lit(v: int) -> Lit:
    return Lit(T.i32, value=int(v))


def uint_var(name: str) -> Var:
    return Var(T.u32, name=name)


def thread_x() -> ThreadBuiltin:
    return ThreadBuiltin(T.u32, uniformity=Uniformity.VARYING,
                         axis="x", source="threadIdx")


def kx_as_uint(e: Expr) -> Cast:
    return Cast(T.u32, to=T.u32, arg=e)
