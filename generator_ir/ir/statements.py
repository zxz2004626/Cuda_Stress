"""Statement nodes for the gpuperf IR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from generator_ir.ir.nodes import Expr
from generator_ir.ir.operators import Compound, IncDec
from generator_ir.ir.typedecl import GType, MemSpace, Qual


@dataclass(eq=False)
class Stmt:
    """Base class for statements.

    ``emit_ctx`` optionally carries a snapshot of the generation context that
    produced the statement (used by the weaving mutator).  It is a plain class
    attribute on the base so dataclass machinery ignores it.
    """
    emit_ctx = None  # type: Optional[object]

    def __hash__(self) -> int:
        return id(self)


# ---------------------------------------------------------------------------
# Simple statements
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class Assign(Stmt):
    """target = value."""
    target: Expr
    value: Expr


@dataclass(eq=False)
class CompoundAssign(Stmt):
    """target <op>= value."""
    target: Expr
    op: Compound
    value: Expr


@dataclass(eq=False)
class IncDecStmt(Stmt):
    """++target / target++ / --target / target--."""
    target: Expr
    op: IncDec
    prefix: bool


@dataclass(eq=False)
class ExprStmt(Stmt):
    """Evaluate an expression for its side effects."""
    expr: Expr


@dataclass(eq=False)
class Declare(Stmt):
    """Variable declaration with optional initializer."""
    name: str
    gtype: GType
    space: MemSpace = field(default=MemSpace.GENERIC)
    init: Optional[Expr] = None
    quals: List[Qual] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Structured statements
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class Block(Stmt):
    stmts: List[Stmt] = field(default_factory=list)

    def append(self, s: Stmt) -> None:
        self.stmts.append(s)

    def insert(self, idx: int, s: Stmt) -> None:
        self.stmts.insert(idx, s)

    def replace(self, idx: int, s: Stmt) -> None:
        self.stmts[idx] = s

    def remove(self, idx: int) -> None:
        self.stmts.pop(idx)

    def index_of(self, s: Stmt) -> int:
        return self.stmts.index(s)

    def __len__(self) -> int:
        return len(self.stmts)


@dataclass(eq=False)
class If(Stmt):
    cond: Expr
    then: Block
    otherwise: Optional[Block] = None

    def kx_has_else(self) -> bool:
        return self.otherwise is not None


@dataclass(eq=False)
class For(Stmt):
    """for (int var = start; var < end; var += step) body"""
    var: str
    start: Expr
    end: Expr
    step: Expr
    body: Block


@dataclass(eq=False)
class While(Stmt):
    cond: Expr
    body: Block


@dataclass(eq=False)
class DoWhile(Stmt):
    body: Block
    cond: Expr


@dataclass(eq=False)
class Case(Stmt):
    """One switch arm; value=None is the default arm."""
    value: Optional[Expr]
    body: Block
    has_break: bool = True

    def kx_is_default(self) -> bool:
        return self.value is None


@dataclass(eq=False)
class Switch(Stmt):
    expr: Expr
    cases: List[Case]

    def kx_has_default(self) -> bool:
        return any(c.kx_is_default() for c in self.cases)


@dataclass(eq=False)
class Return(Stmt):
    value: Optional[Expr] = None


@dataclass(eq=False)
class Break(Stmt):
    pass


@dataclass(eq=False)
class Continue(Stmt):
    pass


# ---------------------------------------------------------------------------
# GPU-specific statements
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class Barrier(Stmt):
    """__syncthreads()."""
    pass


@dataclass(eq=False)
class Fence(Stmt):
    """Memory fence at a kx_scope: device | block | system."""
    kx_scope: str = "device"


@dataclass(eq=False)
class UnrollHint(Stmt):
    """#pragma unroll [factor]; factor=None means full unroll."""
    factor: Optional[int] = None


@dataclass(eq=False)
class RawCode(Stmt):
    """Opaque raw source text (inline PTX, cooperative groups, stress blocks).
    Analysis passes must treat the contents as unknown."""
    text: str = ""
