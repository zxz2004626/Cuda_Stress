"""Divergence analysis for GPU kernels.

A *divergent* value is one that may differ between the lanes of a warp; a
*uniform* value is provably equal across lanes. This module computes, for a
single kernel function, a per-variable divergence classification via a
monotone fixpoint over the three-valued lattice::

        DIVERGENT   (top    -- may differ across lanes)
            |
        UNIFORM     (middle -- provably equal across lanes)
            |
        UNKNOWN     (bottom -- no information yet)

The only root cause of intra-warp divergence in a CUDA/HIP block is
``threadIdx``; ``blockIdx``, ``blockDim``, ``gridDim`` and ``warpSize`` are
warp-uniform, as are kernel parameters and literals. From those seeds
divergence flows through expressions (any divergent operand makes the result
divergent) and through control flow: a variable assigned under a divergent
branch becomes divergent even when its right-hand side is itself uniform,
because different lanes execute the assignment.

Kinds only escalate (UNKNOWN -> UNIFORM -> DIVERGENT) and the variable set is
finite, so the fixpoint terminates.

Consumers:

* :mod:`generator_ir.mutate.div_ops` -- selects divergent variables to propagate.
* :mod:`generator_ir.ir.safety_checks` -- classifies the control flow around
  ``__syncthreads`` barriers.

The module is deliberately self-contained (it imports only the IR node,
statement and program types) so it can be reused without pulling in the
traversal layer's import graph.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Dict, List, Optional, Set

from generator_ir.ir.nodes import (
    Expr, Lit, Var, ThreadBuiltin, BinOp, RelOp, LogicOp, UnOp, Select,
    Cast, Fma, SafeBin, Subscript, Member, AddrOf, Deref, Call, Atomic,
    WarpShuffle, WarpVote,
)
from generator_ir.ir.statements import (
    Stmt, Block, If, For, While, DoWhile, Switch, Assign, CompoundAssign,
    IncDecStmt, Declare,
)
from generator_ir.ir.program import Func


class DivergenceKind(Enum):
    """Three-valued divergence lattice (UNKNOWN < UNIFORM < DIVERGENT)."""

    UNKNOWN = auto()     # lattice bottom -- no information yet
    UNIFORM = auto()     # provably equal across the lanes of a warp
    DIVERGENT = auto()   # lattice top    -- may differ across lanes


def _join(a: DivergenceKind, b: DivergenceKind) -> DivergenceKind:
    """Lattice least-upper-bound (the absorbing element is DIVERGENT)."""
    if a is DivergenceKind.DIVERGENT or b is DivergenceKind.DIVERGENT:
        return DivergenceKind.DIVERGENT
    if a is DivergenceKind.UNIFORM or b is DivergenceKind.UNIFORM:
        return DivergenceKind.UNIFORM
    return DivergenceKind.UNKNOWN


# Builtin sources whose value is identical across the lanes of a warp.
_UNIFORM_SOURCES = frozenset({"blockIdx", "blockDim", "gridDim", "warpSize"})


def _expr_children(expr: Expr) -> List[Optional[Expr]]:
    """Direct sub-expression children of *expr*.

    Mirrors the expression branch of :func:`generator_ir.ir.traversal.child_nodes`
    but stays local to this module to keep the analysis import-light.
    """
    if isinstance(expr, (BinOp, RelOp, LogicOp, SafeBin)):
        return [expr.lhs, expr.rhs]
    if isinstance(expr, UnOp):
        return [expr.arg]
    if isinstance(expr, Select):
        return [expr.cond, expr.when_true, expr.when_false]
    if isinstance(expr, Cast):
        return [expr.arg]
    if isinstance(expr, Fma):
        return [expr.a, expr.b, expr.c]
    if isinstance(expr, Subscript):
        return [expr.array, expr.index]
    if isinstance(expr, Member):
        return [expr.obj]
    if isinstance(expr, (AddrOf, Deref)):
        return [expr.arg]
    if isinstance(expr, Call):
        return list(expr.args)
    if isinstance(expr, Atomic):
        return [expr.addr, expr.operand]
    if isinstance(expr, WarpShuffle):
        return [expr.mask, expr.value, expr.delta]
    if isinstance(expr, WarpVote):
        return [expr.mask, expr.predicate]
    return []  # leaves: Lit, Var, ThreadBuiltin


class DivergenceAnalyzer:
    """Fixpoint divergence analysis over a single kernel function.

    Construct with the kernel :class:`~generator_ir.ir.program.Func`; query the
    result with :meth:`categorize`, :meth:`kx_is_divergent`,
    :meth:`divergent_variables` and :meth:`expr_kind`.
    """

    def __init__(self, kernel: Func) -> None:
        self._kernel = kernel
        self._lat: Dict[str, DivergenceKind] = {}
        self._changed = False
        self._fixpoint()

    # -- public query API --------------------------------------------------

    def categorize(self, name: str) -> DivergenceKind:
        """Divergence of the variable *name*.

        Returns :attr:`DivergenceKind.UNKNOWN` for names the analysis has not
        seen (undeclared globals, opaque references).
        """
        return self._lat.get(name, DivergenceKind.UNKNOWN)

    def kx_is_divergent(self, name: str) -> bool:
        """Convenience: ``categorize(name) is DivergenceKind.DIVERGENT``."""
        return self.categorize(name) is DivergenceKind.DIVERGENT

    def divergent_variables(self) -> Set[str]:
        """Every variable classified DIVERGENT (threadIdx & friends are
        expression builtins, not variables, and are excluded)."""
        return {n for n, k in self._lat.items()
                if k is DivergenceKind.DIVERGENT}

    def expr_kind(self, expr: Optional[Expr]) -> DivergenceKind:
        """Divergence of an arbitrary expression.

        Exposed for callers that need to categorize conditions rather than
        named variables (e.g. the barrier-safety check, which must decide
        whether an ``if`` predicate is lane-varying).
        """
        return self._expr_kind(expr)

    # -- fixpoint driver ---------------------------------------------------

    def _fixpoint(self) -> None:
        # Kernel parameters are passed by value and identical for every lane
        # in the block, so they seed UNIFORM.
        for p in self._kernel.params or []:
            self._lat[p.name] = DivergenceKind.UNIFORM
        body = self._kernel.body
        if body is None:
            return
        # Repeat the forward traverse until no lattice entry changes.
        while True:
            self._changed = False
            self._walk_block(body, divergent=False)
            if not self._changed:
                break

    def _escalate(self, name: str, kind: DivergenceKind) -> None:
        """Raise *name* to ``join(current, kind)``; flag change on growth."""
        cur = self._lat.get(name, DivergenceKind.UNKNOWN)
        nxt = _join(cur, kind)
        if nxt is not cur:
            self._lat[name] = nxt
            self._changed = True

    # -- expression classification ----------------------------------------

    def _expr_kind(self, expr: Optional[Expr]) -> DivergenceKind:
        if expr is None:
            return DivergenceKind.UNKNOWN
        if isinstance(expr, Lit):
            return DivergenceKind.UNIFORM
        if isinstance(expr, ThreadBuiltin):
            if expr.source == "threadIdx":
                return DivergenceKind.DIVERGENT
            return (DivergenceKind.UNIFORM
                    if expr.source in _UNIFORM_SOURCES
                    else DivergenceKind.UNKNOWN)
        if isinstance(expr, Var):
            return self._lat.get(expr.name, DivergenceKind.UNKNOWN)
        # Composite expression: divergent if any operand is divergent,
        # uniform only if every operand is uniform.
        acc = DivergenceKind.UNKNOWN
        for child in _expr_children(expr):
            acc = _join(acc, self._expr_kind(child))
        return acc

    # -- statement traverse (tracks divergent control-flow context) -----------

    def _walk_block(self, block: Block, divergent: bool) -> None:
        for s in block.stmts:
            self._walk_stmt(s, divergent)

    def _walk_stmt(self, s: Stmt, divergent: bool) -> None:
        if isinstance(s, Declare):
            if s.init is not None:
                self._escalate(s.name, self._assigned_kind(s.init, divergent))
        elif isinstance(s, Assign):
            if isinstance(s.target, Var):
                self._escalate(s.target.name,
                               self._assigned_kind(s.value, divergent))
            # Subscript/Member/Deref targets write through a location, not to
            # a named variable, so there is nothing to categorize.
        elif isinstance(s, CompoundAssign):
            if isinstance(s.target, Var):
                # ``x op= v``  <=>  ``x = x op v``: the new value of x joins
                # its old value with the right-hand side.
                k = _join(self._lat.get(s.target.name, DivergenceKind.UNKNOWN),
                          self._expr_kind(s.value))
                if divergent:
                    k = _join(k, DivergenceKind.DIVERGENT)
                self._escalate(s.target.name, k)
        elif isinstance(s, IncDecStmt):
            if isinstance(s.target, Var):
                # ``++x``/``--x``: only the +/- 1 (uniform) is added, so the
                # divergence is that of the current value (plus context).
                self._escalate(s.target.name,
                               self._assigned_kind(s.target, divergent))
        elif isinstance(s, If):
            d = divergent or self._is_divergent_expr(s.cond)
            self._walk_block(s.then, d)
            if s.otherwise is not None:
                self._walk_block(s.otherwise, d)
        elif isinstance(s, For):
            trip = _join(self._expr_kind(s.start),
                         _join(self._expr_kind(s.end), self._expr_kind(s.step)))
            self._escalate(s.var, trip)
            self._walk_block(s.body,
                             divergent or trip is DivergenceKind.DIVERGENT)
        elif isinstance(s, (While, DoWhile)):
            self._walk_block(s.body,
                             divergent or self._is_divergent_expr(s.cond))
        elif isinstance(s, Switch):
            d = divergent or self._is_divergent_expr(s.expr)
            for c in s.cases:
                self._walk_block(c.body, d)
        elif isinstance(s, Block):
            self._walk_block(s, divergent)
        # ExprStmt / Return / Barrier / Fence / UnrollHint / RawCode /
        # Break / Continue define no variables and are skipped.

    # -- small helpers -----------------------------------------------------

    def _is_divergent_expr(self, expr: Optional[Expr]) -> bool:
        return self._expr_kind(expr) is DivergenceKind.DIVERGENT

    def _assigned_kind(self, rhs: Optional[Expr], divergent: bool) -> DivergenceKind:
        """Kind to apply to a target assigned from *rhs* in the current
        context: the rhs kind escalated to DIVERGENT when the surrounding
        control flow is itself divergent."""
        k = self._expr_kind(rhs)
        return _join(k, DivergenceKind.DIVERGENT) if divergent else k
