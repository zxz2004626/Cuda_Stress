"""Memory access-pattern analysis for GPU kernels.

Given a kernel :class:`~generator_ir.ir.program.Func`, :class:`AccessPatternAnalyzer`
walks every :class:`~generator_ir.ir.nodes.Subscript` (memory access) and classifies
it as one of:

  * ``COALESCED`` -- consecutive lanes hit consecutive addresses
    (index is linear in the thread id with unit stride).
  * ``STRIDED``   -- index is linear in the thread id with a fixed stride > 1.
  * ``BROADCAST`` -- every lane reads the same address (index is uniform across
    the warp: a constant or built from ``blockIdx`` / ``gridDim`` / ``blockDim``).
  * ``RANDOM``    -- the index depends on a runtime value in a non-linear way
    (a non-tid variable, ``tid`` under a non-linear operator, ...).
  * ``REDUCTION`` -- the access is the address of an :class:`~generator_ir.ir.nodes.Atomic`
    read-modify-write.

The classification is structural and conservative: when an index *might* be
coalesced but the stride cannot be proven > 1 (e.g. ``tid * blockDim.x``), it is
reported as ``COALESCED``.  The explicit ``Subscript.stride`` annotation set by
the stride-alteration mutator is always honored, so a subscript whose ``stride``
field is > 1 is reported as ``STRIDED`` regardless of the index shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from generator_ir.ir.nodes import (
    Cast, Expr, Lit, Subscript, ThreadBuiltin, UnOp, Var,
    BinOp, SafeBin,
)
from generator_ir.ir.operators import Arith, SafeOp
from generator_ir.ir.program import Func
from generator_ir.ir.traversal import atomics, subscripts, traverse


class AccessPattern(Enum):
    """Kind of memory access pattern a subscript exhibits."""
    COALESCED = "coalesced"
    STRIDED = "strided"
    BROADCAST = "broadcast"
    RANDOM = "random"
    REDUCTION = "reduction"


@dataclass
class AccessRecord:
    """One classified memory access site.

    ``array_name`` is the name of the base array when it is a plain
    :class:`~generator_ir.ir.nodes.Var` (``arr[idx]`` -> ``"arr"``); otherwise it is
    the empty string and callers should fall back to inspecting
    ``subscript.array`` directly.
    """
    subscript: Subscript
    pattern: AccessPattern
    array_name: str
    index: Optional[Expr]


# ---------------------------------------------------------------------------
# Index-expression algebra w.r.t. the thread id
# ---------------------------------------------------------------------------
#
# ``_form`` summarizes how an expression depends on the thread id ("tid"):
#
#   ("none",   0) -- no tid dependence (a constant / block-uniform value)
#   ("lin",    k) -- linear in tid with integer coefficient ``k`` (|k| is the
#                    stride contribution; sign is irrelevant for coalescing)
#   ("nonlin", 0) -- tid appears in a non-linear position (tid*tid, tid%i with
#                    a tid-dependent i, tid indexing another array, ...)
#
# The thread id is recognized either as ``Var(name="tid")`` (the canonical name
# used throughout the generator/mutators) or as
# ``ThreadBuiltin(source="threadIdx")``.

_NONE: Tuple[str, int] = ("none", 0)
_NONLIN: Tuple[str, int] = ("nonlin", 0)
_LIN1: Tuple[str, int] = ("lin", 1)


def _is_tid_leaf(node) -> bool:
    return ((isinstance(node, Var) and node.name == "tid")
            or (isinstance(node, ThreadBuiltin) and node.source == "threadIdx"))


def _contains_tid(expr) -> bool:
    """True if *expr* references the thread id anywhere in its subtree."""
    return any(_is_tid_leaf(n) for n in traverse(expr))


def _int_value(expr) -> Optional[int]:
    """Integer payload of an int :class:`Lit`, else ``None``."""
    if (isinstance(expr, Lit)
            and isinstance(expr.value, int)
            and not isinstance(expr.value, bool)):
        return expr.value
    return None


_SAFE_TO_ARITH: Dict[SafeOp, Arith] = {
    SafeOp.SADD: Arith.ADD, SafeOp.SSUB: Arith.SUB, SafeOp.SMUL: Arith.MUL,
    SafeOp.SDIV: Arith.DIV, SafeOp.SMOD: Arith.MOD,
    SafeOp.SSHL: Arith.SHL, SafeOp.SSHR: Arith.SHR,
}


def _binop_arith(node) -> Optional[Arith]:
    """Effective arithmetic operator of a ``BinOp`` / ``SafeBin``, else None."""
    if isinstance(node, BinOp):
        return node.op
    if isinstance(node, SafeBin):
        return _SAFE_TO_ARITH.get(node.op)
    return None


def _form(expr) -> Tuple[str, int]:
    """Classify *expr*'s dependence on the thread id (see module docstring)."""
    if expr is None:
        return _NONE
    if isinstance(expr, Lit):
        return _NONE
    if isinstance(expr, Var):
        return _LIN1 if expr.name == "tid" else _NONE
    if isinstance(expr, ThreadBuiltin):
        return _LIN1 if expr.source == "threadIdx" else _NONE

    op = _binop_arith(expr)
    if op is not None:
        lf = _form(expr.lhs)
        rf = _form(expr.rhs)
        if lf[0] == "nonlin" or rf[0] == "nonlin":
            return _NONLIN
        if op in (Arith.ADD, Arith.SUB):
            # Two distinct tid terms (tid +/- tid) collapse to a single
            # coefficient only in the degenerate cases; treat as non-linear.
            if lf[0] == "lin" and rf[0] == "lin":
                return _NONLIN
            return lf if lf[0] == "lin" else rf
        if op == Arith.MUL:
            if lf[0] == "lin" and rf[0] == "lin":
                return _NONLIN                       # tid * tid
            if lf[0] == "lin":
                k = _int_value(expr.rhs)
                return ("lin", lf[1] * k) if k is not None else lf
            if rf[0] == "lin":
                k = _int_value(expr.lhs)
                return ("lin", rf[1] * k) if k is not None else rf
            return _NONE
        if op == Arith.MOD:
            # ``x % y`` is the generator's bounds-safe wrap when ``y`` is
            # tid-free; it preserves the lane->address mapping, so propagate
            # the dividend's form.  A tid-dependent divisor is non-linear.
            if rf[0] == "lin":
                return _NONLIN
            return lf
        if op == Arith.SHL:
            # ``tid << k`` is ``tid * 2**k`` -- a power-of-two stride.
            if lf[0] == "lin" and rf[0] == "none":
                k = _int_value(expr.rhs)
                if k is not None and k >= 0:
                    return ("lin", lf[1] * (1 << k))
            if lf[0] == "lin" or rf[0] == "lin":
                return _NONLIN
            return _NONE
        # DIV, SHR, and bitwise ops on tid are not simple strides.
        if lf[0] == "lin" or rf[0] == "lin":
            return _NONLIN
        return _NONE

    # Cast / UnOp are transparent: their result tracks the operand's form.
    if isinstance(expr, (Cast, UnOp)):
        return _form(expr.arg)

    # Any other node (Subscript, Call, Atomic, Fma, ...) that somehow contains
    # the thread id is, by construction, non-linear in tid.
    return _NONLIN if _contains_tid(expr) else _NONE


def _array_name(subscript: Subscript) -> str:
    """Base-array name for ``arr[idx]`` when the base is a plain ``Var``."""
    return subscript.array.name if isinstance(subscript.array, Var) else ""


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class AccessPatternAnalyzer:
    """Classifies every memory access in a kernel function.

    Analysis runs in the constructor; the result is cached and exposed via
    :meth:`records`, :meth:`pattern_summary`, and :meth:`is_coalesced`.
    """

    def __init__(self, func: Func) -> None:
        atomic_addr_ids: Set[int] = {
            id(a.addr) for a in atomics(func)
            if isinstance(a.addr, Subscript)
        }
        self._records: List[AccessRecord] = [
            self._categorize(s, atomic_addr_ids) for s in subscripts(func)
        ]

    def _categorize(self, subscript: Subscript,
                  atomic_addr_ids: Set[int]) -> AccessRecord:
        if id(subscript) in atomic_addr_ids:
            pattern = AccessPattern.REDUCTION
        else:
            pattern = self._categorize_index(subscript)
        return AccessRecord(
            subscript=subscript,
            pattern=pattern,
            array_name=_array_name(subscript),
            index=subscript.index,
        )

    @staticmethod
    def _categorize_index(subscript: Subscript) -> AccessPattern:
        idx = subscript.index
        if idx is None:
            return AccessPattern.BROADCAST
        form = _form(idx)
        if form[0] == "lin":
            stride = max(abs(form[1]), subscript.stride)
            return AccessPattern.STRIDED if stride > 1 else AccessPattern.COALESCED
        if form[0] == "nonlin":
            return AccessPattern.RANDOM
        # No tid involvement. Any remaining Var is a runtime value -> RANDOM;
        # only constants and block/grid builtins remain -> BROADCAST.
        if any(isinstance(n, Var) for n in traverse(idx)):
            return AccessPattern.RANDOM
        return AccessPattern.BROADCAST

    # -- public results -----------------------------------------------------

    def records(self) -> List[AccessRecord]:
        """All classified access sites, in traversal order."""
        return list(self._records)

    def pattern_summary(self) -> Dict[str, int]:
        """Count of each pattern keyed by ``AccessPattern`` name."""
        out: Dict[str, int] = {}
        for rec in self._records:
            out[rec.pattern.name] = out.get(rec.pattern.name, 0) + 1
        return out

    def is_coalesced(self, array_name: str) -> bool:
        """True if *every* access to *array_name* is ``COALESCED``.

        Returns ``False`` when the array has no recorded accesses or when any
        access to it is strided / random / a reduction.
        """
        hits = [r for r in self._records if r.array_name == array_name]
        return bool(hits) and all(
            r.pattern is AccessPattern.COALESCED for r in hits)
