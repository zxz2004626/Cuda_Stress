"""IR traversal, search, and rewrite utilities.

``traverse``-based collection plus the kx_scope-repair and barrier-safety analyses
shared by every mutation operator.  Everything here is generic over the IR
node dataclasses in :mod:`generator_ir.ir`.
"""

from __future__ import annotations

import copy
import re
from typing import Callable, Dict, List, Optional, Set

from generator_ir.ir.typedecl import (
    GType, MemSpace, Qual, TArray, TPointer, TQualified, Uniformity, T,
)
from generator_ir.ir.operators import Arith, Cmp, SafeOp
from generator_ir.ir.nodes import (
    Expr, Lit, Var, ThreadBuiltin, BinOp, RelOp, LogicOp, UnOp, Select,
    Cast, Fma, SafeBin, Subscript, Member, AddrOf, Deref, Call, Atomic,
    WarpShuffle, WarpVote,
)
from generator_ir.ir.statements import (
    Stmt, Assign, CompoundAssign, IncDecStmt, ExprStmt, Declare, Block,
    If, For, While, DoWhile, Switch, Case, Return, Break, Continue,
    Barrier, Fence, UnrollHint, RawCode,
)
from generator_ir.ir.program import Func, Program


# ---------------------------------------------------------------------------
# Deep copy
# ---------------------------------------------------------------------------

def clone_program(program: Program) -> Program:
    """A fully independent deep copy."""
    return copy.deepcopy(program)


# ---------------------------------------------------------------------------
# Generic traversal
# ---------------------------------------------------------------------------

def child_nodes(node) -> List:
    """Direct IR children of *node* (expressions/statements/declarations)."""
    out: List = []

    def add(x):
        if x is not None:
            out.append(x)

    if isinstance(node, Program):
        out.extend(node.decls)
    elif isinstance(node, Func):
        out.extend(node.params or [])
        add(node.body)
    elif isinstance(node, Block):
        out.extend(node.stmts)
    elif isinstance(node, If):
        add(node.cond)
        add(node.then)
        add(node.otherwise)
    elif isinstance(node, For):
        add(node.start)
        add(node.end)
        add(node.step)
        add(node.body)
    elif isinstance(node, While):
        add(node.cond)
        add(node.body)
    elif isinstance(node, DoWhile):
        add(node.body)
        add(node.cond)
    elif isinstance(node, Switch):
        add(node.expr)
        out.extend(node.cases)
    elif isinstance(node, Case):
        add(node.value)
        add(node.body)
    elif isinstance(node, (Assign, CompoundAssign)):
        add(node.target)
        add(node.value)
    elif isinstance(node, IncDecStmt):
        add(node.target)
    elif isinstance(node, ExprStmt):
        add(node.expr)
    elif isinstance(node, Return):
        add(node.value)
    elif isinstance(node, Declare):
        add(node.init)
    elif isinstance(node, (BinOp, RelOp, LogicOp, SafeBin)):
        add(node.lhs)
        add(node.rhs)
    elif isinstance(node, UnOp):
        add(node.arg)
    elif isinstance(node, Select):
        add(node.cond)
        add(node.when_true)
        add(node.when_false)
    elif isinstance(node, Cast):
        add(node.arg)
    elif isinstance(node, Fma):
        add(node.a)
        add(node.b)
        add(node.c)
    elif isinstance(node, Subscript):
        add(node.array)
        add(node.index)
    elif isinstance(node, Member):
        add(node.obj)
    elif isinstance(node, (AddrOf, Deref)):
        add(node.arg)
    elif isinstance(node, Call):
        out.extend(node.args)
    elif isinstance(node, Atomic):
        add(node.addr)
        add(node.operand)
    elif isinstance(node, WarpShuffle):
        add(node.mask)
        add(node.value)
        add(node.delta)
    elif isinstance(node, WarpVote):
        add(node.mask)
        add(node.predicate)
    # Leaves (Lit, Var, ThreadBuiltin, Barrier, Fence, UnrollHint, RawCode,
    # Break, Continue) have no children.
    return out


def traverse(root):
    """Pre-order traversal yielding each node once (id-based cycle guard)."""
    seen: Set[int] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node is None:
            continue
        nid = id(node)
        if nid in seen:
            continue
        seen.add(nid)
        yield node
        children = child_nodes(node)
        # push reversed so yield order is a natural pre-order
        stack.extend(reversed(children))


def collect(root, pred: Callable) -> list:
    """All nodes under *root* matching *pred*."""
    return [n for n in traverse(root) if pred(n)]


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------

def kernel_of(program: Program) -> Optional[Func]:
    return program.fetch_kernel()


def device_functions(program: Program) -> List[Func]:
    return [f for f in program.functions() if f.kx_is_device() and not f.kx_is_kernel()]


SYNTHETIC_LOOP_VARS = frozenset({"_rep", "_ci", "_bc_i", "_fi"})


def for_loops(root) -> List[For]:
    return collect(root, lambda n: isinstance(n, For))


def mutable_for_loops(root) -> List[For]:
    """Loops a mutation may rewrite (excludes fixed harness loops)."""
    return [l for l in for_loops(root) if l.var not in SYNTHETIC_LOOP_VARS]


def while_loops(root) -> List[While]:
    return collect(root, lambda n: isinstance(n, While))


def assignments(root) -> List[Assign]:
    return collect(root, lambda n: isinstance(n, Assign))


def compound_assignments(root) -> List[CompoundAssign]:
    return collect(root, lambda n: isinstance(n, CompoundAssign))


def subscripts(root) -> List[Subscript]:
    return collect(root, lambda n: isinstance(n, Subscript))


def declarations(root) -> List[Declare]:
    return collect(root, lambda n: isinstance(n, Declare))


def if_statements(root) -> List[If]:
    return collect(root, lambda n: isinstance(n, If))


def ternaries(root) -> List[Select]:
    return collect(root, lambda n: isinstance(n, Select))


def atomics(root) -> List[Atomic]:
    return collect(root, lambda n: isinstance(n, Atomic))


def warp_shuffles(root) -> List[WarpShuffle]:
    return collect(root, lambda n: isinstance(n, WarpShuffle))


def fmas(root) -> List[Fma]:
    return collect(root, lambda n: isinstance(n, Fma))


def barriers(root) -> List[Barrier]:
    return collect(root, lambda n: isinstance(n, Barrier))


def unroll_hints(root) -> List[UnrollHint]:
    return collect(root, lambda n: isinstance(n, UnrollHint))


def thread_builtins(root) -> List[ThreadBuiltin]:
    return collect(root, lambda n: isinstance(n, ThreadBuiltin))


def blocks(root) -> List[Block]:
    return collect(root, lambda n: isinstance(n, Block))


def switches(root) -> List[Switch]:
    return collect(root, lambda n: isinstance(n, Switch))


# ---------------------------------------------------------------------------
# Name utilities
# ---------------------------------------------------------------------------

def kx_fresh_name(prefix: str, taken: Set[str], start: int = 0) -> str:
    n = start
    while True:
        name = f"{prefix}_{n}"
        if name not in taken:
            return name
        n += 1


_IDENT_RE = re.compile(r"[A-Za-z_]\w*")


def all_var_names(root) -> Set[str]:
    """Every variable name referenced/declared, including identifiers inside
    RawCode text (opaque raw C that mutations may introduce) and for-loop
    induction variables."""
    names = {v.name for v in collect(root, lambda n: isinstance(n, Var))}
    names.update(d.name for d in declarations(root))
    names.update(l.var for l in for_loops(root))
    for rc in collect(root, lambda n: isinstance(n, RawCode)):
        names.update(_IDENT_RE.findall(rc.text or ""))
    return names


# ---------------------------------------------------------------------------
# Small predicates
# ---------------------------------------------------------------------------

def is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def log2i(n: int) -> int:
    r = 0
    while n > 1:
        n >>= 1
        r += 1
    return r


def unqualified(t: GType) -> GType:
    return t.unqualified() if isinstance(t, TQualified) else t


def is_array_like(t: GType) -> bool:
    return isinstance(unqualified(t), (TArray, TPointer))


# ---------------------------------------------------------------------------
# Global variable replacement
# ---------------------------------------------------------------------------

_EXPR_FIELDS = (
    "lhs", "rhs", "arg", "cond", "when_true", "when_false",
    "a", "b", "c", "array", "index", "obj", "addr", "operand",
    "target", "value", "init", "start", "end", "step",
    "mask", "delta", "predicate", "expr",
)


def replace_var_everywhere(root, name: str, factory: Callable) -> None:
    """Replace every ``Var(name)`` reference with ``factory()`` results.

    *factory* is invoked per site so each replacement is an independent copy.
    """
    for node in traverse(root):
        for f in _EXPR_FIELDS:
            cur = getattr(node, f, None)
            if isinstance(cur, Var) and cur.name == name:
                setattr(node, f, factory())
        args = getattr(node, "args", None)
        if isinstance(args, list):
            for i, a in enumerate(args):
                if isinstance(a, Var) and a.name == name:
                    args[i] = factory()


# ---------------------------------------------------------------------------
# Barrier-placement safety
# ---------------------------------------------------------------------------

# Safety predicates moved to safety_checks.py
from generator_ir.ir.safety_checks import (
    for_trip_uniform,
    kx_loop_body_escapes,
    barrier_ok_in_loop,
    block_is_uniform,
    has_divergent_barrier,
    _is_fv_output_write,
    kernel_has_fv_write,
    _insert_before_last_if,
    fold_into_fv,
    fold_via_opaque_guard,
    keep_value_live,
    _choose_replacement,
    _fix_expr,
    _fix_block,
    fix_free_variables
)


def walk_parent_block(root, target):
    """Find the Block that directly contains target_stmt."""
    for block in blocks(root):
        for s in block.stmts:
            if s is target:
                return block
    return None


# Alias for div_ops.py which uses _of suffix.
walk_parent_block_of = walk_parent_block
