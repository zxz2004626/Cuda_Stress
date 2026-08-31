"""Barrier-placement safety predicates and free-variable repair.

Split from traversal.py. Used by every mutation that inserts or moves a
__syncthreads() to ensure no divergent-barrier deadlock is introduced.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Set
from generator_ir.ir.statements import (
    Block, If, For, While, DoWhile, Switch, Case, Return,
    Break, Continue, Barrier, RawCode, Declare, Assign,
    CompoundAssign, IncDecStmt, ExprStmt,
)
from generator_ir.ir.nodes import Var, ThreadBuiltin, Subscript, Lit, Cast, Expr, BinOp, RelOp
from generator_ir.ir.typedecl import GType, Qual, T, TQualified, TArray, TPointer, TUInt, TInt, TBool
from generator_ir.ir.operators import Arith, Cmp
from generator_ir.ir.program import Program, Func
# NOTE: generator_ir.ir.traversal re-exports these predicates back out at its own
# module bottom, so importing it here at top would form a load-time cycle
# (traversal -> safety_checks -> traversal) that breaks a fresh
# `import generator_ir.ir.safety_checks`.  All uses are inside function bodies, so
# the import is deferred until after every name traversal re-exports is bound
# (see the trailing import at the end of this module).

def for_trip_uniform(loop: For) -> bool:
    """Constant-bound loop with no threadIdx in start/step => every thread in
    the block iterates the same count (barrier-safe body)."""
    if not isinstance(loop.end, Lit):
        return False
    for sub in (loop.start, loop.step):
        if sub is not None and collect(
                sub, lambda n: (isinstance(n, ThreadBuiltin)
                                and n.source == "threadIdx")
                or (isinstance(n, Var) and n.name == "tid")):
            return False
    return True


def kx_loop_body_escapes(body) -> bool:
    """True if *body* can skip past its end for only some threads: a return,
    or a break/continue that belongs to this loop (not captured by a nested
    loop or switch)."""
    hit = [False]

    def rec(n, in_loop, in_switch):
        if hit[0]:
            return
        if isinstance(n, Return):
            hit[0] = True
            return
        if isinstance(n, Continue) and not in_loop:
            hit[0] = True
            return
        if isinstance(n, Break) and not (in_loop or in_switch):
            hit[0] = True
            return
        if isinstance(n, Block):
            for s in n.stmts:
                rec(s, in_loop, in_switch)
        elif isinstance(n, If):
            rec(n.then, in_loop, in_switch)
            if n.otherwise:
                rec(n.otherwise, in_loop, in_switch)
        elif isinstance(n, (For, While, DoWhile)):
            rec(n.body, True, in_switch)
        elif isinstance(n, Switch):
            for c in n.cases:
                rec(c.body, in_loop, True)

    rec(body, False, False)
    return hit[0]


def barrier_ok_in_loop(kernel_body, target: For) -> bool:
    """True if a barrier placed directly in *target*'s body is reached by all
    threads the same number of times."""
    verdict = [None]

    def rec(n, divergence):
        if verdict[0] is not None:
            return
        if isinstance(n, For):
            uniform = for_trip_uniform(n)
            if n is target:
                verdict[0] = (divergence == 0 and uniform)
                return
            rec(n.body, divergence + (0 if uniform else 1))
        elif isinstance(n, Block):
            for s in n.stmts:
                rec(s, divergence)
        elif isinstance(n, If):
            rec(n.then, divergence + 1)
            if n.otherwise:
                rec(n.otherwise, divergence + 1)
        elif isinstance(n, Switch):
            for c in n.cases:
                rec(c.body, divergence + 1)
        elif isinstance(n, (While, DoWhile)):
            rec(n.body, divergence + 1)

    rec(kernel_body, 0)
    if not verdict[0]:
        return False
    return not kx_loop_body_escapes(target.body)


def block_is_uniform(kernel_body, target: Block) -> bool:
    """True if *target* block sits at divergent depth 0 (not nested inside any
    if/switch/while/do-while or non-uniform for)."""
    verdict = [None]

    def rec(n, divergence):
        if verdict[0] is not None:
            return
        if isinstance(n, Block):
            if n is target:
                verdict[0] = (divergence == 0)
                return
            for s in n.stmts:
                rec(s, divergence)
        elif isinstance(n, If):
            rec(n.then, divergence + 1)
            if n.otherwise:
                rec(n.otherwise, divergence + 1)
        elif isinstance(n, Switch):
            for c in n.cases:
                rec(c.body, divergence + 1)
        elif isinstance(n, (While, DoWhile)):
            rec(n.body, divergence + 1)
        elif isinstance(n, For):
            rec(n.body, divergence + (0 if for_trip_uniform(n) else 1))

    rec(kernel_body, 0)
    return bool(verdict[0])


def has_divergent_barrier(kernel_body) -> bool:
    """True if any __syncthreads is reachable under divergent control flow.

    A control-flow edge is divergent when its predicate (or, for loops, the
    trip bounds) transitively depends on ``threadIdx`` -- the only value that
    differs across lanes of a warp. Branches on uniform values
    (``blockIdx``/``blockDim``/``gridDim``/``warpSize``, parameters, literals)
    do not make a barrier divergent, so a barrier under e.g.
    ``if (blockIdx.x == 0)`` or ``for (i = 0; i < n; i++)`` is safe.

    Classification uses :class:`~generator_ir.ir.divergence.DivergenceAnalyzer`,
    which is sound for this IR: intra-warp divergence originates only in
    ``threadIdx`` and is tracked through dataflow and control flow. The
    body is analysed inside a parameter-free shell ``Func``; parameters and
    globals would seed UNIFORM rather than UNKNOWN, but both kinds are
    non-divergent and so cannot change the result.
    """
    from generator_ir.ir.divergence import DivergenceAnalyzer, DivergenceKind

    shell = Func(name="_barrier_check", params=[], ret=T.void,
                 body=kernel_body)
    analyzer = DivergenceAnalyzer(shell)
    bad = [False]

    def is_div(expr) -> bool:
        # Conservative: treat UNKNOWN as divergent (safety > precision).
        return analyzer.expr_kind(expr) is not DivergenceKind.UNIFORM

    def rec(n, divergence: bool):
        if bad[0]:
            return
        if isinstance(n, Barrier):
            if divergence:
                bad[0] = True
            return
        if isinstance(n, Block):
            for s in n.stmts:
                rec(s, divergence)
        elif isinstance(n, If):
            d = divergence or is_div(n.cond)
            rec(n.then, d)
            if n.otherwise:
                rec(n.otherwise, d)
        elif isinstance(n, Switch):
            d = divergence or is_div(n.expr)
            for c in n.cases:
                rec(c.body, d)
        elif isinstance(n, (While, DoWhile)):
            rec(n.body, divergence or is_div(n.cond))
        elif isinstance(n, For):
            trip = (is_div(n.start) or is_div(n.end) or is_div(n.step))
            rec(n.body, divergence or trip)

    rec(kernel_body, False)
    return bad[0]


# ---------------------------------------------------------------------------
# Observable-fold (_fv) liveness helpers
# ---------------------------------------------------------------------------

def _is_fv_output_write(stmt) -> bool:
    """True for the trailing ``output[...] = _fv`` / ``output[idx] ^= _fv``."""
    if not (isinstance(stmt, Assign)
            and isinstance(stmt.target, Subscript)
            and isinstance(stmt.target.array, Var)
            and stmt.target.array.name == "output"):
        return False
    v = stmt.value
    if isinstance(v, Var) and v.name == "_fv":
        return True
    if isinstance(v, BinOp) and v.op is Arith.BXOR:
        lhs_slot = (isinstance(v.lhs, Subscript)
                    and isinstance(v.lhs.array, Var)
                    and v.lhs.array.name == "output")
        rhs_fv = isinstance(v.rhs, Var) and v.rhs.name == "_fv"
        if lhs_slot and rhs_fv:
            return True
    return False


def kernel_has_fv_write(kernel: Func) -> bool:
    """True when the kernel ends with the single observable _fv output write."""
    if kernel.body is None:
        return False
    for stmt in reversed(kernel.body.stmts):
        if isinstance(stmt, If):
            if isinstance(stmt.then, Block):
                return any(_is_fv_output_write(t) for t in stmt.then.stmts)
            return False
    return False


def _insert_before_last_if(kernel: Func, new_stmts: List[Stmt]) -> bool:
    """Insert statements just before the trailing top-level If (the checksum /
    output region), or at the end when no such If exists."""
    if kernel.body is None:
        return False
    stmts = kernel.body.stmts
    pos = len(stmts)
    for i in range(len(stmts) - 1, -1, -1):
        if isinstance(stmts[i], If):
            pos = i
            break
    for j, s in enumerate(new_stmts):
        stmts.insert(pos + j, s)
    return True


def fold_into_fv(kernel: Func, value: Expr) -> bool:
    """Keep *value* live: ``if (tid < n) _fv ^= (unsigned)value;`` inserted
    before the trailing output write."""
    cond = RelOp(T.boolean, op=Cmp.LT,
                 lhs=Var(T.u32, name="tid"),
                 rhs=Cast(T.u32, to=T.u32, arg=Var(T.i32, name="n")))
    folded = Assign(Var(T.u32, name="_fv"),
                    BinOp(T.u32, op=Arith.BXOR,
                          lhs=Var(T.u32, name="_fv"),
                          rhs=Cast(T.u32, to=T.u32, arg=value)))
    return _insert_before_last_if(kernel, [If(cond=cond, then=Block([folded]))])


def fold_via_opaque_guard(kernel: Func, value: Expr, base: str = "ext") -> bool:
    """Keep *value* live in a kernel WITHOUT an _fv write and WITHOUT volatile:
    guard a real store behind the opaque predicate ``((unsigned)n) >> 31``
    (always 0 at runtime, unprovable by the compiler)."""
    taken = all_var_names(kernel)
    tmp = kx_fresh_name(base, taken)
    tmp_decl = Declare(name=tmp, gtype=T.u32,
                       init=Cast(T.u32, to=T.u32, arg=value))
    opaque = BinOp(T.u32, op=Arith.SHR,
                   lhs=Cast(T.u32, to=T.u32, arg=Var(T.i32, name="n")),
                   rhs=Lit(T.u32, value=31))
    store = Assign(Subscript(T.u32,
                             array=Var(TPointer(T.u32), name="output"),
                             index=Var(T.u32, name="tid")),
                   Var(T.u32, name=tmp))
    return _insert_before_last_if(kernel, [tmp_decl, If(cond=opaque, then=Block([store]))])


def keep_value_live(kernel: Func, value: Expr, base: str = "ext",
                    allow_volatile: bool = False) -> bool:
    """Pin *value* as must-evaluate work without changing the kernel result.

    Order of preference: fold into _fv (observable kernels) -> opaque-guarded
    store (volatile-free) -> volatile sink (only when allowed).
    """
    if kernel.body is None:
        return False
    if not allow_volatile and kernel_has_fv_write(kernel):
        return fold_into_fv(kernel, value)
    if not allow_volatile:
        return fold_via_opaque_guard(kernel, value, base)
    taken = all_var_names(kernel)
    tmp = kx_fresh_name(base, taken)
    sink = kx_fresh_name(f"{base}_sink", taken | {tmp})
    cond = RelOp(T.boolean, op=Cmp.LT,
                 lhs=Var(T.u32, name="tid"),
                 rhs=Cast(T.u32, to=T.u32, arg=Var(T.i32, name="n")))
    guarded = If(cond=cond, then=Block([
        Declare(name=tmp, gtype=T.u32, init=Cast(T.u32, to=T.u32, arg=value)),
        Declare(name=sink, gtype=T.u32, init=Var(T.u32, name=tmp),
                quals=[Qual.VOLATILE]),
    ]))
    return _insert_before_last_if(kernel, [guarded])


# ---------------------------------------------------------------------------
# Free-variable repair (post-splice reinstantiation)
# ---------------------------------------------------------------------------

_BUILTIN_NAMES = frozenset({
    "threadIdx", "blockIdx", "blockDim", "gridDim", "warpSize",
})

# Generator variable names a mutation may have deleted/renamed while a RawCode
# block still references them textually.
_RAW_GENVAR_RE = re.compile(r"\b(?:v_\d+|ov_\d+|var_\d+)\b")


def _choose_replacement(var: Var, visible: Dict[str, GType], want_array: bool) -> Optional[str]:
    items = list(visible.items())
    if want_array:
        arrays = [n for n, t in items if is_array_like(t)]
        if "input" in arrays:
            return "input"
        if arrays:
            return arrays[0]
    vtype = type(unqualified(var.ty)) if var.ty is not None else None
    same = [n for n, t in items
            if not is_array_like(t) and type(unqualified(t)) is vtype]
    if same:
        return same[0]
    scalars = [n for n, t in items if not is_array_like(t)]
    return scalars[0] if scalars else None


def _fix_expr(expr, visible: Dict[str, GType], count) -> None:
    if expr is None:
        return
    array_bases = set()
    for sub in collect(expr, lambda n: isinstance(n, Subscript)):
        if isinstance(sub.array, Var):
            array_bases.add(id(sub.array))
    for v in collect(expr, lambda n: isinstance(n, Var)):
        if v.name in visible or v.name in _BUILTIN_NAMES:
            continue
        repl = _choose_replacement(v, visible, id(v) in array_bases)
        if repl and repl != v.name:
            v.name = repl
            count[0] += 1


def _fix_block(block: Block, visible: Dict[str, GType], count) -> None:
    for s in block.stmts:
        if isinstance(s, Declare):
            _fix_expr(s.init, visible, count)
            visible[s.name] = s.gtype
        elif isinstance(s, (Assign, CompoundAssign)):
            _fix_expr(s.target, visible, count)
            _fix_expr(s.value, visible, count)
        elif isinstance(s, IncDecStmt):
            _fix_expr(s.target, visible, count)
        elif isinstance(s, ExprStmt):
            _fix_expr(s.expr, visible, count)
        elif isinstance(s, Return):
            _fix_expr(s.value, visible, count)
        elif isinstance(s, If):
            _fix_expr(s.cond, visible, count)
            _fix_block(s.then, dict(visible), count)
            if s.otherwise is not None:
                _fix_block(s.otherwise, dict(visible), count)
        elif isinstance(s, For):
            _fix_expr(s.start, visible, count)
            _fix_expr(s.end, visible, count)
            _fix_expr(s.step, visible, count)
            child = dict(visible)
            child[s.var] = T.i32
            _fix_block(s.body, child, count)
        elif isinstance(s, (While, DoWhile)):
            _fix_expr(s.cond, visible, count)
            _fix_block(s.body, dict(visible), count)
        elif isinstance(s, Switch):
            _fix_expr(s.expr, visible, count)
            for c in s.cases:
                if c.value is not None:
                    _fix_expr(c.value, visible, count)
                _fix_block(c.body, dict(visible), count)
        elif isinstance(s, Block):
            _fix_block(s, dict(visible), count)
        # Barrier/Fence/Break/Continue/UnrollHint carry no expressions.
        # RawCode is handled textually by the caller.


def fix_free_variables(program: Program) -> int:
    """Rewrite Variable references that do not resolve in their lexical kx_scope
    to in-kx_scope variables of a compatible kind.  Returns the number of
    repaired references."""
    count = [0]
    globals_: Dict[str, GType] = {}
    for d in program.decls:
        if isinstance(d, Declare):
            globals_[d.name] = d.gtype

    for d in program.decls:
        if not (isinstance(d, Func) and d.body is not None):
            continue
        visible = dict(globals_)
        for p in d.params:
            visible[p.name] = p.gtype
        _fix_block(d.body, visible, count)

        declared = (set(globals_)
                    | {p.name for p in d.params}
                    | {s.name for s in collect(d.body, lambda n: isinstance(n, Declare))})

        def _kx_sub(m, _decl=declared):
            name = m.group(0)
            if name in _decl:
                return name
            count[0] += 1
            return "threadIdx.x"

        for rc in collect(d.body, lambda n: isinstance(n, RawCode)):
            rc.text = _RAW_GENVAR_RE.sub(_kx_sub, rc.text)
    return count[0]


# Deferred from the top of the module to break the traversal <-> safety_checks
# import cycle.  By this point every function above is defined, so traversal's
# bottom-of-module re-export of these names resolves cleanly regardless of
# which module is imported first in a fresh process.
from generator_ir.ir.traversal import (  # noqa: E402
    collect, traverse, unqualified, is_array_like, all_var_names, kx_fresh_name,
)
