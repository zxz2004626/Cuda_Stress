"""White-list rewrite candidates for the IR reducer.

Every candidate is a deep-copied Program; the reducer still runs valid_ub and
the pinned interestingness oracle before accepting anything.  These functions
never repair free variables and never edit RawCode text.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
import functools

from generator_ir.ir.nodes import (
    Expr, Lit, Var, ThreadBuiltin, BinOp, RelOp, LogicOp, UnOp, Select,
    Cast, Fma, SafeBin, Subscript, Member, AddrOf, Deref, Call, Atomic,
    WarpShuffle, WarpVote,
)
from generator_ir.ir.operators import Arith, Logic, SafeOp, UnaOp
from generator_ir.ir.program import Func, Program, StructDecl
from generator_ir.ir.statements import (
    Stmt, Assign, CompoundAssign, IncDecStmt, ExprStmt, Declare, Block,
    If, For, While, DoWhile, Switch, Case, Return, Break, Continue,
    Barrier, Fence, UnrollHint, RawCode,
)
from generator_ir.ir.traversal import traverse, collect, clone_program, blocks
from generator_ir.ir.typedecl import (
    GType, MemSpace, Qual, TBool, TChar, TSChar, TShort, TUShort, TInt,
    TUInt, TLong, TULong, THalf, TBF16, TFloat, TDouble, TArray, TPointer,
    TQualified,
)


@dataclass
class Candidate:
    desc: str
    program: Program


# ---------------------------------------------------------------------------
# generic child-path helpers (needed to replace an expression in a clone)
# ---------------------------------------------------------------------------

def child_locations(parent) -> List[tuple]:
    """Return (loc, child) for every direct IR child of *parent*.

    loc is either a field name or (field_name, index) for list fields.
    """
    out: List[tuple] = []
    if parent is None:
        return out
    if isinstance(parent, Program):
        out.extend((("decls", i), d) for i, d in enumerate(parent.decls))
        return out
    if isinstance(parent, Func):
        out.extend((("params", i), p) for i, p in enumerate(parent.params or []))
        if parent.body is not None:
            out.append(("body", parent.body))
        return out
    if isinstance(parent, Block):
        out.extend((("stmts", i), s) for i, s in enumerate(parent.stmts))
        return out
    if isinstance(parent, If):
        out.append(("cond", parent.cond))
        out.append(("then", parent.then))
        if parent.otherwise is not None:
            out.append(("otherwise", parent.otherwise))
        return out
    if isinstance(parent, For):
        out.append(("start", parent.start))
        out.append(("end", parent.end))
        out.append(("step", parent.step))
        out.append(("body", parent.body))
        return out
    if isinstance(parent, While):
        out.append(("cond", parent.cond))
        out.append(("body", parent.body))
        return out
    if isinstance(parent, DoWhile):
        out.append(("body", parent.body))
        out.append(("cond", parent.cond))
        return out
    if isinstance(parent, Switch):
        out.append(("expr", parent.expr))
        out.extend((("cases", i), c) for i, c in enumerate(parent.cases))
        return out
    if isinstance(parent, Case):
        if parent.value is not None:
            out.append(("value", parent.value))
        out.append(("body", parent.body))
        return out
    if isinstance(parent, (Assign, CompoundAssign)):
        out.append(("target", parent.target))
        out.append(("value", parent.value))
        return out
    if isinstance(parent, IncDecStmt):
        out.append(("target", parent.target))
        return out
    if isinstance(parent, ExprStmt):
        out.append(("expr", parent.expr))
        return out
    if isinstance(parent, Return):
        if parent.value is not None:
            out.append(("value", parent.value))
        return out
    if isinstance(parent, Declare):
        if parent.init is not None:
            out.append(("init", parent.init))
        return out
    if isinstance(parent, (BinOp, RelOp, LogicOp, SafeBin)):
        out.append(("lhs", parent.lhs))
        out.append(("rhs", parent.rhs))
        return out
    if isinstance(parent, UnOp):
        out.append(("arg", parent.arg))
        return out
    if isinstance(parent, Select):
        out.append(("cond", parent.cond))
        out.append(("when_true", parent.when_true))
        out.append(("when_false", parent.when_false))
        return out
    if isinstance(parent, Cast):
        out.append(("arg", parent.arg))
        return out
    if isinstance(parent, Fma):
        out.append(("a", parent.a))
        out.append(("b", parent.b))
        out.append(("c", parent.c))
        return out
    if isinstance(parent, Subscript):
        out.append(("array", parent.array))
        out.append(("index", parent.index))
        return out
    if isinstance(parent, Member):
        out.append(("obj", parent.obj))
        return out
    if isinstance(parent, (AddrOf, Deref)):
        out.append(("arg", parent.arg))
        return out
    if isinstance(parent, Call):
        out.extend((("args", i), a) for i, a in enumerate(parent.args))
        return out
    if isinstance(parent, Atomic):
        out.append(("addr", parent.addr))
        out.append(("operand", parent.operand))
        return out
    if isinstance(parent, WarpShuffle):
        out.append(("mask", parent.mask))
        out.append(("value", parent.value))
        out.append(("delta", parent.delta))
        return out
    if isinstance(parent, WarpVote):
        out.append(("mask", parent.mask))
        out.append(("predicate", parent.predicate))
        return out
    return out


def _get_loc(obj, loc):
    if isinstance(loc, tuple):
        field, idx = loc
        return getattr(obj, field)[idx]
    return getattr(obj, loc)


def _set_loc(obj, loc, value) -> None:
    if isinstance(loc, tuple):
        field, idx = loc
        getattr(obj, field)[idx] = value
    else:
        setattr(obj, loc, value)


_PATH_CACHE: Dict[Tuple[int, int], Optional[List]] = {}
_PATH_CACHE_MAX = 4096


def clear_path_cache() -> None:
    """Drop cached (root id, target id) -> path entries.

    IR nodes are eq=False/hash=id objects.  Cached paths are only valid while
    both root and target are alive, so the reducer clears the cache whenever it
    switches to a new current program.
    """
    _PATH_CACHE.clear()


def _path_to(root, target) -> Optional[List]:
    """Return the child path from *root* to *target*, with an id-keyed cache."""
    key = (id(root), id(target))
    if key in _PATH_CACHE:
        return _PATH_CACHE[key]
    stack = [([], root)]
    seen = set()
    while stack:
        path, node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if node is target:
            _cache_path(key, path)
            return path
        for loc, child in child_locations(node):
            if child is not None:
                stack.append((path + [loc], child))
    _cache_path(key, None)
    return None


def _cache_path(key, path) -> None:
    if len(_PATH_CACHE) >= _PATH_CACHE_MAX:
        _PATH_CACHE.pop(next(iter(_PATH_CACHE)), None)
    _PATH_CACHE[key] = path


def _parent_locations(root) -> Dict[int, tuple]:
    """Map id(child) -> (parent, loc) for every IR child of *root*.

    This is the cached replacement for repeated _find_parent_loc scans.
    """
    out: Dict[int, tuple] = {}
    for node in traverse(root):
        for loc, child in child_locations(node):
            if child is not None:
                out[id(child)] = (node, loc)
    return out


def _get_at_path(root, path):
    node = root
    for loc in path:
        node = _get_loc(node, loc)
    return node


def _replace_at_path(root, path, new) -> None:
    if not path:
        return
    parent = _get_at_path(root, path[:-1])
    _set_loc(parent, path[-1], new)


# ---------------------------------------------------------------------------
# use sets
# ---------------------------------------------------------------------------

_IDENT_RE = __import__("re").compile(r"[A-Za-z_]\w*")


def _var_uses(program: Program) -> set:
    return {v.name for v in collect(program, lambda n: isinstance(n, Var))}


def _raw_uses(program: Program) -> set:
    out: set = set()
    for rc in collect(program, lambda n: isinstance(n, RawCode)):
        out.update(_IDENT_RE.findall(rc.text or ""))
    return out


def _call_uses(program: Program) -> set:
    return {c.fn for c in collect(program, lambda n: isinstance(n, Call))}


def _type_tag_uses(program: Program) -> set:
    out: set = set()
    for node in traverse(program):
        for t in _node_types(node):
            tt = t
            while isinstance(tt, TQualified):
                tt = tt.unqualified()
            if hasattr(tt, "tag"):
                out.add(tt.tag)
            if isinstance(tt, TArray):
                inner = tt.elem
                while isinstance(inner, TArray):
                    inner = inner.elem
                if hasattr(inner, "tag"):
                    out.add(inner.tag)
    return out


def _node_types(node):
    out = []
    for attr in ("ty", "gtype", "to", "ret"):
        if hasattr(node, attr):
            v = getattr(node, attr)
            if isinstance(v, GType):
                out.append(v)
    return out


# ---------------------------------------------------------------------------
# pass 1: unused declarations / helpers / structs / globals
# ---------------------------------------------------------------------------

def unused_decl_candidates(program: Program) -> Iterator[Candidate]:
    uses = _var_uses(program) | _raw_uses(program)
    call_uses = _call_uses(program)
    tag_uses = _type_tag_uses(program) | _raw_uses(program)

    # top-level device globals
    for idx, d in enumerate(program.decls):
        if isinstance(d, Declare) and d.name not in uses:
            yield Candidate(
                f"unused-decl: top-level device global {d.name}",
                _program_without_decl_index(program, idx))

    # unreferenced __device__ helpers (kernel is never removed)
    for idx, d in enumerate(program.decls):
        if (isinstance(d, Func) and d.kx_is_device() and not d.kx_is_kernel()
                and d.name not in call_uses and d.name not in uses):
            yield Candidate(
                f"unused-decl: uncalled device helper {d.name}",
                _program_without_decl_index(program, idx))

    # unreferenced struct declarations
    for idx, d in enumerate(program.decls):
        if isinstance(d, StructDecl) and d.struct.tag not in tag_uses:
            yield Candidate(
                f"unused-decl: unreferenced struct {d.struct.tag}",
                _program_without_decl_index(program, idx))

    # local declarations with no remaining use (and no side-effecting init)
    for block in blocks(program):
        path = _path_to(program, block)
        for i, s in enumerate(block.stmts):
            if (isinstance(s, Declare) and s.name not in uses
                    and _expr_side_effect_free(s.init)):
                p = clone_program(program)
                _block_in_clone_at_path(p, path).stmts.pop(i)
                yield Candidate(f"unused-decl: local {s.name}", p)


def _program_without_decl_index(program: Program, idx: int) -> Program:
    p = clone_program(program)
    p.decls.pop(idx)
    return p


def _block_in_clone_at_path(program: Program, path: Optional[List]) -> Block:
    assert path is not None
    return _get_at_path(program, path)


# ---------------------------------------------------------------------------
# pass 2: pure-compute DCE
# ---------------------------------------------------------------------------

_PURE_SAFE_OPS = {Arith.ADD, Arith.SUB, Arith.MUL, Arith.BAND, Arith.BOR,
                  Arith.BXOR}


def _expr_side_effect_free(e: Optional[Expr]) -> bool:
    if e is None:
        return True
    bad = collect(e, lambda n: isinstance(n, (Atomic, WarpShuffle, WarpVote)))
    return not bad


def _loop_control_vars(program: Program) -> set:
    """Names that govern loop termination/trip counts.

    Deleting the only statement that updates one of these can turn a bounded
    loop into an infinite one.  The validity gate cannot observe
    non-termination, so candidate generators must avoid such deletions.
    """
    out: set = set()
    for node in traverse(program):
        if isinstance(node, For):
            out.add(node.var)
            for e in (node.start, node.end, node.step):
                out.update(v.name for v in collect(
                    e, lambda n: isinstance(n, Var)))
        elif isinstance(node, (While, DoWhile)):
            out.update(v.name for v in collect(
                node.cond, lambda n: isinstance(n, Var)))
    return out


def _stmt_writes_var(s: Stmt) -> Optional[str]:
    if isinstance(s, (Assign, CompoundAssign, IncDecStmt)):
        tgt = s.target
        if isinstance(tgt, Var):
            return tgt.name
    return None


def _is_pure_stmt(s: Stmt, loop_vars: Optional[set] = None) -> bool:
    if isinstance(s, (Assign, CompoundAssign, IncDecStmt)):
        tgt = s.target
        if not isinstance(tgt, Var):
            return False
        if tgt.name in ("_fv", "output", "input", "n"):
            return False
        if loop_vars is not None and tgt.name in loop_vars:
            return False
        val = getattr(s, "value", None)
        return _expr_side_effect_free(val) and _expr_side_effect_free(tgt)
    if isinstance(s, ExprStmt):
        return _expr_side_effect_free(s.expr)
    return False


def pure_dce_candidates(program: Program,
                         max_candidates: int = 256,
                         loop_vars: Optional[set] = None) -> Iterator[Candidate]:
    """Pure-compute DCE as chunk deletion (ddmin over removable runs).

    Instead of deleting one pure statement at a time, contiguous removable
    statements in a block are grouped into runs and large chunks are tried
    first.  The static validity gate still re-checks every candidate, so a
    chunk that destroys a live definition is rejected before any GPU work.
    """
    if loop_vars is None:
        loop_vars = _loop_control_vars(program)
    produced = 0

    def _yield_delete(block: Block, path: Optional[List], indices: List[int],
                      run_start: int, run_end: int, desc: str):
        nonlocal produced
        if produced >= max_candidates:
            return
        if not indices:
            return
        delete = set(indices)
        new_stmts = [s for j, s in enumerate(block.stmts)
                     if j not in delete]
        p = clone_program(program)
        _block_in_clone_at_path(p, path).stmts = new_stmts
        produced += 1
        yield Candidate(desc, p)

    for block in blocks(program):
        if produced >= max_candidates:
            break
        path = _path_to(program, block)
        removable = [i for i, s in enumerate(block.stmts)
                     if _is_pure_stmt(s, loop_vars)]
        if not removable:
            continue
        # Group consecutive removable indices into runs.
        runs: List[Tuple[int, int]] = []
        start = prev = removable[0]
        for i in removable[1:]:
            if i == prev + 1:
                prev = i
            else:
                runs.append((start, prev + 1))
                start = prev = i
        runs.append((start, prev + 1))

        for st, end in runs:
            if produced >= max_candidates:
                break
            run_len = end - st
            # Largest candidate first: delete the whole run.
            yield from _yield_delete(
                block, path, list(range(st, end)), st, end,
                f"pure-dce: delete run [{st}:{end}]/{len(block.stmts)}")
            if produced >= max_candidates:
                break
            # ddmin chunking inside the run.
            chunk = max(1, run_len // 2)
            while chunk >= 1 and produced < max_candidates:
                starts = [st, st + chunk]
                if chunk * 2 <= run_len:
                    starts += list(range(st + chunk, end - chunk + 1, chunk))
                starts = list(dict.fromkeys(starts))
                for a in starts:
                    b = min(a + chunk, end)
                    if a >= b or b > end:
                        continue
                    yield from _yield_delete(
                        block, path, list(range(a, b)), a, b,
                        f"pure-dce: delete run chunk [{a}:{b}]/{len(block.stmts)}")
                    if produced >= max_candidates:
                        break
                if chunk == 1:
                    break
                chunk = max(1, chunk // 2)


# ---------------------------------------------------------------------------
# pass 3: whole control-flow deletion
# ---------------------------------------------------------------------------

def cf_delete_candidates(program: Program) -> Iterator[Candidate]:
    for block in blocks(program):
        path = _path_to(program, block)
        for i, s in enumerate(block.stmts):
            if isinstance(s, (If, For, While, DoWhile, Switch)):
                p = clone_program(program)
                _block_in_clone_at_path(p, path).stmts.pop(i)
                yield Candidate(
                    f"cf-delete: whole {type(s).__name__} at {i}", p)


# ---------------------------------------------------------------------------
# pass 3b: branch simplification and unreachable-code deletion
# ---------------------------------------------------------------------------

def branch_simplify_candidates(program: Program) -> Iterator[Candidate]:
    """Fold if(const) branches and delete empty branches.

    Only conditions that are side-effect-free literals (or trivially true
    constant expressions) are folded.  Empty If statements with a pure
    condition are deleted outright.
    """
    for block in blocks(program):
        path = _path_to(program, block)
        for i, s in enumerate(block.stmts):
            if not isinstance(s, If):
                continue
            if not _expr_side_effect_free(s.cond):
                continue
            lit = _const_bool(s.cond)
            then_stmts = list(s.then.stmts)
            else_stmts = list(s.otherwise.stmts) if s.otherwise is not None else []
            replacement: Optional[List[Stmt]] = None
            desc = ""
            if lit is True:
                replacement = then_stmts
                desc = f"if-fold: const-true If at {i}"
            elif lit is False:
                replacement = else_stmts
                desc = f"if-fold: const-false If at {i}"
            elif not then_stmts and not else_stmts:
                replacement = []
                desc = f"if-simplify: empty If at {i}"
            elif not then_stmts and s.otherwise is None:
                replacement = []
                desc = f"if-simplify: empty then If at {i}"
            if replacement is None:
                continue
            p = clone_program(program)
            blk = _block_in_clone_at_path(p, path)
            cloned_if = blk.stmts[i]
            if lit is True:
                rep = list(cloned_if.then.stmts)
            elif lit is False:
                rep = list(cloned_if.otherwise.stmts) if cloned_if.otherwise is not None else []
            else:
                rep = []
            blk.stmts[i:i + 1] = rep
            yield Candidate(desc, p)


def unreachable_delete_candidates(program: Program) -> Iterator[Candidate]:
    """Delete statements that follow Return/Break/Continue in the same block.

    Such statements are unreachable in C; removing them can unlock larger HDD
    chunks in a later pass.
    """
    for block in blocks(program):
        path = _path_to(program, block)
        for i, s in enumerate(block.stmts):
            if not isinstance(s, (Return, Break, Continue)):
                continue
            tail = i + 1
            if tail >= len(block.stmts):
                continue
            # Do not delete declarations that are the only visible definition
            # for a later block?  C scoping is block-based and these statements
            # are already unreachable, so the validity gate is the final judge.
            p = clone_program(program)
            _block_in_clone_at_path(p, path).stmts = block.stmts[:tail]
            yield Candidate(
                f"unreachable-delete: drop {len(block.stmts) - tail} stmts "
                f"after {type(s).__name__} at {i}", p)


def _const_bool(e: Optional[Expr]) -> Optional[bool]:
    e = _strip_casts(e)
    if isinstance(e, Lit):
        if isinstance(e.value, bool):
            return e.value
        if isinstance(e.value, int):
            return bool(e.value)
        return None
    if isinstance(e, UnOp) and e.op is UnaOp.LNOT:
        v = _const_bool(e.arg)
        if v is not None:
            return not v
    if isinstance(e, LogicOp):
        a = _const_bool(e.lhs)
        b = _const_bool(e.rhs)
        if a is not None and b is not None:
            if e.op is Logic.AND:
                return a and b
            if e.op is Logic.OR:
                return a or b
    return None


def _strip_casts(e: Optional[Expr]) -> Optional[Expr]:
    while isinstance(e, Cast):
        e = e.arg
    return e


# ---------------------------------------------------------------------------
# pass 4: block-statement HDD (max-first / ddmin chunk deletion)
# ---------------------------------------------------------------------------

def hdd_candidates_for_block(program: Program, block: Block,
                             max_candidates: int = 64,
                             loop_vars: Optional[set] = None) -> Iterator[Candidate]:
    """Yield large-chunk-first deletion candidates for one block.

    The reducer restarts after every accepted candidate, so this is a greedy
    ddmin over the current block snapshot.
    """
    stmts = block.stmts
    n = len(stmts)
    if n < 2:
        return
    path = _path_to(program, block)
    if loop_vars is None:
        loop_vars = _loop_control_vars(program)
    protected = {j for j, s in enumerate(stmts)
                 if _stmt_writes_var(s) in loop_vars}
    produced = 0

    def deletion_would_remove_protected(st: int, end: int) -> bool:
        for j in range(st, end):
            if j in protected:
                return True
        return False

    chunk = n // 2
    while chunk >= 1 and produced < max_candidates:
        # Try deleting the first chunk, then the second chunk (complement
        # style), then sliding windows for smaller chunk sizes.
        starts = [0, chunk]
        if chunk * 2 <= n:
            starts += list(range(chunk, n - chunk + 1, chunk))
        starts = list(dict.fromkeys(starts))
        for st in starts:
            if st + chunk > n:
                continue
            if deletion_would_remove_protected(st, st + chunk):
                continue
            new_stmts = [s for j, s in enumerate(stmts)
                         if not (st <= j < st + chunk)]
            p = clone_program(program)
            _block_in_clone_at_path(p, path).stmts = new_stmts
            produced += 1
            yield Candidate(
                f"hdd: delete block stmts [{st}:{st + chunk}]/{n}", p)
            if produced >= max_candidates:
                break
        chunk = max(1, chunk // 2)
        if chunk == 1:
            # Single-statement refinement is deliberately limited to small
            # blocks.  For large blocks pure-dce runs later and handles
            # individual pure statements far more cheaply.
            if n <= 24:
                for j in range(n):
                    if j in protected:
                        continue
                    new_stmts = stmts[:j] + stmts[j + 1:]
                    p = clone_program(program)
                    _block_in_clone_at_path(p, path).stmts = new_stmts
                    produced += 1
                    yield Candidate(
                        f"hdd: delete block stmt {j}/{n}", p)
                    if produced >= max_candidates:
                        break
            break


# ---------------------------------------------------------------------------
# pass 5: expression literalization
# ---------------------------------------------------------------------------

_LIT_SUPPORTED_TYPES = (TBool, TChar, TSChar, TShort, TUShort, TInt, TUInt,
                        TLong, TULong, THalf, TBF16, TFloat, TDouble)


def _lit_for_type(t: GType) -> Lit:
    tt = t.unqualified() if isinstance(t, TQualified) else t
    if isinstance(tt, (TBool, TChar, TSChar, TShort, TUShort, TInt, TUInt,
                       TLong, TULong)):
        return Lit(tt, value=0)
    return Lit(tt, value=0.0)


def _expr_replaceable(e: Expr, parent: Any, loc: Any) -> bool:
    if not isinstance(getattr(e, "ty", None), _LIT_SUPPORTED_TYPES):
        return False
    if isinstance(e, (Lit, Var, ThreadBuiltin)):
        return False
    # Never rewrite a subscript index.
    if isinstance(parent, Subscript) and loc == "index":
        return False
    # Never rewrite divisor/modulus/shift amount of a SafeBin.
    if isinstance(parent, SafeBin) and loc == "rhs" and parent.op in (
            SafeOp.SDIV, SafeOp.SMOD, SafeOp.SSHL, SafeOp.SSHR):
        return False
    # Never rewrite warp masks.
    if isinstance(parent, (WarpShuffle, WarpVote)) and loc == "mask":
        return False
    # Never rewrite loop start/end/step.
    if isinstance(parent, For) and loc in ("start", "end", "step"):
        return False
    # Never rewrite smem/global subscripts (or their index is already covered).
    if isinstance(e, Subscript):
        return False
    return True


def expr_lit_candidates(program: Program,
                         max_candidates: int = 192) -> Iterator[Candidate]:
    """Literalize replaceable expressions with ddmin chunking.

    Parent lookup is done once (id-keyed map) instead of scanning the whole
    tree for every expression.  Candidates are yielded lazily and replace
    several expressions at a time for large programs.
    """
    parent_locs = _parent_locations(program)
    path_cache: Dict[int, Optional[List]] = {}
    nodes = [n for n in traverse(program)
             if isinstance(n, Expr)
             and _expr_replaceable(n, *parent_locs.get(id(n), (None, None)))]
    produced = 0

    def _path_for(e: Expr) -> Optional[List]:
        key = id(e)
        if key not in path_cache:
            path_cache[key] = _path_to(program, e)
        return path_cache[key]

    def _yield_chunk(indices: List[int], desc: str):
        nonlocal produced
        if produced >= max_candidates:
            return
        p = clone_program(program)
        # Compute every path on the original tree first, then apply deepest
        # paths first.  If an ancestor and a descendant are both selected in
        # the same chunk, replacing the descendant first keeps the ancestor
        # path valid; the ancestor replacement then overwrites the descendant,
        # which is the intended "replace whole expression" behavior.
        paths: List[Tuple[int, Optional[List], Expr]] = []
        for idx in indices:
            path = _path_for(nodes[idx])
            if path is not None:
                paths.append((len(path), path, nodes[idx]))
        paths.sort(key=lambda x: -x[0])
        for _, path, e in paths:
            _replace_at_path(p, path, _lit_for_type(e.ty))
        produced += 1
        yield Candidate(desc, p)

    n = len(nodes)
    if n == 0:
        return
    # Whole-set first, then ddmin chunks, then a limited number of singles.
    yield from _yield_chunk(list(range(n)), f"expr-lit: replace all {n}")
    chunk = max(1, n // 2)
    while chunk >= 1 and produced < max_candidates:
        starts = [0, chunk]
        if chunk * 2 <= n:
            starts += list(range(chunk, n - chunk + 1, chunk))
        starts = list(dict.fromkeys(starts))
        for a in starts:
            b = min(a + chunk, n)
            if a >= b:
                continue
            yield from _yield_chunk(
                list(range(a, b)),
                f"expr-lit: replace [{a}:{b}]/{n}")
            if produced >= max_candidates:
                break
        if chunk == 1:
            break
        chunk = max(1, chunk // 2)


def _find_parent_loc(root, target):
    """Compatibility wrapper; prefer _parent_locations for bulk lookups."""
    return _parent_locations(root).get(id(target))


# ---------------------------------------------------------------------------
# pass 5b: adjacent loop merging (same body, constant trip counts)
# ---------------------------------------------------------------------------

def merge_adjacent_loop_candidates(program: Program,
                                   max_candidates: int = 64) -> Iterator[Candidate]:
    """Merge adjacent For loops that have the same body and constant trip count.

    ``for (i = 0; i < 3; i += 1) B; for (j = 0; j < 4; j += 1) B;`` becomes a
    single loop with trip count 7 when B does not reference either loop
    variable.  The validity gate re-checks the result.
    """
    from .structure import _ser
    produced = 0
    for block in blocks(program):
        if produced >= max_candidates:
            break
        path = _path_to(program, block)
        stmts = block.stmts
        for i in range(len(stmts) - 1):
            a, b = stmts[i], stmts[i + 1]
            if not (isinstance(a, For) and isinstance(b, For)):
                continue
            if not (_lit_int(a.start) and _lit_int(a.end) and _lit_int(a.step)
                    and _lit_int(b.start) and _lit_int(b.end) and _lit_int(b.step)):
                continue
            a_start, a_end, a_step = _lit_int(a.start), _lit_int(a.end), _lit_int(a.step)
            b_start, b_end, b_step = _lit_int(b.start), _lit_int(b.end), _lit_int(b.step)
            if a_step != b_step or a_start != b_start:
                continue
            if _ser(a.body) != _ser(b.body):
                continue
            body_vars = {v.name for v in collect(a.body, lambda n: isinstance(n, Var))}
            if a.var in body_vars or b.var in body_vars:
                continue
            # Body must be loop-var-free; the two loops must have the same
            # start/step so concatenating trip counts is exact.
            merged_end = a_end + b_end
            if merged_end < 0 or merged_end > 1000000:
                continue
            p = clone_program(program)
            blk = _block_in_clone_at_path(p, path)
            first = blk.stmts[i]
            first.end = Lit(first.end.ty, value=merged_end)
            blk.stmts.pop(i + 1)
            produced += 1
            yield Candidate(
                f"merge-adjacent-loops: merge {a.var} and {b.var} "
                f"({a_end}+{b_end})", p)
            if produced >= max_candidates:
                break


def _lit_int(e: Optional[Expr]):
    e = _strip_casts(e)
    if isinstance(e, Lit) and isinstance(e.value, int) and not isinstance(e.value, bool):
        return e.value
    return None


# ---------------------------------------------------------------------------
# pass 5c: conservative common-subexpression elimination
# ---------------------------------------------------------------------------

_CSE_NAME_RE = __import__("re").compile(r"^_cse_\d+$")


def cse_candidates(program: Program, max_candidates: int = 48) -> Iterator[Candidate]:
    """Hoist duplicated side-effect-free subexpressions within one statement.

    For every Assign/Declare/Return/ExprStmt, count exact (structural) duplicate
    non-leaf scalar subexpressions.  When a subexpression occurs at least twice,
    introduce ``unsigned int _cse_N = <expr>;`` immediately before the statement
    and replace the duplicates with ``_cse_N``.  This reduces emitted source
    size for the generator's heavily repeated arithmetic.
    """
    from .structure import _ser
    produced = 0
    used_names = {v.name for v in collect(program, lambda n: isinstance(n, Var))}
    for block in blocks(program):
        if produced >= max_candidates:
            break
        path = _path_to(program, block)
        for si, s in enumerate(block.stmts):
            if produced >= max_candidates:
                break
            root_expr: Optional[Expr] = None
            if isinstance(s, Assign):
                root_expr = s.value
            elif isinstance(s, Declare):
                root_expr = s.init
            elif isinstance(s, Return):
                root_expr = s.value
            elif isinstance(s, ExprStmt):
                root_expr = s.expr
            elif isinstance(s, If):
                root_expr = s.cond
            if root_expr is None:
                continue
            counts: Dict[str, List[Expr]] = {}
            for e in traverse(root_expr):
                if not isinstance(e, Expr) or isinstance(e, (Lit, Var, ThreadBuiltin)):
                    continue
                if not _expr_side_effect_free(e):
                    continue
                if not _is_scalar_type(getattr(e, "ty", None)):
                    continue
                counts.setdefault(_ser(e), []).append(e)
            for key, dups in counts.items():
                if len(dups) < 2:
                    continue
                if produced >= max_candidates:
                    break
                chosen = dups[0]
                lit_type = _narrow_scalar_type(chosen.ty)
                if lit_type is None:
                    continue
                name = _fresh_cse_name(used_names)
                used_names.add(name)
                p = clone_program(program)
                blk = _block_in_clone_at_path(p, path)
                # Rebuild the paths in the clone for the selected statement.
                clone_stmt = blk.stmts[si]
                clone_root: Optional[Expr] = None
                if isinstance(clone_stmt, Assign):
                    clone_root = clone_stmt.value
                elif isinstance(clone_stmt, Declare):
                    clone_root = clone_stmt.init
                elif isinstance(clone_stmt, Return):
                    clone_root = clone_stmt.value
                elif isinstance(clone_stmt, ExprStmt):
                    clone_root = clone_stmt.expr
                elif isinstance(clone_stmt, If):
                    clone_root = clone_stmt.cond
                if clone_root is None:
                    continue
                clone_dups = []
                for e in traverse(clone_root):
                    if isinstance(e, Expr) and _ser(e) == key:
                        clone_dups.append(e)
                if len(clone_dups) < 2:
                    continue
                for e in clone_dups:
                    _replace_at_path(p, _path_to(p, e), Var(lit_type, name=name))
                # Insert the hoisted declaration before the statement.
                decl = Declare(name=name, gtype=lit_type,
                               init=clone_program(chosen))
                blk.stmts.insert(si, decl)
                produced += 1
                yield Candidate(
                    f"cse: hoist {key[:40]} x{len(clone_dups)}", p)
                break  # one CSE per statement per pass


def _is_scalar_type(t: Optional[GType]) -> bool:
    tt = t.unqualified() if isinstance(t, TQualified) else t
    return isinstance(tt, (TBool, TChar, TSChar, TShort, TUShort, TInt,
                           TUInt, TLong, TULong, THalf, TBF16, TFloat, TDouble))


def _narrow_scalar_type(t: Optional[GType]) -> Optional[GType]:
    tt = t.unqualified() if isinstance(t, TQualified) else t
    if isinstance(tt, (TUInt, TInt, TLong, TULong, TShort, TUShort)):
        return tt
    return tt if isinstance(tt, (TBool, TChar, TSChar, THalf, TBF16, TFloat, TDouble)) else None


def _fresh_cse_name(used_names) -> str:
    i = 0
    while f"_cse_{i}" in used_names:
        i += 1
    return f"_cse_{i}"


# ---------------------------------------------------------------------------
# pass 5d: constant propagation for never-written scalar declarations
# ---------------------------------------------------------------------------

def constant_fold_candidates(program: Program,
                             max_candidates: int = 64) -> Iterator[Candidate]:
    """Replace Var references to never-written, literal-initialized scalars
    with their literal value.  The declaration is left for the unused-decl
    pass to remove, keeping each candidate a pure expression rewrite."""
    produced = 0
    decls = [d for d in traverse(program)
             if isinstance(d, Declare)
             and d.name and d.init is not None
             and _is_scalar_type(d.gtype)
             and _lit_int(d.init) is not None]
    written = {_stmt_writes_var(s) for s in traverse(program)
               if isinstance(s, (Assign, CompoundAssign, IncDecStmt))}
    addr_taken = {v.name for v in traverse(program)
                  if isinstance(v, AddrOf) and isinstance(v.arg, Var)}
    parent_locs = _parent_locations(program)
    path_cache: Dict[int, Optional[List]] = {}

    for d in decls:
        if produced >= max_candidates:
            break
        if d.name in written or d.name in addr_taken:
            continue
        uses = [v for v in traverse(program)
                if isinstance(v, Var) and v.name == d.name]
        if not uses:
            continue
        lit = Lit(d.gtype.unqualified() if isinstance(d.gtype, TQualified) else d.gtype,
                  value=_lit_int(d.init))
        p = clone_program(program)
        for u in uses:
            path = path_cache.get(id(u))
            if path is None:
                path = _path_to(program, u)
                path_cache[id(u)] = path
            if path is not None:
                _replace_at_path(p, path, clone_program(lit))
        produced += 1
        yield Candidate(f"const-fold: replace {d.name} uses", p)


# ---------------------------------------------------------------------------
# pass 5e: inline single-use assignments
# ---------------------------------------------------------------------------

def inline_single_use_candidates(program: Program,
                                 max_candidates: int = 64) -> Iterator[Candidate]:
    """Inline assignments whose target is referenced exactly once.

    ``v = expr; ... use(v) ...`` becomes ``... use(expr) ...`` and the
    assignment is removed.  The validity gate rejects any candidate where the
    expression is not in scope or is evaluated in a different path context.
    """
    produced = 0
    parent_locs = _parent_locations(program)
    all_vars = [v for v in traverse(program) if isinstance(v, Var)]
    var_counts: Dict[str, int] = {}
    for v in all_vars:
        var_counts[v.name] = var_counts.get(v.name, 0) + 1

    for block in blocks(program):
        if produced >= max_candidates:
            break
        path = _path_to(program, block)
        for i, s in enumerate(block.stmts):
            if produced >= max_candidates:
                break
            if not isinstance(s, Assign):
                continue
            tgt = s.target
            if not isinstance(tgt, Var) or not _is_scalar_type(getattr(tgt, "ty", None)):
                continue
            # Exactly one read reference besides this assignment's target.
            if var_counts.get(tgt.name, 0) != 2:
                continue
            if not _expr_side_effect_free(s.value):
                continue
            uses = [v for v in all_vars if v.name == tgt.name and v is not tgt]
            if len(uses) != 1:
                continue
            use = uses[0]
            if (isinstance(s.value, Expr) and use is s.value):
                continue
            parent_loc = parent_locs.get(id(use))
            if parent_loc is None:
                continue
            use_path = _path_to(program, use)
            if use_path is None:
                continue
            p = clone_program(program)
            blk = _block_in_clone_at_path(p, path)
            blk.stmts.pop(i)
            use_path_adj = list(use_path)
            # If the use is in the same block after the removed assignment,
            # its statement index shifts down by one.
            if (len(use_path_adj) > len(path)
                    and use_path_adj[:len(path)] == path
                    and isinstance(use_path_adj[len(path)], tuple)
                    and use_path_adj[len(path)][0] == "stmts"
                    and use_path_adj[len(path)][1] > i):
                old_loc = use_path_adj[len(path)]
                use_path_adj[len(path)] = ("stmts", old_loc[1] - 1)
            _replace_at_path(p, use_path_adj, clone_program(s.value))
            produced += 1
            yield Candidate(
                f"inline-single-use: {tgt.name} used once", p)


# ---------------------------------------------------------------------------
# pass 6: reduce body_repeat / _rep loop trip count
# ---------------------------------------------------------------------------

def repeat_candidates(program: Program) -> Iterator[Candidate]:
    kernel = program.fetch_kernel()
    if kernel is None or kernel.body is None:
        return
    reps = [f for f in traverse(kernel.body)
            if isinstance(f, For) and f.var == "_rep"
            and isinstance(f.start, Lit) and isinstance(f.end, Lit)
            and isinstance(f.step, Lit)]
    current = program.launch.body_repeat
    for rep in reps:
        end_v = rep.end.value
        if not isinstance(end_v, int) or end_v <= 1:
            continue
        for new_v in sorted({end_v - 1, max(1, end_v // 2)}, reverse=True):
            if new_v >= end_v:
                continue
            p = clone_program(program)
            path = _path_to(program, rep)
            rep2 = _get_at_path(p, path)
            rep2.end = Lit(rep2.end.ty, value=new_v)
            p.launch.body_repeat = new_v
            yield Candidate(
                f"reduce-repeat: _rep {end_v} -> {new_v}", p)
    # Also reduce launch metadata if no _rep loop exists.
    if not reps and current > 1:
        p = clone_program(program)
        p.launch.body_repeat = max(1, current - 1)
        yield Candidate(
            f"reduce-repeat: body_repeat {current} -> {p.launch.body_repeat}", p)
