"""Static UB-free gate for reduced generator_ir Programs.

This is deliberately stricter than "still emits valid C".  It checks the
generator invariants that a blind DELETE can destroy: free variables,
unguarded div/mod/shift, unsafe subscripts, definite assignment of scalars,
smem cross-thread protocols, divergent barriers, and warp collectives in
divergent control flow.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

from generator_ir.ir.nodes import (
    Expr, Lit, Var, ThreadBuiltin, BinOp, RelOp, LogicOp, UnOp, Select,
    Cast, Fma, SafeBin, Subscript, Member, AddrOf, Deref, Call, Atomic,
    WarpShuffle, WarpVote,
)
from generator_ir.ir.operators import Arith, Cmp, Compound, Logic, SafeOp, UnaOp
from generator_ir.ir.program import Func, Program, StructDecl
from generator_ir.ir.statements import (
    Stmt, Assign, CompoundAssign, IncDecStmt, ExprStmt, Declare, Block,
    If, For, While, DoWhile, Switch, Case, Return, Break, Continue,
    Barrier, Fence, UnrollHint, RawCode,
)
from generator_ir.ir.traversal import traverse, collect, unqualified, is_array_like
from generator_ir.ir.typedecl import (
    GType, MemSpace, Qual, TBool, TChar, TSChar, TShort, TUShort, TInt,
    TUInt, TLong, TULong, THalf, TBF16, TFloat, TDouble, TArray, TPointer,
    TQualified, TVoid, T,
)
from generator_ir.ir.safety_checks import has_divergent_barrier


_BUILTINS = frozenset({"threadIdx", "blockIdx", "blockDim", "gridDim", "warpSize"})
_RAW_GENVAR_RE = re.compile(r"\b(?:v_\d+|ov_\d+|var_\d+|_fv|_sp_\d+|tid|input|output|n)\b")
_UNSAFE_ARITH = {Arith.DIV, Arith.MOD, Arith.SHL, Arith.SHR}
_SHIFT_MASKS = {3, 7, 15, 31, 63}


class ValidityError(Exception):
    pass


def _strip_casts(e: Optional[Expr]) -> Optional[Expr]:
    while isinstance(e, Cast):
        e = e.arg
    return e


def _var_name(e: Optional[Expr]) -> Optional[str]:
    e = _strip_casts(e)
    if isinstance(e, Var):
        return e.name
    return None


def _lit_value(e: Optional[Expr]):
    e = _strip_casts(e)
    if isinstance(e, Lit):
        return e.value
    return None


# ---------------------------------------------------------------------------
# type helpers
# ---------------------------------------------------------------------------

def _is_scalar_lit_supported(t: Optional[GType]) -> bool:
    if isinstance(t, TQualified):
        t = t.unqualified()
    return isinstance(t, (TBool, TChar, TSChar, TShort, TUShort, TInt, TUInt,
                          TLong, TULong, THalf, TBF16, TFloat, TDouble))


def _is_memory_type(t: Optional[GType]) -> bool:
    if t is None:
        return False
    return is_array_like(t)


def _type_of_decl(d: Declare) -> GType:
    return d.gtype


def _is_shared_decl(d: Declare) -> bool:
    return d.space is MemSpace.SHARED or Qual.SHARED_MEM in d.quals


def _is_global_decl(d: Declare) -> bool:
    return d.space in (MemSpace.GLOBAL, MemSpace.CONSTANT) or Qual.DEVICE in d.quals


# ---------------------------------------------------------------------------
# expression children
# ---------------------------------------------------------------------------

def _expr_children(e: Expr) -> List[Optional[Expr]]:
    if isinstance(e, (BinOp, RelOp, LogicOp, SafeBin)):
        return [e.lhs, e.rhs]
    if isinstance(e, UnOp):
        return [e.arg]
    if isinstance(e, Select):
        return [e.cond, e.when_true, e.when_false]
    if isinstance(e, Cast):
        return [e.arg]
    if isinstance(e, Fma):
        return [e.a, e.b, e.c]
    if isinstance(e, Subscript):
        return [e.array, e.index]
    if isinstance(e, Member):
        return [e.obj]
    if isinstance(e, (AddrOf, Deref)):
        return [e.arg]
    if isinstance(e, Call):
        return list(e.args)
    if isinstance(e, Atomic):
        return [e.addr, e.operand]
    if isinstance(e, WarpShuffle):
        return [e.mask, e.value, e.delta]
    if isinstance(e, WarpVote):
        return [e.mask, e.predicate]
    return []


# ---------------------------------------------------------------------------
# arithmetic safety
# ---------------------------------------------------------------------------

def _guarded_div_mod_rhs(e: Optional[Expr]) -> bool:
    """Pattern library for divisor/modulus expressions that are never zero."""
    e = _strip_casts(e)
    if isinstance(e, BinOp):
        if e.op is Arith.BOR:
            return _lit_value(e.rhs) == 1
        # (x & mask) + 1 is nonzero for any integer x (mask >= 0 keeps only
        # the low bits, so the sum is in [1, mask+1]).  The generator uses
        # this pattern (and x | 1 above) to make raw DIV/MOD safe without
        # paying for a SafeBin on the divisor.
        if e.op is Arith.ADD:
            lhs = _strip_casts(e.lhs)
            if isinstance(lhs, BinOp) and lhs.op is Arith.BAND:
                mask = _lit_value(lhs.rhs)
                return isinstance(mask, int) and mask >= 0
    if isinstance(e, Select):
        # cond ? y : z is nonzero when both arms are provably nonzero.
        return (_guarded_div_mod_rhs(e.when_true)
                and _guarded_div_mod_rhs(e.when_false))
    return False


def _guarded_shift_rhs(e: Optional[Expr]) -> bool:
    e = _strip_casts(e)
    if isinstance(e, BinOp) and e.op is Arith.BAND:
        return _lit_value(e.rhs) in _SHIFT_MASKS
    return False


# ---------------------------------------------------------------------------
# lightweight interval analysis for provably-in-range indices and safe raw ops
# ---------------------------------------------------------------------------

_RANGE_UNKNOWN = object()


def _rng(*pairs):
    valid = [p for p in pairs if p is not None]
    if not valid:
        return None
    los = [a for a, _ in valid]
    his = [b for _, b in valid]
    return (min(los), max(his))


def _rng_add(a, b):
    if a is None or b is None:
        return None
    return (a[0] + b[0], a[1] + b[1])


def _rng_sub(a, b):
    if a is None or b is None:
        return None
    return (a[0] - b[1], a[1] - b[0])


def _rng_mul(a, b):
    if a is None or b is None:
        return None
    vals = [a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]]
    return (min(vals), max(vals))


def _rng_bitwise_with_lit(a, lit, op):
    if a is None:
        return None
    lo, hi = a
    if hi - lo > 1024:
        return None
    start = max(lo, -1024)
    end = min(hi, 1024)
    if start > end:
        return None
    vals = []
    for v in range(start, end + 1):
        if op is Arith.BAND:
            vals.append(v & lit)
        elif op is Arith.BXOR:
            vals.append(v ^ lit)
        else:
            return None
    return (min(vals), max(vals))


def _build_range_env(program: Program, func: Func) -> Dict[str, tuple]:
    lc = program.launch
    env: Dict[str, tuple] = {
        "threadIdx.x": (0, lc.block[0] - 1),
        "threadIdx.y": (0, lc.block[1] - 1),
        "threadIdx.z": (0, lc.block[2] - 1),
        "blockIdx.x": (0, lc.grid[0] - 1),
        "blockIdx.y": (0, lc.grid[1] - 1),
        "blockIdx.z": (0, lc.grid[2] - 1),
        "blockDim.x": (lc.block[0], lc.block[0]),
        "blockDim.y": (lc.block[1], lc.block[1]),
        "blockDim.z": (lc.block[2], lc.block[2]),
        "gridDim.x": (lc.grid[0], lc.grid[0]),
        "gridDim.y": (lc.grid[1], lc.grid[1]),
        "gridDim.z": (lc.grid[2], lc.grid[2]),
        "warpSize": (32, 32),
        "tid": (0, max(0, _total_threads(program) - 1)),
    }
    # Seed simple loop/declarations ranges; repeat to pick up dependent decls.
    for _ in range(8):
        changed = False
        for node in traverse(func):
            if isinstance(node, Declare) and node.name not in env and node.init is not None:
                r = _range_of(node.init, env)
                if r is not None:
                    env[node.name] = r
                    changed = True
            elif isinstance(node, For):
                s = _range_of(node.start, env)
                e = _range_of(node.end, env)
                st = _range_of(node.step, env)
                if node.var not in env and s is not None and e is not None and st is not None:
                    if s[0] == s[1] and e[0] == e[1] and st[0] == st[1] and st[0] > 0:
                        env[node.var] = (s[0], max(s[0], e[0] - st[0]))
                        changed = True
        if not changed:
            break
    return env


def _range_of(e: Optional[Expr], env: Dict[str, tuple]):
    """Return (lo, hi) inclusive integer bounds, or None if unknown."""
    if e is None:
        return None
    e = _strip_casts(e)
    if isinstance(e, Lit):
        v = e.value
        if isinstance(v, bool):
            return (int(v), int(v))
        if isinstance(v, int):
            return (v, v)
        return None
    if isinstance(e, Var):
        return env.get(e.name)
    if isinstance(e, ThreadBuiltin):
        if e.source == "warpSize":
            return (32, 32)
        return env.get(f"{e.source}.{e.axis}")
    if isinstance(e, BinOp):
        a = _range_of(e.lhs, env)
        b = _range_of(e.rhs, env)
        if a is None and b is None:
            return None
        if e.op is Arith.ADD:
            return _rng_add(a, b)
        if e.op is Arith.SUB:
            return _rng_sub(a, b)
        if e.op is Arith.MUL:
            return _rng_mul(a, b)
        if e.op is Arith.BAND:
            lit = _lit_value(e.rhs)
            if isinstance(lit, int):
                return _rng_bitwise_with_lit(a, lit, Arith.BAND)
            return (0, 0xFFFFFFFF) if a is not None and a[0] >= 0 else None
        if e.op is Arith.BXOR:
            lit = _lit_value(e.rhs)
            if isinstance(lit, int):
                return _rng_bitwise_with_lit(a, lit, Arith.BXOR)
            return None
        if e.op is Arith.DIV:
            if b is not None and a is not None and b[0] > 0 and a[0] >= 0:
                return (a[0] // b[1], a[1] // b[0])
            return None
        if e.op in (Arith.SHL, Arith.SHR):
            return None
        return None
    if isinstance(e, SafeBin):
        if e.op is SafeOp.SMOD:
            m = _lit_value(e.rhs)
            if isinstance(m, int) and m > 0:
                return (0, m - 1)
            rhs = _range_of(e.rhs, env)
            if rhs is not None and rhs[0] > 0:
                return (0, rhs[1] - 1)
        return None
    if isinstance(e, Select):
        a = _range_of(e.when_true, env)
        b = _range_of(e.when_false, env)
        return _rng(a, b)
    return None


def _interval_in_range(e: Optional[Expr], env: Dict[str, tuple], lo: int, hi: int) -> bool:
    r = _range_of(e, env)
    return r is not None and r[0] >= lo and r[1] < hi


def _arith_errors(program: Program) -> List[str]:
    """Context-aware arithmetic safety.

    Instead of a flat traversal with a single global range environment, the
    checker walks the structured control flow and refines the environment with
    branch conditions (e.g. ``if (rhs != 0)`` makes raw DIV/MOD safe inside
    the branch).  It is intentionally monotone: a path that cannot be
    represented is left unchanged, so the original program remains accepted.
    """
    errs: List[str] = []
    kernel = program.fetch_kernel()
    if kernel is None:
        return errs
    base_env = _build_range_env(program, kernel)

    def check_expr(e: Optional[Expr], env: Dict[str, tuple]) -> None:
        for n in traverse(e):
            if isinstance(n, BinOp):
                if n.op in _UNSAFE_ARITH:
                    ok = ((n.op in (Arith.DIV, Arith.MOD)
                           and (_guarded_div_mod_rhs(n.rhs)
                                or _raw_div_mod_safe(n, env)))
                          or (n.op in (Arith.SHL, Arith.SHR)
                              and (_guarded_shift_rhs(n.rhs)
                                   or _raw_shift_safe(n, env))))
                    if not ok:
                        errs.append(f"unguarded raw BinOp {n.op.name}")
            elif isinstance(n, CompoundAssign):
                if n.op in (Compound.DIV, Compound.MOD, Compound.SHL, Compound.SHR):
                    errs.append(f"unguarded raw CompoundAssign {n.op.name}")

    def walk_block(block: Block, env: Dict[str, tuple]) -> None:
        for s in block.stmts:
            if isinstance(s, (Assign, CompoundAssign, IncDecStmt)):
                if isinstance(s, Assign):
                    check_expr(s.target, env)
                    check_expr(s.value, env)
                elif isinstance(s, CompoundAssign):
                    check_expr(s.target, env)
                    check_expr(s.value, env)
                else:
                    check_expr(s.target, env)
            elif isinstance(s, ExprStmt):
                check_expr(s.expr, env)
            elif isinstance(s, Declare):
                check_expr(s.init, env)
            elif isinstance(s, Return):
                check_expr(s.value, env)
            elif isinstance(s, If):
                check_expr(s.cond, env)
                then_env = _copy_env(env)
                _assume_cond(then_env, s.cond, True)
                walk_block(s.then, then_env)
                if s.otherwise is not None:
                    else_env = _copy_env(env)
                    _assume_cond(else_env, s.cond, False)
                    walk_block(s.otherwise, else_env)
            elif isinstance(s, For):
                check_expr(s.start, env)
                check_expr(s.end, env)
                check_expr(s.step, env)
                body_env = _copy_env(env)
                _assume_for_range(body_env, s)
                walk_block(s.body, body_env)
            elif isinstance(s, (While, DoWhile)):
                check_expr(s.cond, env)
                body_env = _copy_env(env)
                _assume_cond(body_env, s.cond, True)
                if isinstance(s, DoWhile):
                    walk_block(s.body, body_env)
                else:
                    walk_block(s.body, body_env)
            elif isinstance(s, Switch):
                check_expr(s.expr, env)
                for c in s.cases:
                    if c.value is not None:
                        check_expr(c.value, env)
                    walk_block(c.body, _copy_env(env))
            elif isinstance(s, Block):
                walk_block(s, env)
            # Barrier/Fence/RawCode/Break/Continue have no arithmetic.

    for func in program.decls:
        if isinstance(func, Func) and func.body is not None:
            walk_block(func.body, _copy_env(base_env))
    return errs


def _copy_env(env: Dict[str, tuple]) -> Dict[str, tuple]:
    return dict(env)


def _assume_for_range(env: Dict[str, tuple], loop: For) -> None:
    """Refine env with the For loop's induction-variable range."""
    s = _range_of(loop.start, env)
    e = _range_of(loop.end, env)
    st = _range_of(loop.step, env)
    if s is None or e is None or st is None:
        return
    if st[0] == st[1] and st[0] > 0 and s[0] == s[1] and e[0] == e[1]:
        env[loop.var] = (s[0], max(s[0], e[0] - st[0]))
        return
    if st[0] == st[1] and st[0] != 0:
        # General monotone loop: last value before start >= end.
        # Keep it conservative but useful for positive steps.
        if st[0] > 0:
            lo = min(s[0], s[1])
            hi = max(s[0], max(s[0], e[1] - st[0] if e[1] - st[0] >= lo else lo))
            env[loop.var] = (lo, hi)
    # Negative step is uncommon in generator loops; leave env untouched.


def _assume_cond(env: Dict[str, tuple], cond: Optional[Expr], positive: bool) -> None:
    """Refine ranges in *env* using a structured branch condition.

    Only simple Var-against-constant relational patterns are recognized.  If a
    fact cannot be represented (e.g. an exclusion with both positive and
    negative values), the environment is conservatively left unchanged.
    """
    e = _strip_casts(cond)
    if isinstance(e, LogicOp):
        if e.op is Logic.AND and positive:
            _assume_cond(env, e.lhs, True)
            _assume_cond(env, e.rhs, True)
        # OR branches need a union lattice; skip for now.
        return
    if isinstance(e, UnOp) and e.op is UnaOp.LNOT:
        _assume_cond(env, e.arg, not positive)
        return
    if not isinstance(e, RelOp):
        return
    name, lit, op = None, None, e.op
    lv, rv = _var_name(e.lhs), _var_name(e.rhs)
    if lv is not None and _lit_value(e.rhs) is not None:
        name, lit = lv, _lit_value(e.rhs)
    elif rv is not None and _lit_value(e.lhs) is not None:
        name, lit = rv, _lit_value(e.lhs)
        op = op.swapped()
    else:
        return
    if name is None or lit is None or name in _BUILTINS:
        return
    if not isinstance(lit, int) or isinstance(lit, bool):
        return
    old = env.get(name)
    if old is None:
        # We never create half-open ranges; only an exact equality is useful
        # enough for generated guards.
        if positive and op is Cmp.EQ:
            env[name] = (lit, lit)
        return
    if not positive:
        op = op.negated()
    # Now derive the fact assuming the (possibly negated) condition is true.
    lo, hi = old
    if op is Cmp.EQ:
        if lo <= lit <= hi:
            env[name] = (lit, lit)
        elif lit < lo or lit > hi:
            env[name] = (lo, lo)  # unreachable-ish; keep a point
        return
    if op is Cmp.NE:
        # Only useful when the old range is one-sided and contains zero.
        if lit == 0:
            if lo == 0 and hi > 0:
                env[name] = (1, hi)
            elif hi == 0 and lo < 0:
                env[name] = (lo, -1)
        return
    if op is Cmp.LT:
        env[name] = (lo, min(hi, lit - 1)) if lo <= lit - 1 else (lo, lo)
    elif op is Cmp.LE:
        env[name] = (lo, min(hi, lit)) if lo <= lit else (lo, lo)
    elif op is Cmp.GT:
        env[name] = (max(lo, lit + 1), hi) if lit + 1 <= hi else (hi, hi)
    elif op is Cmp.GE:
        env[name] = (max(lo, lit), hi) if lit <= hi else (hi, hi)


def _raw_div_mod_safe(n: BinOp, env: Dict[str, tuple]) -> bool:
    r = _range_of(n.rhs, env)
    if r is None or r[0] * r[1] <= 0:
        return False
    if r[0] > 0 or r[1] < 0:
        # For signed division the only other trap is INT_MIN / -1.  The
        # generator leaves raw DIV/MOD only on unsigned values (threadIdx.x).
        return True
    return False


def _raw_shift_safe(n: BinOp, env: Dict[str, tuple]) -> bool:
    r = _range_of(n.rhs, env)
    if r is None:
        return False
    lhs = _strip_casts(n.lhs)
    if isinstance(lhs, (Var, ThreadBuiltin, Lit)):
        ty = getattr(lhs, "ty", None)
        width_bits = (ty.width * 8) if ty is not None and hasattr(ty, "width") else 32
    else:
        width_bits = 32
    return 0 <= r[0] and r[1] < width_bits


# ---------------------------------------------------------------------------
# subscript safety
# ---------------------------------------------------------------------------

def _block_threads(program: Program) -> int:
    lc = program.launch
    return int(lc.block[0] * lc.block[1] * lc.block[2])


def _total_threads(program: Program) -> int:
    lc = program.launch
    return int(lc.block_count * lc.threads_per_block)


def _buffer_n(program: Program) -> int:
    lc = program.launch
    if lc.n_elements > 0:
        return int(lc.n_elements)
    return max(_total_threads(program), 1024)


def _buf_size(program: Program) -> int:
    n = _buffer_n(program)
    return n * n if program.launch.kx_is_2d else n


def _subscript_errors(program: Program, original: Optional[Program] = None) -> List[str]:
    errs: List[str] = []
    kernel = program.fetch_kernel()
    if kernel is None:
        return errs

    # Declared array lengths by variable name (function-local + globals).
    array_lens: Dict[str, int] = {}
    shared_names: Set[str] = set()
    for d in program.decls:
        if isinstance(d, Declare):
            total = 1
            inner: GType = d.gtype
            while isinstance(inner, TArray):
                total *= inner.count
                inner = inner.elem
            if total > 1:
                array_lens[d.name] = total
            if _is_shared_decl(d):
                shared_names.add(d.name)
    for d in collect(kernel, lambda n: isinstance(n, Declare)):
        total = 1
        inner: GType = d.gtype
        while isinstance(inner, TArray):
            total *= inner.count
            inner = inner.elem
        if total > 1:
            array_lens[d.name] = total
        if _is_shared_decl(d):
            shared_names.add(d.name)

    buf_size = _buf_size(program)
    block_threads = _block_threads(program)
    total_threads = _total_threads(program)
    kernel_params = {p.name for p in kernel.params}
    env = _build_range_env(program, kernel)

    # Original-index fallback: our rewrite set never edits an existing
    # Subscript.index, so an index identical to one in the UB-free original
    # remains safe.  This is only used when the original program is supplied.
    orig_sub_keys = set()
    if original is not None:
        from .structure import _ser
        orig_kernel = original.fetch_kernel()
        if orig_kernel is not None:
            for sub in collect(orig_kernel, lambda x: isinstance(x, Subscript)):
                if isinstance(sub.array, Var):
                    orig_sub_keys.add((sub.array.name, _ser(sub.index)))

    def is_per_thread_ok(arr_name: str) -> bool:
        if arr_name in ("output", "input"):
            return True
        if arr_name in array_lens and arr_name in shared_names:
            return array_lens[arr_name] >= block_threads
        return False

    def index_ok(idx: Expr, arr_name: str) -> bool:
        idx0 = idx
        idx = _strip_casts(idx)

        # SafeBin(SMOD, _, n) with n the kernel 'n' parameter (or any scalar
        # kernel parameter name) keeps the index in [0, n-1].
        if isinstance(idx, SafeBin) and idx.op is SafeOp.SMOD:
            rhs_name = _var_name(idx.rhs)
            if rhs_name in kernel_params:
                return True
            rhs_lit = _lit_value(idx.rhs)
            if isinstance(rhs_lit, int) and rhs_lit > 0:
                if arr_name in array_lens:
                    return rhs_lit <= array_lens[arr_name]
                if arr_name in ("input", "output"):
                    return rhs_lit <= buf_size
            return False

        if isinstance(idx, (Var, ThreadBuiltin)):
            if isinstance(idx, ThreadBuiltin) and idx.source == "threadIdx":
                if is_per_thread_ok(arr_name):
                    return True
            elif isinstance(idx, Var) and idx.name in ("tid", "threadIdx"):
                if is_per_thread_ok(arr_name):
                    return True
            # Fall through: interval analysis may prove other vars/axes in
            # range (e.g. bin_idx = SMOD(hval, 64) with an unknown hval).

        # Generator's tid & 0xFFFF form (only valid when the buffer really is
        # at least that large).
        if isinstance(idx, BinOp) and idx.op is Arith.BAND:
            mask = _lit_value(idx.rhs)
            if isinstance(mask, int) and mask > 0:
                span = mask + 1
                if arr_name in array_lens:
                    return span <= array_lens[arr_name]
                if arr_name in ("input", "output"):
                    return span <= total_threads or span <= buf_size
            return False

        # Literal index into a known array.
        lit = _lit_value(idx)
        if isinstance(lit, int) and not isinstance(lit, bool):
            if arr_name in array_lens:
                return 0 <= lit < array_lens[arr_name]
            if arr_name in ("input", "output"):
                return 0 <= lit < buf_size

        # Provably in-range computed index (interval analysis).
        if arr_name in array_lens:
            return _interval_in_range(idx0, env, 0, array_lens[arr_name])
        if arr_name in ("input", "output"):
            return _interval_in_range(idx0, env, 0, buf_size)
        return False

    from .structure import _ser
    for sub in collect(kernel, lambda x: isinstance(x, Subscript)):
        if not isinstance(sub.array, Var):
            continue
        arr_name = sub.array.name
        if index_ok(sub.index, arr_name):
            continue
        if (arr_name, _ser(sub.index)) in orig_sub_keys:
            continue
        errs.append(
            f"unsafe subscript index for {arr_name}: "
            f"{type(sub.index).__name__}")
    return errs


# ---------------------------------------------------------------------------
# free variables + lexical scoping
# ---------------------------------------------------------------------------

_RAW_DECL_RE = re.compile(
    r"^\s*(?:(?:extern\s+)?(?:const|volatile|unsigned|signed|int|float|double|"
    r"long|short|char|bool|__half|__nv_bfloat16|void)\s+)+"
    r"(?:\*\s*)*([A-Za-z_]\w*)\s*(?:\[[^]]*\])?\s*(?:=|;)")


def _raw_declared_names(text: str) -> Set[str]:
    """Best-effort extraction of generator variable names declared by opaque
    RawCode statements (e.g. ``int _sp_0 = __syncthreads_or(...);``)."""
    out: Set[str] = set()
    for line in (text or "").split("\n"):
        m = _RAW_DECL_RE.match(line)
        if m:
            out.add(m.group(1))
    return out


def _free_variable_errors(program: Program) -> List[str]:
    errs: List[str] = []
    globals_: Set[str] = set()
    for d in program.decls:
        if isinstance(d, Declare):
            globals_.add(d.name)

    for func in program.decls:
        if not (isinstance(func, Func) and func.body is not None):
            continue
        visible: Dict[str, GType] = {}
        for name in globals_:
            visible[name] = T.u32
        for p in func.params:
            visible[p.name] = p.gtype
        errs.extend(_free_vars_block(func.body, dict(visible)))
    return errs


def _free_vars_block(block: Block, visible: Dict[str, GType]) -> List[str]:
    errs: List[str] = []
    for s in block.stmts:
        errs.extend(_free_vars_stmt(s, visible))
        # Declarations are visible only after their statement in C.
        if isinstance(s, Declare):
            visible[s.name] = s.gtype
        elif isinstance(s, RawCode):
            for name in _raw_declared_names(s.text):
                visible[name] = T.u32
    return errs


def _free_vars_expr(e: Optional[Expr], visible: Dict[str, GType]) -> List[str]:
    if e is None:
        return []
    errs: List[str] = []
    if isinstance(e, Var) and e.name not in visible and e.name not in _BUILTINS:
        errs.append(f"free variable {e.name!r}")
    for child in _expr_children(e):
        errs.extend(_free_vars_expr(child, visible))
    return errs


def _free_vars_stmt(s: Stmt, visible: Dict[str, GType]) -> List[str]:
    errs: List[str] = []
    if isinstance(s, Assign):
        errs.extend(_free_vars_expr(s.target, visible))
        errs.extend(_free_vars_expr(s.value, visible))
    elif isinstance(s, CompoundAssign):
        errs.extend(_free_vars_expr(s.target, visible))
        errs.extend(_free_vars_expr(s.value, visible))
    elif isinstance(s, IncDecStmt):
        errs.extend(_free_vars_expr(s.target, visible))
    elif isinstance(s, ExprStmt):
        errs.extend(_free_vars_expr(s.expr, visible))
    elif isinstance(s, Declare):
        errs.extend(_free_vars_expr(s.init, visible))
    elif isinstance(s, Return):
        errs.extend(_free_vars_expr(s.value, visible))
    elif isinstance(s, If):
        errs.extend(_free_vars_expr(s.cond, visible))
        errs.extend(_free_vars_block(s.then, dict(visible)))
        if s.otherwise is not None:
            errs.extend(_free_vars_block(s.otherwise, dict(visible)))
    elif isinstance(s, For):
        errs.extend(_free_vars_expr(s.start, visible))
        errs.extend(_free_vars_expr(s.end, visible))
        errs.extend(_free_vars_expr(s.step, visible))
        child = dict(visible)
        child[s.var] = T.i32
        errs.extend(_free_vars_block(s.body, child))
    elif isinstance(s, (While, DoWhile)):
        errs.extend(_free_vars_expr(s.cond, visible))
        errs.extend(_free_vars_block(s.body, dict(visible)))
    elif isinstance(s, Switch):
        errs.extend(_free_vars_expr(s.expr, visible))
        for c in s.cases:
            if c.value is not None:
                errs.extend(_free_vars_expr(c.value, visible))
            errs.extend(_free_vars_block(c.body, dict(visible)))
    elif isinstance(s, RawCode):
        raw_declared = _raw_declared_names(s.text or "")
        for name in _RAW_GENVAR_RE.findall(s.text or ""):
            if name in raw_declared:
                continue
            if name not in visible and name not in _BUILTINS:
                errs.append(f"RawCode references undeclared {name!r}")
    return errs


# ---------------------------------------------------------------------------
# definite assignment
# ---------------------------------------------------------------------------

def _definite_assignment_errors(program: Program) -> List[str]:
    errs: List[str] = []
    globals_: Set[str] = set()
    for d in program.decls:
        if isinstance(d, Declare):
            globals_.add(d.name)

    for func in program.decls:
        if not (isinstance(func, Func) and func.body is not None):
            continue
        initialized: Set[str] = set(globals_)
        for p in func.params:
            initialized.add(p.name)
        # Array/pointer variables are memory objects; reads through
        # Subscript.array are not scalar reads.
        memory_names = _memory_names(func, program)
        errs.extend(_da_block(func.body, set(initialized), memory_names))
    return errs


def _memory_names(func: Func, program: Program) -> Set[Tuple[str, GType]]:
    out: Set[Tuple[str, GType]] = set()
    for p in func.params:
        out.add((p.name, p.gtype))
    for d in collect(func, lambda n: isinstance(n, Declare)):
        out.add((d.name, d.gtype))
    for d in program.decls:
        if isinstance(d, Declare):
            out.add((d.name, d.gtype))
    return out


def _is_memory_name(name: str, memory_names: Set[Tuple[str, GType]]) -> bool:
    for n, t in memory_names:
        if n == name and _is_memory_type(t):
            return True
    return False


def _da_expr(e: Optional[Expr], init: Set[str],
             memory_names: Set[Tuple[str, GType]]) -> List[str]:
    if e is None:
        return []
    errs: List[str] = []
    if isinstance(e, Var):
        if (e.name not in init and e.name not in _BUILTINS
                and not _is_memory_name(e.name, memory_names)):
            errs.append(f"read of uninitialized variable {e.name!r}")
    for child in _expr_children(e):
        errs.extend(_da_expr(child, init, memory_names))
    return errs


def _da_block(block: Block, init: Set[str],
              memory_names: Set[Tuple[str, GType]]) -> List[str]:
    errs: List[str] = []
    cur = set(init)
    for s in block.stmts:
        errs.extend(_da_stmt(s, cur, memory_names))
        _da_apply_defs(s, cur)
    return errs


def _da_stmt(s: Stmt, cur: Set[str],
             memory_names: Set[Tuple[str, GType]]) -> List[str]:
    errs: List[str] = []
    if isinstance(s, Assign):
        errs.extend(_da_expr(s.target, cur, memory_names))
        errs.extend(_da_expr(s.value, cur, memory_names))
    elif isinstance(s, CompoundAssign):
        errs.extend(_da_expr(s.target, cur, memory_names))
        errs.extend(_da_expr(s.value, cur, memory_names))
    elif isinstance(s, IncDecStmt):
        errs.extend(_da_expr(s.target, cur, memory_names))
    elif isinstance(s, ExprStmt):
        errs.extend(_da_expr(s.expr, cur, memory_names))
    elif isinstance(s, Declare):
        errs.extend(_da_expr(s.init, cur, memory_names))
    elif isinstance(s, Return):
        errs.extend(_da_expr(s.value, cur, memory_names))
    elif isinstance(s, If):
        errs.extend(_da_expr(s.cond, cur, memory_names))
        then_cur = set(cur)
        errs.extend(_da_block(s.then, then_cur, memory_names))
        if s.otherwise is not None:
            else_cur = set(cur)
            errs.extend(_da_block(s.otherwise, else_cur, memory_names))
    elif isinstance(s, For):
        errs.extend(_da_expr(s.start, cur, memory_names))
        errs.extend(_da_expr(s.end, cur, memory_names))
        errs.extend(_da_expr(s.step, cur, memory_names))
        body_cur = set(cur)
        body_cur.add(s.var)
        errs.extend(_da_block(s.body, body_cur, memory_names))
    elif isinstance(s, (While, DoWhile)):
        errs.extend(_da_expr(s.cond, cur, memory_names))
        body_cur = set(cur)
        errs.extend(_da_block(s.body, body_cur, memory_names))
    elif isinstance(s, Switch):
        errs.extend(_da_expr(s.expr, cur, memory_names))
        for c in s.cases:
            if c.value is not None:
                errs.extend(_da_expr(c.value, cur, memory_names))
            case_cur = set(cur)
            errs.extend(_da_block(c.body, case_cur, memory_names))
    return errs


def _da_apply_defs(s: Stmt, cur: Set[str]) -> None:
    if isinstance(s, Declare):
        if s.init is not None:
            cur.add(s.name)
        elif _is_memory_type(s.gtype):
            cur.add(s.name)
    elif isinstance(s, Assign):
        if isinstance(s.target, Var):
            cur.add(s.target.name)
    elif isinstance(s, CompoundAssign):
        if isinstance(s.target, Var):
            cur.add(s.target.name)
    elif isinstance(s, IncDecStmt):
        if isinstance(s.target, Var):
            cur.add(s.target.name)
    elif isinstance(s, RawCode):
        # Opaque RawCode may declare and initialize generator variables.
        for name in _raw_declared_names(s.text or ""):
            cur.add(name)


# ---------------------------------------------------------------------------
# smem protocol
# ---------------------------------------------------------------------------

def _smem_protocol_errors(program: Program) -> List[str]:
    """Catch reductions that deleted a shared write or the barrier that made
    it visible, without trying to re-implement a full array-element dataflow.

    A shared read is accepted when, on the same structured-control-flow path,
    at least one write to that array and one __syncthreads() have already been
    observed.  This is deliberately weaker than a full per-element analysis but
    is strong enough for our deletion-only rewrite set: DCE never moves writes
    or barriers, and HDD can only remove them.  If either the only write or the
    only barrier disappears, the read is rejected."""
    kernel = program.fetch_kernel()
    if kernel is None or kernel.body is None:
        return []
    shared_names: Set[str] = set()
    for d in collect(kernel, lambda n: isinstance(n, Declare)):
        if _is_shared_decl(d):
            shared_names.add(d.name)
    if not shared_names:
        return []

    errs: List[str] = []

    def subscripts(e: Optional[Expr]) -> List[Subscript]:
        return collect(e, lambda n: isinstance(n, Subscript)) if e is not None else []

    def check_reads(e: Optional[Expr], barrier_seen: int,
                    writes_before: Dict[str, int]) -> None:
        for sub in subscripts(e):
            if not (isinstance(sub.array, Var)
                    and sub.array.name in shared_names):
                continue
            name = sub.array.name
            if barrier_seen <= 0 or writes_before.get(name, 0) <= 0:
                errs.append(f"shared {name} read before write+barrier")

    def apply_writes(e: Optional[Expr], writes_before: Dict[str, int]) -> None:
        for sub in subscripts(e):
            if isinstance(sub.array, Var) and sub.array.name in shared_names:
                writes_before[sub.array.name] = writes_before.get(sub.array.name, 0) + 1

    def walk_block(block: Block, barrier_seen: int,
                   writes_before: Dict[str, int]) -> Tuple[int, Dict[str, int]]:
        writes_before = dict(writes_before)
        for s in block.stmts:
            if isinstance(s, Assign):
                check_reads(s.value, barrier_seen, writes_before)
                apply_writes(s.target, writes_before)
            elif isinstance(s, CompoundAssign):
                check_reads(s.target, barrier_seen, writes_before)
                check_reads(s.value, barrier_seen, writes_before)
                apply_writes(s.target, writes_before)
            elif isinstance(s, IncDecStmt):
                check_reads(s.target, barrier_seen, writes_before)
                apply_writes(s.target, writes_before)
            elif isinstance(s, ExprStmt):
                check_reads(s.expr, barrier_seen, writes_before)
            elif isinstance(s, Declare):
                check_reads(s.init, barrier_seen, writes_before)
            elif isinstance(s, Return):
                check_reads(s.value, barrier_seen, writes_before)
            elif isinstance(s, Barrier):
                barrier_seen += 1
            elif isinstance(s, If):
                check_reads(s.cond, barrier_seen, writes_before)
                pre_b, pre_w = barrier_seen, dict(writes_before)
                then_b, then_w = walk_block(s.then, pre_b, pre_w)
                if s.otherwise is not None:
                    else_b, else_w = walk_block(s.otherwise, pre_b, pre_w)
                else:
                    else_b, else_w = pre_b, pre_w
                barrier_seen = min(then_b, else_b)
                writes_before = {k: max(then_w.get(k, 0), else_w.get(k, 0))
                                 for k in set(then_w) | set(else_w)}
            elif isinstance(s, For):
                check_reads(s.start, barrier_seen, writes_before)
                check_reads(s.end, barrier_seen, writes_before)
                check_reads(s.step, barrier_seen, writes_before)
                pre_b, pre_w = barrier_seen, dict(writes_before)
                body_b, body_w = walk_block(s.body, pre_b, pre_w)
                barrier_seen = min(pre_b, body_b)
                writes_before = {k: max(pre_w.get(k, 0), body_w.get(k, 0))
                                 for k in set(pre_w) | set(body_w)}
            elif isinstance(s, (While, DoWhile)):
                check_reads(s.cond, barrier_seen, writes_before)
                pre_b, pre_w = barrier_seen, dict(writes_before)
                body_b, body_w = walk_block(s.body, pre_b, pre_w)
                barrier_seen = min(pre_b, body_b)
                writes_before = {k: max(pre_w.get(k, 0), body_w.get(k, 0))
                                 for k in set(pre_w) | set(body_w)}
            elif isinstance(s, Switch):
                check_reads(s.expr, barrier_seen, writes_before)
                pre_b, pre_w = barrier_seen, dict(writes_before)
                case_bs = [pre_b]
                case_ws = [pre_w]
                for c in s.cases:
                    if c.value is not None:
                        check_reads(c.value, pre_b, pre_w)
                    cb, cw = walk_block(c.body, pre_b, pre_w)
                    case_bs.append(cb)
                    case_ws.append(cw)
                barrier_seen = min(case_bs)
                writes_before = {k: max(w.get(k, 0) for w in case_ws)
                                 for k in set().union(*(set(w) for w in case_ws))}
            elif isinstance(s, Block):
                barrier_seen, writes_before = walk_block(s, barrier_seen, writes_before)
        return barrier_seen, writes_before

    walk_block(kernel.body, 0, {})
    return errs


# ---------------------------------------------------------------------------
# divergence / warp collectives
# ---------------------------------------------------------------------------

def _divergence_errors(program: Program) -> List[str]:
    errs: List[str] = []
    kernel = program.fetch_kernel()
    if kernel is None or kernel.body is None:
        return errs
    if has_divergent_barrier(kernel.body):
        errs.append("divergent barrier")
    from generator_ir.ir.divergence import DivergenceAnalyzer, DivergenceKind

    analyzer = DivergenceAnalyzer(kernel)

    def is_div(expr) -> bool:
        return analyzer.expr_kind(expr) is not DivergenceKind.UNIFORM

    def walk_block(block: Block, divergence: bool) -> None:
        for s in block.stmts:
            if isinstance(s, (WarpShuffle, WarpVote)):
                pass
            if isinstance(s, If):
                d = divergence or is_div(s.cond)
                walk_block(s.then, d)
                if s.otherwise is not None:
                    walk_block(s.otherwise, d)
            elif isinstance(s, For):
                d = divergence or is_div(s.start) or is_div(s.end) or is_div(s.step)
                walk_block(s.body, d)
            elif isinstance(s, (While, DoWhile)):
                walk_block(s.body, divergence or is_div(s.cond))
            elif isinstance(s, Switch):
                d = divergence or is_div(s.expr)
                for c in s.cases:
                    walk_block(c.body, d)
            elif isinstance(s, Block):
                walk_block(s, divergence)
            else:
                for sub in collect(s, lambda n: isinstance(n, (WarpShuffle, WarpVote))):
                    if divergence:
                        errs.append("warp collective under divergent control flow")
    walk_block(kernel.body, False)
    return errs


# ---------------------------------------------------------------------------
# non-termination checks
# ---------------------------------------------------------------------------

def _nontermination_errors(program: Program) -> List[str]:
    """Reject programs whose loops can no longer make progress.

    Candidate generators already avoid deleting loop-control-variable writes;
    this is the belt-and-suspenders gate for arbitrary rewritten Programs.
    """
    errs: List[str] = []
    kernel = program.fetch_kernel()
    if kernel is None:
        return errs
    params = {p.name for p in kernel.params}
    globals_ = {d.name for d in program.decls if isinstance(d, Declare)}

    def writes_in_block(block: Block) -> Set[str]:
        out: Set[str] = set()
        for st in block.stmts:
            w = _stmt_writes_var(st)
            if w is not None:
                out.add(w)
            if isinstance(st, For):
                out.update(writes_in_block(st.body))
            elif isinstance(st, (While, DoWhile)):
                out.update(writes_in_block(st.body))
            elif isinstance(st, If):
                out.update(writes_in_block(st.then))
                if st.otherwise is not None:
                    out.update(writes_in_block(st.otherwise))
            elif isinstance(st, Switch):
                for c in st.cases:
                    out.update(writes_in_block(c.body))
            elif isinstance(st, Block):
                out.update(writes_in_block(st))
        return out

    for node in traverse(program):
        if isinstance(node, For):
            step = _lit_value(node.step)
            if isinstance(step, int) and step == 0:
                errs.append(f"non-terminating For loop {node.var!r} (zero step)")
        elif isinstance(node, While):
            cond_vars = {v.name for v in collect(node.cond, lambda n: isinstance(n, Var))}
            cond_vars -= _BUILTINS | params | globals_
            if _lit_value(node.cond) is True:
                if not any(isinstance(s, Break) for s in traverse(node.body)):
                    errs.append("non-terminating while(true) without break")
                continue
            body_writes = writes_in_block(node.body)
            for v in sorted(cond_vars):
                if v not in body_writes:
                    errs.append(
                        f"non-terminating While: condition variable {v!r} "
                        "is never written in loop body")
        elif isinstance(node, DoWhile):
            cond_vars = {v.name for v in collect(node.cond, lambda n: isinstance(n, Var))}
            cond_vars -= _BUILTINS | params | globals_
            body_writes = writes_in_block(node.body)
            for v in sorted(cond_vars):
                if v not in body_writes:
                    errs.append(
                        f"non-terminating DoWhile: condition variable {v!r} "
                        "is never written in loop body")
    return errs


def _stmt_writes_var(s: Stmt) -> Optional[str]:
    if isinstance(s, (Assign, CompoundAssign, IncDecStmt)):
        tgt = s.target
        if isinstance(tgt, Var):
            return tgt.name
    return None


# ---------------------------------------------------------------------------
# top-level gate
# ---------------------------------------------------------------------------

def validity_errors(program: Program, original: Optional[Program] = None,
                    emit_check: bool = True) -> List[str]:
    """Return a list of UB/safety violations (empty == valid)."""
    errs: List[str] = []
    try:
        errs.extend(_free_variable_errors(program))
        errs.extend(_arith_errors(program))
        errs.extend(_subscript_errors(program, original))
        errs.extend(_definite_assignment_errors(program))
        errs.extend(_smem_protocol_errors(program))
        errs.extend(_divergence_errors(program))
        errs.extend(_nontermination_errors(program))
        if emit_check:
            from generator_ir.emit.nvidia import CudaEmitter
            from generator_ir.emit.host_harness import cuda_driver_spec
            try:
                src = CudaEmitter().rendersrc(program)
            except Exception as exc:  # noqa: BLE001
                errs.append(f"emit failed: {exc}")
                return errs
            kernel = program.fetch_kernel()
            if kernel is None:
                errs.append("program has no kernel")
            else:
                try:
                    cuda_driver_spec(program, kernel)
                except Exception as exc:  # noqa: BLE001
                    errs.append(f"cuda_driver_spec failed: {exc}")
    except Exception as exc:  # noqa: BLE001
        errs.append(f"validity checker crashed: {exc}")
    return errs


def valid_ub(program: Program, original: Optional[Program] = None,
             emit_check: bool = True) -> bool:
    return not validity_errors(program, original=original, emit_check=emit_check)
