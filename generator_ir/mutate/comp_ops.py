"""Compute/register mutation operators (COMP1..COMP6 + extensions)."""

from __future__ import annotations

import copy
import random
from typing import List, Optional, Set, Tuple

from generator_ir.ir.typedecl import (
    GType, T, MemSpace, Qual, TArray, TPointer, TInt, TUInt, TFloat,
)
from generator_ir.ir.operators import Arith, Cmp, SafeOp, Compound
from generator_ir.ir.nodes import (
    Expr, Lit, Var, ThreadBuiltin, BinOp, RelOp, LogicOp, Cast, SafeBin,
    Subscript, Call, Fma,
)
from generator_ir.ir.statements import (
    Stmt, Assign, CompoundAssign, ExprStmt, Declare, Block, If, For,
    Barrier, Fence, UnrollHint,
)
from generator_ir.ir.program import Func, Program
from generator_ir.ir import traversal as traverse
from generator_ir.ir.traversal import (
    clone_program, kernel_of, collect, mutable_for_loops, assignments,
    compound_assignments, declarations, blocks, all_var_names, kx_fresh_name,
    replace_var_everywhere,
    is_pow2, log2i, fmas as find_fmas, unroll_hints, keep_value_live,
    device_functions,
)
from generator_ir.mutate.foundation import Mutation, cuda_sm_version, kx_is_cuda


def _uint(v: int) -> Lit:
    return Lit(T.u32, value=v)


def _int(v: int) -> Lit:
    return Lit(T.i32, value=v)


def _tid(axis: str = "x") -> ThreadBuiltin:
    return ThreadBuiltin(T.u32, axis=axis, source="threadIdx")


def _parent_block(root, target: Stmt) -> Optional[Block]:
    for block in blocks(root):
        for s in block.stmts:
            if s is target:
                return block
    return None


def _replace_expr_node(root, old: Expr, new: Expr) -> None:
    """Replace every direct reference to *old* (by identity) with *new*."""
    from generator_ir.ir.traversal import _EXPR_FIELDS
    for node in traverse.traverse(root):
        for f in _EXPR_FIELDS:
            if getattr(node, f, None) is old:
                setattr(node, f, new)
        args = getattr(node, "args", None)
        if isinstance(args, list):
            for i, a in enumerate(args):
                if a is old:
                    args[i] = new


# ---------------------------------------------------------------------------
# COMP1: work coarsening
# ---------------------------------------------------------------------------




class CompWorkCoarsen(Mutation):
    """Wrap a mutable loop in a K-way coarsened loop (start/step *= K, inner
    j-loop with (i+j) substitution + bounds guard); grid.x //= K."""

    name = "COMP1_work_coarsen"
    category = "compute"
    sweep_space = [2, 4, 8]

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None or kernel.body is None:
            return prog
        k = param if param is not None else rng.choice(self.sweep_space)
        loops = mutable_for_loops(kernel)
        if not loops:
            return prog
        loop = rng.choice(loops)
        taken = all_var_names(kernel)
        self._coarsen(loop, k, taken, rng)
        gx, gy, gz = prog.launch.grid
        prog.launch.grid = (max(1, gx // k), gy, gz)
        return prog

    def _coarsen(self, loop: For, k: int, taken: set, rng: random.Random) -> None:
        inner = kx_fresh_name("coarse_k", taken)
        taken.add(inner)

        loop.start = BinOp(T.i32, op=Arith.MUL,
                           lhs=copy.deepcopy(loop.start), rhs=_int(k))
        loop.step = BinOp(T.i32, op=Arith.MUL,
                          lhs=copy.deepcopy(loop.step), rhs=_int(k))

        old_body = copy.deepcopy(loop.body)
        combined = BinOp(T.i32, op=Arith.ADD,
                         lhs=Var(T.i32, name=loop.var),
                         rhs=Var(T.i32, name=inner))
        # Substitute loop.var -> (loop.var + inner) by identity, snapshotting
        # the sites first.  replace_var_everywhere re-traverses the freshly
        # inserted (loop.var + inner) nodes -- which themselves reference
        # loop.var -- so it would expand each substitution without bound.
        for v in collect(old_body,
                         lambda n: isinstance(n, Var) and n.name == loop.var):
            _replace_expr_node(old_body, v, copy.deepcopy(combined))

        guard = RelOp(T.boolean, op=Cmp.LT,
                      lhs=BinOp(T.i32, op=Arith.ADD,
                                lhs=Var(T.i32, name=loop.var),
                                rhs=Var(T.i32, name=inner)),
                      rhs=copy.deepcopy(loop.end))
        inner_loop = For(var=inner, start=_int(0), end=_int(k), step=_int(1),
                         body=Block([If(cond=guard, then=old_body)]))

        unroll = rng.choice([None, 2, 4, k])
        if unroll is not None:
            loop.body = Block([UnrollHint(factor=unroll), inner_loop])
        else:
            loop.body = Block([inner_loop])

    def perturb_opt(self, program, rng, param=None) -> Program:
        return self.perturb(program, rng, param)

    def perturb_deopt(self, program, rng, param=None) -> Program:
        """Work thinning: per-iteration overhead that wastes cycles."""
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None:
            return prog
        k = param if param is not None else rng.choice(self.sweep_space)
        loops = mutable_for_loops(kernel)
        if not loops:
            return prog
        loop = rng.choice(loops)
        if loop.body is None:
            return prog
        taken = all_var_names(kernel)
        allow_vol = getattr(prog.launch, "allow_volatile", False)
        acc = kx_fresh_name("overhead_acc", taken)
        taken.add(acc)
        loop.body.stmts.insert(0, Declare(
            name=acc, gtype=T.u32, space=MemSpace.REGISTER,
            init=_uint(0),
            quals=[Qual.VOLATILE] if allow_vol else []))
        loop.body.stmts.append(CompoundAssign(
            Var(T.u32, name=acc), Compound.ADD,
            BinOp(T.u32, op=Arith.MUL,
                  lhs=BinOp(T.u32, op=Arith.BXOR,
                            lhs=Cast(T.u32, to=T.u32, arg=_tid("x")),
                            rhs=Cast(T.u32, to=T.u32,
                                     arg=Var(T.i32, name=loop.var))),
                  rhs=_uint(k))))
        # Opaque never-true guard keeps the accumulator observed.
        cond = RelOp(T.boolean, op=Cmp.EQ, lhs=Var(T.u32, name=acc),
                     rhs=_uint(0xDEADBEEF))
        loop.body.stmts.append(If(cond=cond, then=Block([Fence(kx_scope="block")]),
                                  otherwise=None))
        return prog


# ---------------------------------------------------------------------------
# COMP2: FMA decomposition
# ---------------------------------------------------------------------------

class CompFmaDecomposition(Mutation):
    """fma(a,b,c) -> t = a*b; v = t + c (temp widens the re-fusion window)."""

    name = "COMP2_fma_decomposition"
    category = "compute"

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None:
            return prog
        fmas = find_fmas(kernel)
        if not fmas:
            return prog
        target = rng.choice(fmas)
        taken = all_var_names(kernel)
        mul = BinOp(target.ty, op=Arith.MUL,
                    lhs=copy.deepcopy(target.a), rhs=copy.deepcopy(target.b))
        if rng.random() < 0.6:
            tmp = kx_fresh_name("fma_tmp", taken)
            tmp_decl = Declare(name=tmp, gtype=target.ty, init=mul)
            add = BinOp(target.ty, op=Arith.ADD,
                        lhs=Var(target.ty, name=tmp),
                        rhs=copy.deepcopy(target.c))
            # Insert the temp declaration before the enclosing statement.
            stmt = self._stmt_containing(kernel, target)
            parent = _parent_block(kernel, stmt) if stmt is not None else None
            if stmt is not None and parent is not None:
                for i, s in enumerate(parent.stmts):
                    if s is stmt:
                        parent.stmts.insert(i, tmp_decl)
                        break
                _replace_expr_node(kernel, target, add)
            else:
                add2 = BinOp(target.ty, op=Arith.ADD, lhs=mul,
                             rhs=copy.deepcopy(target.c))
                _replace_expr_node(kernel, target, add2)
        else:
            add = BinOp(target.ty, op=Arith.ADD, lhs=mul,
                        rhs=copy.deepcopy(target.c))
            _replace_expr_node(kernel, target, add)
        return prog

    @staticmethod
    def _stmt_containing(root, expr: Expr) -> Optional[Stmt]:
        for node in traverse.traverse(root):
            if isinstance(node, Stmt) and node is not expr:
                if any(n is expr for n in traverse.traverse(node)):
                    # shallowest stmt that contains expr
                    if isinstance(node, (Assign, Declare, ExprStmt,
                                         CompoundAssign)):
                        return node
        return None

    def perturb_opt(self, program: Program, rng: random.Random, param=None) -> Program:
        """Opt direction: fuse a*b+c back into an FMA.

        Finds BinOp(ADD, BinOp(MUL, a, b), c) (either operand order) and
        replaces the addition with Fma(a, b, c).  The emitter degrades
        non-float FMAs back to a plain multiply-add, so integer patterns
        are harmless."""
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None:
            return prog
        cands = []
        for node in traverse.traverse(kernel):
            if not (isinstance(node, BinOp) and node.op is Arith.ADD):
                continue
            if isinstance(node.lhs, BinOp) and node.lhs.op is Arith.MUL:
                cands.append((node, node.lhs.lhs, node.lhs.rhs, node.rhs))
            elif isinstance(node.rhs, BinOp) and node.rhs.op is Arith.MUL:
                cands.append((node, node.rhs.lhs, node.rhs.rhs, node.lhs))
        if not cands:
            return prog
        target, a, b, c = rng.choice(cands)
        fma = Fma(target.ty, a=copy.deepcopy(a), b=copy.deepcopy(b),
                  c=copy.deepcopy(c))
        _replace_expr_node(kernel, target, fma)
        return prog

    def perturb_deopt(self, program: Program, rng: random.Random, param=None) -> Program:
        """Deopt direction: decompose FMA (base perturb)."""
        return self.perturb(program, rng, param)


# ---------------------------------------------------------------------------
# COMP3: strength-reduction reversal
# ---------------------------------------------------------------------------

class CompStrengthReductionReversal(Mutation):
    """x>>k -> x/2^k ; x<<k -> x*2^k ; x&(N-1) -> x%N (N power of two)."""

    name = "COMP3_strength_reduction_reversal"
    category = "compute"

    def _categorize(self, b: BinOp) -> Optional[str]:
        if not isinstance(b.rhs, Lit) or not isinstance(b.rhs.value, int):
            return None
        v = b.rhs.value
        if b.op is Arith.SHR and 0 < v < 31:
            return "rshift"
        if b.op is Arith.SHL and 0 < v < 31:
            return "lshift"
        if b.op is Arith.BAND and v > 0 and is_pow2(v + 1):
            return "and_mask"
        return None

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None:
            return prog
        cands = []
        for b in collect(kernel, lambda n: isinstance(n, BinOp)):
            kind = self._categorize(b)
            if kind:
                cands.append((b, kind))
        if not cands:
            return prog
        target, kind = rng.choice(cands)
        k = target.rhs.value
        if kind == "rshift":
            repl = BinOp(target.ty, op=Arith.DIV,
                         lhs=copy.deepcopy(target.lhs), rhs=_int(1 << k))
        elif kind == "lshift":
            repl = BinOp(target.ty, op=Arith.MUL,
                         lhs=copy.deepcopy(target.lhs), rhs=_int(1 << k))
        else:
            repl = BinOp(target.ty, op=Arith.MOD,
                         lhs=copy.deepcopy(target.lhs), rhs=_int(k + 1))
        _replace_expr_node(kernel, target, repl)
        return prog

    def perturb_opt(self, program, rng, param=None) -> Program:
        """Opt direction: reverse -- x / 2^k -> x >> k, x * 2^k -> x << k,
        x % N -> x & (N-1)."""
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None:
            return prog
        cands = []
        for b in collect(kernel, lambda n: isinstance(n, BinOp)):
            if not isinstance(b.rhs, Lit) or not isinstance(b.rhs.value, int):
                continue
            v = b.rhs.value
            if b.op is Arith.DIV and is_pow2(v):
                cands.append((b, Arith.SHR, log2i(v)))
            elif b.op is Arith.MUL and is_pow2(v):
                cands.append((b, Arith.SHL, log2i(v)))
            elif b.op is Arith.MOD and is_pow2(v):
                cands.append((b, Arith.BAND, v - 1))
        if not cands:
            return prog
        target, op, val = rng.choice(cands)
        repl = BinOp(target.ty, op=op,
                     lhs=copy.deepcopy(target.lhs), rhs=_int(val))
        _replace_expr_node(kernel, target, repl)
        return prog

    def perturb_deopt(self, program, rng, param=None) -> Program:
        return self.perturb(program, rng, param)


# ---------------------------------------------------------------------------
# COMP4: trip-count opacification
# ---------------------------------------------------------------------------

class CompTripCountOpacify(Mutation):
    """Replace a literal loop bound with an opaque-but-equal expression
    ((N/2)*2 or N|0) so the compiler must evaluate it dynamically."""

    name = "COMP4_trip_count_opacify"
    category = "compute"

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None or kernel.body is None:
            return prog
        loops = mutable_for_loops(kernel)
        if not loops:
            return prog
        loop = rng.choice(loops)
        if isinstance(loop.end, Lit) and isinstance(loop.end.value, int):
            v = loop.end.value
            if rng.random() < 0.5:
                # (v//2)*2 -- opaque to constant-folding; == v for even v,
                # == v-1 for odd v.  Floor (not ceil) so the rewritten bound
                # never exceeds the original and cannot drive an extra,
                # out-of-bounds iteration.
                loop.end = BinOp(T.i32, op=Arith.MUL,
                                 lhs=_int(v // 2), rhs=_int(2))
            else:
                loop.end = BinOp(T.i32, op=Arith.BOR,
                                 lhs=copy.deepcopy(loop.end), rhs=_int(0))
        else:
            loop.end = BinOp(T.i32, op=Arith.ADD,
                             lhs=copy.deepcopy(loop.end), rhs=_int(0))
        return prog

    def perturb_opt(self, program, rng, param=None) -> Program:
        """De-opacify: fold (N/2)*2 / N|0 / N+0 shapes back to literals."""
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None or kernel.body is None:
            return prog
        for loop in mutable_for_loops(kernel):
            v = self._extract_literal(loop.end)
            if v is not None:
                loop.end = Lit(T.i32, value=v)
                return prog
        return prog

    @staticmethod
    def _extract_literal(e: Expr) -> Optional[int]:
        if isinstance(e, Lit) and isinstance(e.value, int):
            return e.value
        if isinstance(e, BinOp) and isinstance(e.rhs, Lit):
            if e.op is Arith.BOR and e.rhs.value == 0:
                return CompTripCountOpacify._extract_literal(e.lhs)
            if e.op is Arith.ADD and e.rhs.value == 0:
                return CompTripCountOpacify._extract_literal(e.lhs)
            if (e.op is Arith.MUL and isinstance(e.lhs, Lit)
                    and e.rhs.value == 2):
                return e.lhs.value * 2
        return None

    def perturb_deopt(self, program, rng, param=None) -> Program:
        return self.perturb(program, rng, param)


# ---------------------------------------------------------------------------
# COMP5: loop fission
# ---------------------------------------------------------------------------

class CompLoopFission(Mutation):
    """Split one loop body into N consecutive loops over the same range."""

    name = "COMP5_loop_fission"
    category = "compute"
    sweep_space = [2, 3, 4]

    @staticmethod
    def _safe_split_points(stmts: List[Stmt]) -> List[int]:
        """Split indices where no later chunk reads an earlier chunk's def.

        A def is either a declaration or an assignment in the earlier chunk;
        fission moves the later chunk into its own loop, where a flow-carried
        variable would hold only the last iteration's value instead of the
        per-iteration value the original computed."""
        safe: List[int] = []
        used_later: Set[str] = set()
        # compute used names per position (names referenced but not declared here)
        for i in range(1, len(stmts)):
            used_later = set()
            for s in stmts[i:]:
                for v in collect(s, lambda n: isinstance(n, Var)):
                    used_later.add(v.name)
            defined_here = {s.name for s in stmts[:i] if isinstance(s, Declare)}
            # Include variables ASSIGNED in the earlier chunk (declared outside
            # the loop): reading them later is a flow dependence fission breaks.
            defined_here |= CompLoopFission._written_vars(Block(list(stmts[:i])))
            # Reverse loop-carried dependence: a variable written in the LATER
            # chunk and read in the EARLIER chunk is loop-carried in the
            # original (the earlier chunk reads the previous iteration's
            # later-chunk write).  Fission moves the write into its own loop,
            # so the earlier chunk would read the initial value every iteration
            # instead -- the reverse-dependence break the fusion guard checks
            # in the opposite direction.
            written_later = CompLoopFission._written_vars(Block(list(stmts[i:])))
            read_earlier = CompLoopFission._read_vars(Block(list(stmts[:i])))
            if not (defined_here & used_later) and not (written_later & read_earlier):
                safe.append(i)
        return safe

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None or kernel.body is None:
            return prog
        n_parts = param if param is not None else rng.choice(self.sweep_space)
        loops = [l for l in mutable_for_loops(kernel)
                 if l.body and len(l.body.stmts) >= n_parts]
        if not loops:
            return prog
        loop = rng.choice(loops)
        parent = _parent_block(kernel, loop)
        if parent is None:
            return prog
        stmts = loop.body.stmts
        safe = self._safe_split_points(stmts)
        if not safe:
            return prog
        n_splits = min(n_parts - 1, len(safe))
        chosen = sorted(rng.sample(safe, n_splits))
        bounds = [0] + chosen + [len(stmts)]
        new_loops = []
        for k in range(len(bounds) - 1):
            chunk = copy.deepcopy(stmts[bounds[k]:bounds[k + 1]])
            new_loops.append(For(var=loop.var,
                                 start=copy.deepcopy(loop.start),
                                 end=copy.deepcopy(loop.end),
                                 step=copy.deepcopy(loop.step),
                                 body=Block(chunk)))
        for i, s in enumerate(parent.stmts):
            if s is loop:
                parent.stmts[i:i + 1] = new_loops
                break
        return prog

    def perturb_opt(self, program, rng, param=None) -> Program:
        """Loop fusion: merge two consecutive loops over the same range.

        Only fuses when loop2 does not read any variable written by loop1
        (conservative dependence guard) -- otherwise the merge would move
        those reads ahead of loop1's remaining iterations."""
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None:
            return prog
        candidates = []
        for block in blocks(kernel):
            for i in range(len(block.stmts) - 1):
                a, b = block.stmts[i], block.stmts[i + 1]
                if not (isinstance(a, For) and isinstance(b, For)):
                    continue
                if not (self._same_expr(a.start, b.start)
                        and self._same_expr(a.end, b.end)
                        and self._same_expr(a.step, b.step)):
                    continue
                written = self._written_vars(a.body) - {a.var}
                read = self._read_vars(b.body) - {b.var}
                # Both flow directions must stay clean.  In the original, loop
                # a runs entirely before loop b, so a never observes b's writes;
                # a reverse dependence (b writes, a reads) is equally broken by
                # fusion -- later a-iterations would see b's per-iteration writes.
                if (written & read) or (
                    (self._written_vars(b.body) - {b.var})
                    & (self._read_vars(a.body) - {a.var})):
                    continue
                candidates.append((block, i, a, b))
        if not candidates:
            return prog
        block, i, a, b = rng.choice(candidates)
        body_b = copy.deepcopy(b.body)
        if a.var != b.var:
            replace_var_everywhere(body_b, b.var,
                                   lambda: Var(T.i32, name=a.var))
        merged = For(var=a.var, start=copy.deepcopy(a.start),
                     end=copy.deepcopy(a.end), step=copy.deepcopy(a.step),
                     body=Block(list(a.body.stmts) + list(body_b.stmts)))
        block.stmts[i:i + 2] = [merged]
        return prog

    @staticmethod
    def _written_vars(node) -> Set[str]:
        names: Set[str] = set()
        for a in assignments(node):
            t = a.target
            if isinstance(t, Var):
                names.add(t.name)
            elif isinstance(t, Subscript) and isinstance(t.array, Var):
                names.add(t.array.name)
        for ca in compound_assignments(node):
            t = ca.target
            if isinstance(t, Var):
                names.add(t.name)
            elif isinstance(t, Subscript) and isinstance(t.array, Var):
                names.add(t.array.name)
        return names

    @staticmethod
    def _read_vars(node) -> Set[str]:
        return {v.name for v in collect(node, lambda n: isinstance(n, Var))}

    @staticmethod
    def _same_expr(a: Expr, b: Expr) -> bool:
        if isinstance(a, Lit) and isinstance(b, Lit):
            return a.value == b.value
        if isinstance(a, Var) and isinstance(b, Var):
            return a.name == b.name
        if isinstance(a, BinOp) and isinstance(b, BinOp):
            return (a.op is b.op
                    and CompLoopFission._same_expr(a.lhs, b.lhs)
                    and CompLoopFission._same_expr(a.rhs, b.rhs))
        return False

    def perturb_deopt(self, program, rng, param=None) -> Program:
        return self.perturb(program, rng, param)


# ---------------------------------------------------------------------------
# COMP6: inline switch (helper forceinline / noinline flip)
# ---------------------------------------------------------------------------

class CompInlineSwitch(Mutation):
    """Extract 1-5 kernel statements into a new __device__ helper function
    plus a call site (perffuzzer COMP6); falls back to flipping a helper's
    inline hint when extraction is impossible."""

    name = "COMP6_inline_switch"
    category = "compute"

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        hint = param if param in ("__forceinline__", "__noinline__") \
            else "__forceinline__"
        prog, ok = self._extract(program, rng, hint)
        if ok:
            return prog
        return self._flip(program, rng, param)

    def perturb_opt(self, program, rng, param=None) -> Program:
        """Opt direction: extract with __forceinline__."""
        prog, ok = self._extract(program, rng, "__forceinline__")
        if ok:
            return prog
        return self._flip(program, rng, "__forceinline__")

    def perturb_deopt(self, program, rng, param=None) -> Program:
        """Deopt direction: extract with __noinline__ (prevents inlining)."""
        prog, ok = self._extract(program, rng, "__noinline__")
        if ok:
            return prog
        return self._flip(program, rng, "__noinline__")

    # -- extraction -----------------------------------------------------

    def _extract(self, program: Program, rng: random.Random,
                 hint: str) -> Tuple[Program, bool]:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None or kernel.body is None:
            return prog, False
        stmts = kernel.body.stmts
        if len(stmts) < 3:
            return prog, False

        start = rng.randint(0, len(stmts) - 2)
        end = rng.randint(start + 1, min(start + 5, len(stmts)))
        extracted = stmts[start:end]

        has_work = any(isinstance(s, (Assign, CompoundAssign, ExprStmt, For))
                       for s in extracted)
        if not has_work:
            return prog, False

        taken = all_var_names(prog)
        fn_name = kx_fresh_name("device_helper", taken)

        used = set()
        for s in extracted:
            for v in traverse.collect(s, lambda n: isinstance(n, Var)):
                used.add(v.name)
        # Any name DECLARED within the extracted range is a helper-local and
        # must not become a parameter: top-level Declares, Declares nested in
        # loop/if bodies (e.g. ExtPrefetchDistance's `pf`, ExtWarpCentric's
        # `wc_ballot`), and for-loop header vars.  A nested decl missed here
        # would become a call-site argument referencing a name that moved into
        # the helper -> undefined identifier (uncompilable mutant).
        declared_in_extract = {
            d.name for s in extracted
            for d in traverse.collect(s, lambda n: isinstance(n, Declare))
        }
        for s in extracted:
            for f in traverse.collect(s, lambda n: isinstance(n, For)):
                declared_in_extract.add(f.var)
        param_vars = used - declared_in_extract

        # Array-indexed vars cannot be passed as scalar params: bail.
        array_indexed = set()
        for s in extracted:
            for sub in traverse.collect(s, lambda n: isinstance(n, Subscript)
                                    and isinstance(n.array, Var)):
                array_indexed.add(sub.array.name)
        if array_indexed & param_vars:
            return prog, False

        # Bail if anything declared inside the extracted range is used
        # after it -- the helper is void and cannot return the value.
        later_used = set()
        for s in stmts[end:]:
            for v in traverse.collect(s, lambda n: isinstance(n, Var)):
                later_used.add(v.name)
        if later_used & declared_in_extract:
            return prog, False

        var_type = {d.name: d.gtype for d in declarations(kernel)}
        for kp in kernel.params or []:
            var_type.setdefault(kp.name, kp.gtype)

        params = []
        for pname in sorted(param_vars):
            ptype = var_type.get(pname)
            if ptype is None:
                return prog, False
            params.append(Declare(name=pname, gtype=ptype))

        new_fn = Func(name=fn_name, params=params, ret=T.void,
                      body=Block(copy.deepcopy(extracted)),
                      quals=[Qual.DEVICE], inline_hint=hint)
        call = ExprStmt(Call(T.void, fn=fn_name,
                             args=[Var(p.gtype, name=p.name) for p in params]))
        stmts[start:end] = [call]
        for i, d in enumerate(prog.decls):
            if d is kernel:
                prog.decls.insert(i, new_fn)
                break
        return prog, True

    # -- fallback: inline-hint flip --------------------------------------

    def _flip(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        helpers = device_functions(prog)
        if not helpers:
            return prog
        h = rng.choice(helpers)
        h.inline_hint = (rng.choice(["__forceinline__", "__noinline__"])
                         if param is None else param)
        return prog


# ---------------------------------------------------------------------------
# EXT: unroll factor
# ---------------------------------------------------------------------------



class ExtUnrollFactor(Mutation):
    """Add or change #pragma unroll before a for-loop."""

    name = "EXT_unroll_factor"
    category = "compute"
    sweep_space = [1, 2, 4, 8]

    @staticmethod
    def _kx_apply(kernel: Func, factor, emit_disable: bool = False) -> None:
        """Strip existing unroll hints; add UnrollHint(factor) before every
        for-loop.  factor=None -> '#pragma unroll' (full); factor=1 only
        emits when emit_disable (deopt: explicit '#pragma unroll 1')."""
        for block in blocks(kernel):
            new_stmts = []
            for stmt in block.stmts:
                if isinstance(stmt, UnrollHint):
                    continue
                if isinstance(stmt, For):
                    if factor is None or factor > 1 or \
                            (emit_disable and factor == 1):
                        new_stmts.append(UnrollHint(factor=factor))
                new_stmts.append(stmt)
            block.stmts = new_stmts

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None or kernel.body is None:
            return prog
        if not mutable_for_loops(kernel):
            return prog
        factor = param if param is not None else rng.choice(self.sweep_space)
        self._kx_apply(kernel, factor)
        return prog

    def perturb_opt(self, program, rng, param=None) -> Program:
        """Opt direction: aggressive unroll -- 4, 8, or full (None)."""
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None or kernel.body is None:
            return prog
        if not mutable_for_loops(kernel):
            return prog
        factor = param if param is not None else rng.choice([4, 8, None])
        self._kx_apply(kernel, factor)
        return prog

    def perturb_deopt(self, program, rng, param=None) -> Program:
        """Deopt direction: '#pragma unroll 1' on every loop (disables
        unrolling -- more loop overhead, less ILP)."""
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None or kernel.body is None:
            return prog
        if not mutable_for_loops(kernel):
            return prog
        self._kx_apply(kernel, 1, emit_disable=True)
        return prog


# ---------------------------------------------------------------------------
# EXT: block size scan
# ---------------------------------------------------------------------------

class ExtBlockSizeSweep(Mutation):
    """Change the launch block size, rescaling the grid to keep coverage."""

    name = "EXT_block_size_sweep"
    category = "compute"
    sweep_space = [64, 128, 256, 512, 1024]

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        if prog.launch.kx_is_2d:
            return prog
        # Block-size rescaling is unsafe when the kernel uses shared memory:
        # the per-block shared slots are sized for the original block, and a
        # new block size can alias or under-fill them.
        if prog.launch.shared_bytes > 0:
            return prog
        kernel = kernel_of(prog)
        if kernel is not None and any(d.space is MemSpace.SHARED
                                      for d in declarations(kernel)):
            return prog
        bx = param if param is not None else rng.choice(self.sweep_space)
        launch = prog.launch
        total = launch.threads_per_block * launch.block_count
        new_bx = max(32, int(bx))
        new_gx = max(1, -(-total // new_bx))
        launch.block = (new_bx, 1, 1)
        launch.grid = (new_gx, launch.grid[1], launch.grid[2])
        if kernel is not None and kernel.launch_bounds:
            lb = kernel.launch_bounds
            if lb[0] < new_bx:
                kernel.launch_bounds = (new_bx,) + tuple(lb[1:])
        return prog

    def perturb_opt(self, program, rng, param=None) -> Program:
        """Opt direction: hardware-friendly block size (256 or 512)."""
        bx = param if param is not None else rng.choice([256, 512])
        return self.perturb(program, rng, bx)

    def perturb_deopt(self, program, rng, param=None) -> Program:
        """Deopt direction: warp-underutilizing (64) or register-pressured
        (1024) block size."""
        bx = param if param is not None else rng.choice([64, 1024])
        return self.perturb(program, rng, bx)


# ---------------------------------------------------------------------------
# EXT: register pressure inflation
# ---------------------------------------------------------------------------

class ExtRegisterPressureInflation(Mutation):
    """Declare N live locals with cross-mixed values (spill pressure)."""

    name = "EXT_register_pressure_inflation"
    category = "compute"
    sweep_space = [8, 16, 32, 64]

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None or kernel.body is None:
            return prog
        n = param if param is not None else rng.choice(self.sweep_space)
        taken = all_var_names(kernel)
        allow_vol = getattr(prog.launch, "allow_volatile", False)
        names: List[str] = []
        decls: List[Stmt] = []
        for i in range(n):
            vn = kx_fresh_name("rp_inflate", taken)
            taken.add(vn)
            names.append(vn)
            init = BinOp(T.u32, op=Arith.BXOR,
                         lhs=Cast(T.u32, to=T.u32, arg=_tid("x")),
                         rhs=_uint((0x9E3779B9 * (i + 1)) & 0xFFFFFFFF))
            decls.append(Declare(name=vn, gtype=T.u32, init=init,
                                 quals=[Qual.VOLATILE] if allow_vol else []))
        # Cross-mix so every var depends on others (longer live ranges).
        mix: List[Stmt] = []
        for i, vn in enumerate(names):
            j = (i + 1) % n
            k = (i + 7) % n
            mix.append(Assign(Var(T.u32, name=vn),
                              BinOp(T.u32, op=Arith.BXOR,
                                    lhs=Var(T.u32, name=vn),
                                    rhs=BinOp(T.u32, op=Arith.ADD,
                                              lhs=Var(T.u32, name=names[j]),
                                              rhs=Var(T.u32, name=names[k])))))
        kernel.body.stmts = decls + mix + list(kernel.body.stmts)
        folded: Expr = Var(T.u32, name=names[0])
        for vn in names[1:]:
            folded = BinOp(T.u32, op=Arith.BXOR, lhs=folded,
                           rhs=Var(T.u32, name=vn))
        keep_value_live(kernel, folded, base="rp_inflate",
                        allow_volatile=allow_vol)
        return prog


# ---------------------------------------------------------------------------
# EXT: launch bounds scan
# ---------------------------------------------------------------------------

class ExtLaunchBoundsSweep(Mutation):
    """Sweep __launch_bounds__ (max_threads, min_blocks) hints."""

    name = "EXT_launch_bounds_sweep"
    category = "compute"
    sweep_space = [(64, 2), (128, 2), (256, 1), (256, 2), (512, 1)]

    @staticmethod
    def _kx_apply(prog: Program, kernel: Func, lb) -> None:
        # gpuperf clamp-up: never set max_threads below the actual block.
        max_threads = max(int(lb[0]), prog.launch.threads_per_block)
        kernel.launch_bounds = (max_threads,) + tuple(lb[1:])
        # __launch_bounds__ and __maxnreg__ are mutually exclusive in CUDA.
        kernel.maxnreg = None

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None:
            return prog
        lb = tuple(param) if param is not None else rng.choice(self.sweep_space)
        self._kx_apply(prog, kernel, lb)
        return prog

    def perturb_opt(self, program, rng, param=None) -> Program:
        """Opt direction: tighter launch bounds / lower register budgets."""
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None:
            return prog
        lb = tuple(param) if param is not None \
            else rng.choice([(128, 2), (256, 2), (512, 1)])
        self._kx_apply(prog, kernel, lb)
        return prog

    def perturb_deopt(self, program, rng, param=None) -> Program:
        """Deopt direction: drop launch bounds, allow high register use."""
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None:
            return prog
        kernel.launch_bounds = None
        kernel.maxnreg = None
        return prog


# ---------------------------------------------------------------------------
# EXT: CUDA kernel attribute scan
# ---------------------------------------------------------------------------

class ExtCudaKernelAttrSweep(Mutation):
    """Sweep launch_bounds + cluster dims (sm_90+) launch metadata."""

    name = "EXT_cuda_kernel_attr_sweep"
    category = "compute"
    sweep_space = [
        (128, 2, None),
        (256, 2, None),
        (512, 1, None),
        (256, 2, (2, 1, 1)),
        (512, 1, (2, 1, 1)),
    ]

    def _kx_valid_choices(self, prog: Program) -> List[tuple]:
        actual = prog.launch.threads_per_block
        sm = cuda_sm_version(prog)
        grid = prog.launch.grid
        choices = []
        for max_threads, min_blocks, cluster in self.sweep_space:
            if max_threads < actual:
                continue
            if cluster is not None:
                if sm < 90 or actual > 512:
                    continue
                cx, cy, cz = cluster
                if grid[0] % cx or grid[1] % cy or grid[2] % cz:
                    continue
            choices.append((max_threads, min_blocks, cluster))
        if not choices:
            choices.append((max(actual, 256), 1, None))
        return choices

    @staticmethod
    def _kx_apply(kernel: Func, choice) -> None:
        max_threads, min_blocks, cluster = choice
        kernel.launch_bounds = (max_threads, min_blocks)
        kernel.maxnreg = None
        kernel.cluster_dims = cluster

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        if not kx_is_cuda(prog, 70):
            return prog
        kernel = kernel_of(prog)
        if kernel is None:
            return prog
        choices = self._kx_valid_choices(prog)
        choice = tuple(param) if param is not None else rng.choice(choices)
        if choice not in choices:
            choice = rng.choice(choices)
        self._kx_apply(kernel, choice)
        return prog

    def perturb_opt(self, program, rng, param=None) -> Program:
        """Opt direction: occupancy-friendly metadata (min_blocks>=2, no
        cluster shaping)."""
        prog = clone_program(program)
        if not kx_is_cuda(prog, 70):
            return prog
        kernel = kernel_of(prog)
        if kernel is None:
            return prog
        choices = self._kx_valid_choices(prog)
        opt = [c for c in choices if c[1] >= 2 and c[2] is None]
        self._kx_apply(kernel, opt[0] if opt else choices[0])
        return prog

    def perturb_deopt(self, program, rng, param=None) -> Program:
        """Deopt direction: relax occupancy hints, drop cluster shaping."""
        prog = clone_program(program)
        if not kx_is_cuda(prog, 70):
            return prog
        kernel = kernel_of(prog)
        if kernel is None:
            return prog
        actual = prog.launch.threads_per_block
        kernel.launch_bounds = (max(actual, 512), 1)
        kernel.maxnreg = None
        kernel.cluster_dims = None
        return prog


# ---------------------------------------------------------------------------
# EXT: CUDA DPX rewrite (sm_90)
# ---------------------------------------------------------------------------

class ExtCudaDpxRewrite(Mutation):
    """Inject a DPX min/max idiom on Hopper-class targets (sm_90+ only)."""

    name = "EXT_cuda_dpx_rewrite"
    category = "compute"
    sweep_space = [
        "__vimax3_u32", "__vimin3_u32", "__viaddmax_u32", "__viaddmin_u32",
        "__vimax3_s32", "__vimin3_s32", "__viaddmax_s32", "__viaddmin_s32",
    ]

    @staticmethod
    def _kx_build_call(fn: str, rng: random.Random) -> Call:
        if "_u32" in fn:
            ty = T.u32

            def tid_v():
                return Var(T.u32, name="tid")

            def lit(v):
                return _uint(v)
        else:
            ty = T.i32

            def tid_v():
                return Cast(T.i32, to=T.i32, arg=Var(T.u32, name="tid"))

            def lit(v):
                return _int(v)
        a = BinOp(ty, op=Arith.BXOR, lhs=tid_v(),
                  rhs=lit(rng.choice([3, 7, 15, 31])))
        b = BinOp(ty, op=Arith.ADD, lhs=tid_v(),
                  rhs=lit(rng.choice([1, 5, 9, 17])))
        c = BinOp(ty, op=Arith.SUB, lhs=tid_v(),
                  rhs=lit(rng.choice([0, 1, 2, 4])))
        return Call(ty, fn=fn, args=[a, b, c], result_ty=ty)

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        if prog.launch.kx_is_2d or not kx_is_cuda(prog, 90):
            return prog
        kernel = kernel_of(prog)
        if kernel is None or kernel.body is None:
            return prog
        fn = str(param) if param is not None else rng.choice(self.sweep_space)
        call = self._kx_build_call(fn, rng)
        # Fold through the standard live-value sink (inserted before the
        # output region): real DPX work, checksum untouched, deterministic.
        keep_value_live(kernel, call, base="cuda_dpx",
                        allow_volatile=getattr(prog.launch, "allow_volatile", False))
        return prog

    def perturb_opt(self, program, rng, param=None) -> Program:
        """Opt direction: prefer max-style DPX idioms."""
        fn = param if param is not None \
            else rng.choice(["__vimax3_u32", "__viaddmax_u32"])
        return self.perturb(program, rng, param=fn)

    def perturb_deopt(self, program, rng, param=None) -> Program:
        """Deopt direction: prefer min/reduction-style DPX idioms."""
        fn = param if param is not None \
            else rng.choice(["__vimin3_u32", "__viaddmin_s32"])
        return self.perturb(program, rng, param=fn)
