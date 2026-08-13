"""Divergence mutation operators (DIV1..DIV4 + extensions)."""

from __future__ import annotations

import copy
import random
from typing import List, Optional, Set

from generator_ir.ir.typedecl import (
    GType, T, MemSpace, Qual, TArray, TPointer, TInt, TUInt, TFloat, TBool,
)
from generator_ir.ir.operators import Arith, Cmp, SafeOp, Logic, Compound
from generator_ir.ir.nodes import (
    Expr, Lit, Var, ThreadBuiltin, BinOp, RelOp, LogicOp, Cast, SafeBin,
    Subscript, Call, Atomic, WarpShuffle, WarpVote, Select, AddrOf,
)
from generator_ir.ir.statements import (
    Stmt, Assign, CompoundAssign, ExprStmt, Declare, Block, If, For,
    While, Barrier, Switch, Case, Compound as _Compound,
)
from generator_ir.ir.program import Func, Program
from generator_ir.ir.divergence import DivergenceAnalyzer
from generator_ir.ir.traversal import (
    clone_program, kernel_of, collect, mutable_for_loops, assignments,
    compound_assignments, subscripts, declarations, if_statements, atomics,
    blocks, all_var_names, kx_fresh_name, barrier_ok_in_loop, for_trip_uniform,
    keep_value_live, thread_builtins, switches, walk_parent_block_of,
    replace_var_everywhere,
)
from generator_ir.mutate.foundation import Mutation, cuda_sm_version, kx_is_cuda


def _uint(v: int) -> Lit:
    return Lit(T.u32, value=v)


def _int(v: int) -> Lit:
    return Lit(T.i32, value=v)


def _tid(axis: str = "x") -> ThreadBuiltin:
    return ThreadBuiltin(T.u32, axis=axis, source="threadIdx")


def _has_tid(node) -> bool:
    """True when the subtree references threadIdx (perffuzzer ThreadIdExpr)."""
    return bool(collect(node, lambda n: isinstance(n, ThreadBuiltin)
                        and n.source == "threadIdx"))


def _same_expr(a, b) -> bool:
    """Conservative structural equality for index expressions."""
    if type(a) is not type(b):
        return False
    if isinstance(a, Lit):
        return a.value == b.value
    if isinstance(a, Var):
        return a.name == b.name
    if isinstance(a, ThreadBuiltin):
        return a.source == b.source and a.axis == b.axis
    if isinstance(a, (BinOp, SafeBin)):
        return (a.op is b.op and _same_expr(a.lhs, b.lhs)
                and _same_expr(a.rhs, b.rhs))
    if isinstance(a, Cast):
        return a.to == b.to and _same_expr(a.arg, b.arg)
    if isinstance(a, Subscript):
        return _same_expr(a.array, b.array) and _same_expr(a.index, b.index)
    return False


# ---------------------------------------------------------------------------
# DIV1: divergence source injection
# ---------------------------------------------------------------------------




class DivDivergenceSource(Mutation):
    """Inject genuine thread-dependence: threadIdx into input indices (inside
    the modulo bound), shift amounts, or constant operands.  Semantics-
    changing by design, always bounded and deterministic."""

    name = "DIV1_divergence_source"
    category = "divergence"
    sweep_space = [1, 2, 3, 4]

    def _inject_index(self, acc: Subscript, rng: random.Random) -> None:
        tid = _tid(rng.choice(["x", "y"]))
        tid_mod = BinOp(T.u32, op=Arith.MOD, lhs=tid, rhs=_uint(32))
        idx = acc.index
        if isinstance(idx, SafeBin) and idx.op is SafeOp.SMOD:
            # inject INSIDE the modulus so the index stays bounded
            idx.lhs = BinOp(T.u32, op=Arith.ADD,
                            lhs=copy.deepcopy(idx.lhs), rhs=tid_mod)
        else:
            # Wrap against n: a bare index (tid or a loop counter) plus
            # tid%32 could otherwise read past the n-element input buffer
            # (out-of-bounds, nondeterministic) -- the docstring's "always
            # bounded" guarantee that the SafeBin branch already honors.
            n_u = Cast(T.u32, to=T.u32, arg=Var(T.i32, name="n"))
            acc.index = SafeBin(T.u32, op=SafeOp.SMOD,
                                lhs=BinOp(T.u32, op=Arith.ADD,
                                          lhs=copy.deepcopy(idx), rhs=tid_mod),
                                rhs=n_u)

    def _inject_arith(self, node: BinOp, rng: random.Random) -> None:
        tid = _tid(rng.choice(["x", "y"]))
        strategy = rng.randint(0, 2)
        right_fp = getattr(node.rhs.ty, "is_fp", False) if node.rhs is not None else False
        if strategy == 0 and node.op in (Arith.SHR, Arith.SHL):
            node.rhs = BinOp(T.u32, op=Arith.BAND, lhs=tid, rhs=_uint(3))
        elif strategy == 1 and isinstance(node.rhs, Lit) and not right_fp:
            old = copy.deepcopy(node.rhs)
            node.rhs = BinOp(node.rhs.ty, op=Arith.BXOR, lhs=old,
                             rhs=BinOp(T.u32, op=Arith.BAND, lhs=tid, rhs=_uint(3)))
        elif not right_fp:
            old = copy.deepcopy(node.rhs)
            node.rhs = BinOp(old.ty, op=Arith.BOR, lhs=old,
                             rhs=BinOp(T.u32, op=Arith.BAND, lhs=tid, rhs=_uint(1)))
        else:
            # Float-safe: add a small threadIdx-dependent offset.
            old = copy.deepcopy(node.rhs)
            fty = old.ty if old.ty is not None else T.f32
            tid_mod = BinOp(T.u32, op=Arith.MOD, lhs=tid, rhs=_uint(4))
            node.rhs = BinOp(old.ty, op=Arith.ADD, lhs=old,
                             rhs=Cast(fty, to=fty, arg=tid_mod))

    def _inject_assign(self, assign: Assign, rng: random.Random) -> None:
        """Make an assignment value thread-dependent: a = expr + tid % K."""
        tid = _tid(rng.choice(["x", "y"]))
        k = rng.choice([2, 4, 8, 16, 32])
        tid_mod = BinOp(T.u32, op=Arith.MOD, lhs=tid, rhs=_uint(k))
        old = copy.deepcopy(assign.value)
        if old.ty is not None and getattr(old.ty, "is_int", False):
            assign.value = BinOp(old.ty, op=Arith.ADD, lhs=old, rhs=tid_mod)
        else:
            fty = old.ty if old.ty is not None else T.f32
            assign.value = BinOp(fty, op=Arith.ADD, lhs=old,
                                 rhs=Cast(fty, to=fty, arg=tid_mod))

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None:
            return prog
        count = param if param is not None else rng.choice(self.sweep_space)
        injected = 0

        # Only bounded read-only input reads may be reindexed.
        uniform_reads = [
            a for a in subscripts(kernel)
            if isinstance(a.array, Var) and a.array.name == "input"
            and not collect(a.index, lambda n: isinstance(n, ThreadBuiltin))]
        if uniform_reads and injected < count:
            for acc in rng.sample(uniform_reads,
                                  min(count - injected, len(uniform_reads))):
                self._inject_index(acc, rng)
                injected += 1

        if injected < count:
            # BinOps inside a non-`input` Subscript index must be excluded:
            # making an output[] or shared[] write index divergent causes
            # OOB / write races.
            forbidden: Set[int] = set()
            for sub in subscripts(kernel):
                if (isinstance(sub.array, Var)
                        and sub.array.name != "input"):
                    forbidden.update(id(b) for b in collect(
                        sub.index, lambda n: isinstance(n, BinOp)))
            arith_sites = [
                n for n in collect(kernel, lambda x: isinstance(x, BinOp))
                if n.op in (Arith.ADD, Arith.SUB, Arith.MUL, Arith.SHL,
                            Arith.SHR, Arith.BAND, Arith.BOR, Arith.BXOR)
                and id(n) not in forbidden
                and not _has_tid(n)]
            rng.shuffle(arith_sites)
            for node in arith_sites:
                if injected >= count:
                    break
                self._inject_arith(node, rng)
                injected += 1

        # Strategy 3: inject divergence into assignment values.
        if injected < count:
            uniform_assigns = [a for a in assignments(kernel)
                               if not _has_tid(a.value)]
            if uniform_assigns:
                for a in rng.sample(uniform_assigns,
                                    min(count - injected,
                                        len(uniform_assigns))):
                    self._inject_assign(a, rng)
                    injected += 1
        return prog

    # -- divergence removal (opt direction) ------------------------------

    @staticmethod
    def _simplify_index(acc: Subscript) -> bool:
        """i + threadIdx.x % 32 -> i (either operand order)."""
        idx = acc.index
        if isinstance(idx, BinOp) and idx.op is Arith.ADD:
            if _has_tid(idx.rhs) and not _has_tid(idx.lhs):
                acc.index = copy.deepcopy(idx.lhs)
                return True
            if _has_tid(idx.lhs) and not _has_tid(idx.rhs):
                acc.index = copy.deepcopy(idx.rhs)
                return True
        return False

    @staticmethod
    def _simplify_arith(node: BinOp) -> bool:
        """b >> (tid & 3) -> b >> 0; x * (y | (tid & 1)) -> x * y; ditto ^."""
        if not _has_tid(node.rhs):
            return False
        if node.op in (Arith.SHR, Arith.SHL):
            node.rhs = _uint(0)
            return True
        if isinstance(node.rhs, BinOp) and node.rhs.op in (Arith.BOR,
                                                           Arith.BXOR):
            inner = node.rhs
            if _has_tid(inner.rhs) and not _has_tid(inner.lhs):
                node.rhs = copy.deepcopy(inner.lhs)
                return True
            if _has_tid(inner.lhs) and not _has_tid(inner.rhs):
                node.rhs = copy.deepcopy(inner.rhs)
                return True
        return False

    @staticmethod
    def _simplify_assign(assign: Assign) -> bool:
        """a = expr + tid % K -> a = expr (either operand order)."""
        val = assign.value
        if isinstance(val, BinOp) and val.op is Arith.ADD:
            if _has_tid(val.rhs) and not _has_tid(val.lhs):
                assign.value = copy.deepcopy(val.lhs)
                return True
            if _has_tid(val.lhs) and not _has_tid(val.rhs):
                assign.value = copy.deepcopy(val.rhs)
                return True
        return False

    def perturb_opt(self, program, rng, param=None) -> Program:
        """Opt direction: REMOVE existing divergence -- simplify threadIdx-
        dependent indices, shift amounts, OR/XOR operands and assignment
        addends back to uniform equivalents."""
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None:
            return prog
        count = param if param is not None else rng.choice(self.sweep_space)
        simplified = 0

        divergent_reads = [a for a in subscripts(kernel) if _has_tid(a.index)]
        if divergent_reads and simplified < count:
            for acc in rng.sample(divergent_reads,
                                  min(count - simplified,
                                      len(divergent_reads))):
                if self._simplify_index(acc):
                    simplified += 1

        if simplified < count:
            divergent_arith = [n for n in collect(
                kernel, lambda x: isinstance(x, BinOp)) if _has_tid(n)]
            if divergent_arith:
                for node in rng.sample(divergent_arith,
                                       min(count - simplified,
                                           len(divergent_arith))):
                    if self._simplify_arith(node):
                        simplified += 1

        if simplified < count:
            divergent_assigns = [a for a in assignments(kernel)
                                 if _has_tid(a.value)]
            if divergent_assigns:
                for a in rng.sample(divergent_assigns,
                                    min(count - simplified,
                                        len(divergent_assigns))):
                    if self._simplify_assign(a):
                        simplified += 1
        return prog

    def perturb_deopt(self, program, rng, param=None) -> Program:
        return self.perturb(program, rng, param)


# ---------------------------------------------------------------------------
# DIV2: divergence propagation
# ---------------------------------------------------------------------------

class DivDivergencePropagation(Mutation):
    """Propagate a kernel-kx_scope divergent variable into another computation
    (dataflow) or a branch predicate (control flow)."""

    name = "DIV2_divergence_propagation"
    category = "divergence"
    sweep_space = [1, 2, 3]

    def _kernel_scope_divergent(self, kernel: Func) -> List[str]:
        """Top-level kernel-body declarations that the divergence analysis
        classifies as divergent.

        Restricted to kernel-body-direct declarations because inner-kx_scope
        names are invisible at the injection sites -- this is what keeps the
        output compilable (the audited implementation's use-before-kx_scope
        failures came from picking inner-kx_scope names).

        Delegating to :class:`~generator_ir.ir.divergence.DivergenceAnalyzer`
        catches not just direct ``threadIdx`` seeds but also dataflow
        propagation (``a = tid`` then ``b = a``) and assignments made inside
        divergent branches.
        """
        body = kernel.body
        if body is None:
            return []
        top = [s.name for s in body.stmts if isinstance(s, Declare)]
        analyzer = DivergenceAnalyzer(kernel)
        return sorted(name for name in top if analyzer.kx_is_divergent(name))

    def _spread_dataflow(self, kernel: Func, name: str, rng: random.Random) -> bool:
        assigns = []
        for a in assignments(kernel):
            if isinstance(a.target, Var) and a.target.name == name:
                continue
            if collect(a.value, lambda n: isinstance(n, Var) and n.name == name):
                continue
            assigns.append(a)
        # Only sites AFTER the declaration (program order).
        body_stmts = kernel.body.stmts
        decl_idx = next((i for i, s in enumerate(body_stmts)
                         if isinstance(s, Declare) and s.name == name), -1)
        if decl_idx >= 0:
            visible: Set[int] = set()
            for i in range(decl_idx + 1, len(body_stmts)):
                for a in assignments(body_stmts[i]):
                    visible.add(id(a))
            assigns = [a for a in assigns if id(a) in visible]
        if not assigns:
            return False
        target = rng.choice(assigns)
        ref = Var(T.u32, name=name)
        old = copy.deepcopy(target.value)
        choice = rng.randint(0, 2)
        if choice == 0:
            target.value = BinOp(T.u32, op=Arith.MUL, lhs=old, rhs=ref)
        elif choice == 1:
            target.value = BinOp(T.u32, op=Arith.ADD, lhs=old, rhs=ref)
        else:
            target.value = BinOp(old.ty, op=Arith.BXOR, lhs=old, rhs=ref)
        return True

    def _spread_control_flow(self, kernel: Func, name: str, rng: random.Random) -> bool:
        body_stmts = kernel.body.stmts
        decl_idx = next((i for i, s in enumerate(body_stmts)
                         if isinstance(s, Declare) and s.name == name), -1)
        # Only top-level ifs after the decl can see the name.
        candidates = []
        for i, s in enumerate(body_stmts):
            if not isinstance(s, If):
                continue
            if decl_idx >= 0 and i <= decl_idx:
                continue
            if collect(s.cond, lambda n: isinstance(n, Var) and n.name == name):
                continue
            # SAFETY: never make the guard of a barrier-containing branch
            # divergent -- divergent __syncthreads deadlocks (perffuzzer
            # applies the same _contains_sync filter to its channel).
            if collect(s, lambda n: isinstance(n, Barrier)):
                continue
            candidates.append(s)
        if not candidates:
            return False
        target = rng.choice(candidates)
        ref = Var(T.u32, name=name)
        mask = BinOp(T.u32, op=Arith.BAND, lhs=ref, rhs=_uint(1))
        pred = RelOp(T.boolean, op=Cmp.NE, lhs=mask, rhs=_uint(0))
        target.cond = LogicOp(T.boolean, op=Logic.AND,
                              lhs=target.cond, rhs=pred)
        return True

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None or kernel.body is None:
            return prog
        count = param if param is not None else rng.choice(self.sweep_space)
        names = self._kernel_scope_divergent(kernel)
        if not names:
            # Self-seed a kernel-kx_scope divergence source: uint div_src = tid.
            taken = all_var_names(kernel)
            div_name = kx_fresh_name("div_src", taken)
            kernel.body.stmts.insert(0, Declare(
                name=div_name, gtype=T.u32, space=MemSpace.REGISTER,
                init=_tid("x")))
            names = [div_name]
        for _ in range(count):
            name = rng.choice(names)
            if rng.random() < 0.5:
                ok = self._spread_dataflow(kernel, name, rng)
            else:
                ok = self._spread_control_flow(kernel, name, rng)
            if not ok:
                # fall back to the other channel
                alt = (self._spread_control_flow
                       if rng.random() < 0.5 else self._spread_dataflow)
                ok = alt(kernel, name, rng)
            if ok:
                # Pick up newly divergent kernel-kx_scope variables.
                for nd in self._kernel_scope_divergent(kernel):
                    if nd not in names:
                        names.append(nd)
        return prog

    # -- propagation removal (opt direction) -----------------------------

    def _drop_divergent_if(self, kernel: Func, rng: random.Random) -> bool:
        """Flatten an if whose condition is divergent to its then-branch."""
        div_names = set(self._kernel_scope_divergent(kernel))
        for block in blocks(kernel):
            for i, stmt in enumerate(block.stmts):
                if not isinstance(stmt, If):
                    continue
                cond_names = {v.name for v in collect(
                    stmt.cond, lambda n: isinstance(n, Var))}
                if (cond_names & div_names) or _has_tid(stmt.cond):
                    then_stmts = list(stmt.then.stmts) if stmt.then else []
                    block.stmts[i:i + 1] = then_stmts
                    return True
        return False

    def _drop_divergent_dataflow(self, kernel: Func, div_names: List[str],
                                   rng: random.Random) -> bool:
        """Replace uses of a divergent variable in an assignment with the
        uniform constant 1."""
        if not div_names:
            return False
        for block in blocks(kernel):
            for stmt in block.stmts:
                if not isinstance(stmt, (Assign, CompoundAssign)):
                    continue
                for name in div_names:
                    if collect(stmt.value, lambda n: isinstance(n, Var)
                               and n.name == name):
                        replace_var_everywhere(stmt.value, name,
                                               lambda: _uint(1))
                        return True
        return False

    def perturb_opt(self, program, rng, param=None) -> Program:
        """Opt direction: REMOVE propagation -- flatten divergent branches
        and substitute divergent vars with uniform constants."""
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None or kernel.body is None:
            return prog
        count = param if param is not None else rng.choice(self.sweep_space)
        names = self._kernel_scope_divergent(kernel)
        for _ in range(count):
            if rng.random() < 0.6:
                ok = self._drop_divergent_if(kernel, rng)
            else:
                ok = self._drop_divergent_dataflow(kernel, names, rng)
            if not ok:
                if rng.random() < 0.6:
                    self._drop_divergent_dataflow(kernel, names, rng)
                else:
                    self._drop_divergent_if(kernel, rng)
        return prog

    def perturb_deopt(self, program, rng, param=None) -> Program:
        return self.perturb(program, rng, param)


# ---------------------------------------------------------------------------
# DIV3: barrier insert/remove
# ---------------------------------------------------------------------------

class DivBarrierInsertRemove(Mutation):
    """Insert barriers into uniform loops, or remove barriers from kernels
    that do not touch shared memory."""

    name = "DIV3_barrier_insert_remove"
    category = "divergence"
    sweep_space = [0, 1, 2, 4]

    @staticmethod
    def _uses_shared(kernel: Func) -> bool:
        for d in declarations(kernel):
            if d.space is MemSpace.SHARED:
                return True
        return False

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None or kernel.body is None:
            return prog
        count = param if param is not None else rng.choice(self.sweep_space)

        if count == 0:
            if self._uses_shared(kernel):
                return prog
            for loop in mutable_for_loops(kernel):
                if loop.body is not None:
                    loop.body.stmts = [s for s in loop.body.stmts
                                       if not isinstance(s, Barrier)]
            return prog

        loops = [l for l in mutable_for_loops(kernel)
                 if barrier_ok_in_loop(kernel.body, l)]
        for loop in loops:
            if loop.body is None or not loop.body.stmts:
                continue
            step = max(1, len(loop.body.stmts) // (count + 1))
            inserted = 0
            new_stmts: List[Stmt] = []
            for i, s in enumerate(loop.body.stmts):
                new_stmts.append(s)
                if inserted < count and (i + 1) % step == 0:
                    if isinstance(s, UnrollHintRef):
                        continue
                    new_stmts.append(Barrier())
                    inserted += 1
            while inserted < count:
                new_stmts.append(Barrier())
                inserted += 1
            loop.body.stmts = new_stmts
        return prog

    def perturb_opt(self, program, rng, param=None) -> Program:
        return self.perturb(program, rng, 0)

    def perturb_deopt(self, program, rng, param=None) -> Program:
        return self.perturb(program, rng, param if param is not None else rng.choice([1, 2, 4]))


from generator_ir.ir.statements import UnrollHint as UnrollHintRef


# ---------------------------------------------------------------------------
# DIV4: atomic substitution (regular op -> atomic RMW)
# ---------------------------------------------------------------------------

class DivAtomicSubstitution(Mutation):
    """Replace a compound assignment into a global/shared slot with an atomic
    RMW on the same slot (same value, same address -- deterministic because
    each slot is single-writer in generated kernels)."""

    name = "DIV4_atomic_substitution"
    category = "divergence"
    sweep_space = [1, 2, 4]

    _OP_TO_ATOMIC = {
        "add": "Add", "sub": "Sub", "band": "And", "bor": "Or",
        "bxor": "Xor",
    }
    _ATOMIC_TO_COMPOUND = {
        "Add": Compound.ADD, "Sub": Compound.SUB, "And": Compound.BAND,
        "Or": Compound.BOR, "Xor": Compound.BXOR,
    }

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None:
            return prog
        count = param if param is not None else rng.choice(self.sweep_space)

        # Candidate: compound assigns to output[] slots with uint type.
        cands = []
        for ca in compound_assignments(kernel):
            if not isinstance(ca.target, Subscript):
                continue
            if not (isinstance(ca.target.array, Var)
                    and ca.target.array.name == "output"):
                continue
            atomic = self._OP_TO_ATOMIC.get(ca.op.name.lower())
            if atomic is None:
                continue
            cands.append((ca, atomic, ca.value))
        # Also: output[i] = output[i] + val -> atomicAdd(&output[i], val).
        for a in assignments(kernel):
            if not (isinstance(a.target, Subscript)
                    and isinstance(a.target.array, Var)
                    and a.target.array.name == "output"):
                continue
            if not (isinstance(a.value, BinOp) and a.value.op is Arith.ADD):
                continue
            lhs = a.value.lhs
            if not (isinstance(lhs, Subscript)
                    and isinstance(lhs.array, Var)
                    and lhs.array.name == a.target.array.name
                    and _same_expr(a.target.index, lhs.index)):
                continue
            cands.append((a, "Add", a.value.rhs))
        if not cands:
            return prog
        for stmt, atomic, value in rng.sample(cands, min(count, len(cands))):
            repl = ExprStmt(Atomic(T.u32, what=atomic,
                                   addr=copy.deepcopy(stmt.target),
                                   operand=copy.deepcopy(value)))
            parent = walk_parent_block_of(kernel, stmt)
            if parent is None:
                continue
            for i, s in enumerate(parent.stmts):
                if s is stmt:
                    parent.stmts[i] = repl
                    break
        return prog

    def perturb_opt(self, program, rng, param=None) -> Program:
        """Opt direction: de-atomize -- atomicAdd(&x, v) -> x += v, etc."""
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None:
            return prog
        count = param if param is not None else rng.choice(self.sweep_space)
        cands = []
        for block in blocks(kernel):
            for i, s in enumerate(block.stmts):
                if (isinstance(s, ExprStmt) and isinstance(s.expr, Atomic)
                        and s.expr.what in self._ATOMIC_TO_COMPOUND):
                    cands.append((block, i, s.expr))
        if not cands:
            return prog
        for block, idx, atom in rng.sample(cands, min(count, len(cands))):
            if isinstance(atom.addr, AddrOf):
                target = copy.deepcopy(atom.addr.arg)
            elif isinstance(atom.addr, (Var, Subscript)):
                target = copy.deepcopy(atom.addr)
            else:
                continue
            block.stmts[idx] = CompoundAssign(
                target, self._ATOMIC_TO_COMPOUND[atom.what],
                copy.deepcopy(atom.operand))
        return prog

    def perturb_deopt(self, program, rng, param=None) -> Program:
        return self.perturb(program, rng, param)




class ExtBranchToArithmetic(Mutation):
    """if (c) x = a; else x = b;  ->  x = c*a + (1-c)*b; (arithmetic blend).

    The boolean condition is cast to the (numeric) value type so it acts as a
    0/1 kx_multiplier; only numeric value types are eligible, since the blend
    requires arithmetic on the value."""

    name = "EXT_branch_to_arithmetic"
    category = "divergence"

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None:
            return prog
        cands = []
        for ifs in if_statements(kernel):
            if ifs.otherwise is None:
                continue
            if (len(ifs.then.stmts) == 1 and len(ifs.otherwise.stmts) == 1):
                t, e = ifs.then.stmts[0], ifs.otherwise.stmts[0]
                if (isinstance(t, Assign) and isinstance(e, Assign)
                        and isinstance(t.target, Var) and isinstance(e.target, Var)
                        and t.target.name == e.target.name
                        and (t.value.ty.is_int or t.value.ty.is_fp)):
                    cands.append((ifs, t, e))
        if not cands:
            return prog
        ifs, t, e = rng.choice(cands)
        ty = t.value.ty
        cond_as_num = Cast(ty, to=ty, arg=copy.deepcopy(ifs.cond))
        one = Lit(ty, value=1)
        # x = (uint)c*a + ((uint)(1 - c))*b
        ca = BinOp(ty, op=Arith.MUL,
                   lhs=copy.deepcopy(cond_as_num),
                   rhs=copy.deepcopy(t.value))
        one_minus_c = BinOp(ty, op=Arith.SUB,
                            lhs=copy.deepcopy(one),
                            rhs=cond_as_num)
        one_cb = BinOp(ty, op=Arith.MUL,
                       lhs=one_minus_c,
                       rhs=copy.deepcopy(e.value))
        blend = BinOp(ty, op=Arith.ADD, lhs=ca, rhs=one_cb)
        repl = Assign(copy.deepcopy(t.target), blend)
        parent = walk_parent_block_of(kernel, ifs)
        if parent is None:
            return prog
        for i, s in enumerate(parent.stmts):
            if s is ifs:
                parent.stmts[i] = repl
                break
        return prog


# ---------------------------------------------------------------------------
# EXT: reconvergence shift (move trailing common code into both branches)
# ---------------------------------------------------------------------------

class ExtReconvergenceShift(Mutation):
    """Move the statement following an if/else into both branch bodies
    (duplicated), shifting the reconvergence point."""

    name = "EXT_reconvergence_shift"
    category = "divergence"

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None:
            return prog
        for block in blocks(kernel):
            for i, s in enumerate(block.stmts[:-1]):
                if not isinstance(s, If) or s.otherwise is None:
                    continue
                nxt = block.stmts[i + 1]
                # Only move pure assignments (no barriers/decls/control flow).
                if not isinstance(nxt, (Assign, CompoundAssign, ExprStmt)):
                    continue
                s.then.stmts.append(copy.deepcopy(nxt))
                s.otherwise.stmts.append(copy.deepcopy(nxt))
                del block.stmts[i + 1]
                return prog
        return prog


# ---------------------------------------------------------------------------
# EXT: predicated <-> branched
# ---------------------------------------------------------------------------

class ExtPredicatedBranched(Mutation):
    """Convert a ternary assignment into an if/else (or vice versa)."""

    name = "EXT_predicated_branched"
    category = "divergence"

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None:
            return prog
        # ternary assign -> if/else
        cands = []
        for a in assignments(kernel):
            if isinstance(a.value, Select) and isinstance(a.target, Var):
                cands.append(a)
        if not cands:
            return prog
        a = rng.choice(cands)
        sel = a.value
        assert isinstance(sel, Select)
        repl = If(cond=sel.cond,
                  then=Block([Assign(copy.deepcopy(a.target), sel.when_true)]),
                  otherwise=Block([Assign(copy.deepcopy(a.target), sel.when_false)]))
        parent = walk_parent_block_of(kernel, a)
        if parent is None:
            return prog
        for i, s in enumerate(parent.stmts):
            if s is a:
                parent.stmts[i] = repl
                break
        return prog


# ---------------------------------------------------------------------------
# EXT: warp-centric refactoring (wrap a body in a lane-uniform loop)
# ---------------------------------------------------------------------------

class ExtWarpCentric(Mutation):
    """Wrap a uniform loop body's scalar work in a warp-iteration shape:
    for (lane_step) with shfl-based broadcast."""

    name = "EXT_warp_centric"
    category = "divergence"

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None or kernel.body is None:
            return prog
        if not kx_is_cuda(prog) and not prog.launch.arch.startswith("gfx"):
            return prog
        loops = [l for l in mutable_for_loops(kernel)
                 if for_trip_uniform(l) and l.body is not None]
        if not loops:
            return prog
        loop = rng.choice(loops)
        # Insert a warp-vote on the loop counter at the body head (uniform).
        vote = WarpVote(T.u32, kind="ballot_sync",
                        mask=_uint(0xFFFFFFFF),
                        predicate=RelOp(T.boolean, op=Cmp.LT,
                                        lhs=Var(T.i32, name=loop.var),
                                        rhs=copy.deepcopy(loop.end)))
        tmp = kx_fresh_name("wc_ballot", all_var_names(kernel))
        loop.body.stmts.insert(0, Declare(name=tmp, gtype=T.u32, init=vote))
        # Keep the vote live without leaving its kx_scope: fold into _fv when it
        # is visible, else into a fresh accumulator declared before the loop
        # (both keep `tmp` referenced only where it is in kx_scope).
        if "_fv" in all_var_names(kernel):
            loop.body.stmts.append(Assign(
                Var(T.u32, name="_fv"),
                BinOp(T.u32, op=Arith.BXOR,
                      lhs=Var(T.u32, name="_fv"),
                      rhs=Var(T.u32, name=tmp))))
        else:
            acc = kx_fresh_name("wc_acc", all_var_names(kernel))
            parent = walk_parent_block_of(kernel, loop)
            if parent is None:
                return prog
            for i, s in enumerate(parent.stmts):
                if s is loop:
                    parent.stmts.insert(i, Declare(name=acc, gtype=T.u32,
                                                   init=_uint(0)))
                    break
            loop.body.stmts.append(Assign(
                Var(T.u32, name=acc),
                BinOp(T.u32, op=Arith.BXOR,
                      lhs=Var(T.u32, name=acc),
                      rhs=Var(T.u32, name=tmp))))
            keep_value_live(kernel, Var(T.u32, name=acc), base="wc_acc",
                            allow_volatile=getattr(prog.launch, "allow_volatile", False))
        return prog


# ---------------------------------------------------------------------------
# EXT: switch divergence (convert statements to a switch tree)
# ---------------------------------------------------------------------------

class ExtSwitchDivergence(Mutation):
    """Rebuild one statement as a per-lane divergent switch with per-arm
    tweaked variants (2/4/8 arms + default when >2)."""

    name = "EXT_switch_divergence"
    category = "divergence"
    sweep_space = [2, 4, 8]

    @staticmethod
    def _kx_tweak_stmt(stmt: Stmt, rng: random.Random) -> Stmt:
        """A semantics-varying clone of *stmt* for the non-base arms."""
        cloned = copy.deepcopy(stmt)
        if isinstance(cloned, Assign) and isinstance(cloned.value, BinOp):
            if cloned.value.op is Arith.ADD:
                cloned.value.lhs, cloned.value.rhs = (
                    copy.deepcopy(cloned.value.rhs),
                    copy.deepcopy(cloned.value.lhs))
            elif cloned.value.op is Arith.MUL:
                cloned.value = BinOp(cloned.value.ty, op=Arith.ADD,
                                     lhs=copy.deepcopy(cloned.value.lhs),
                                     rhs=copy.deepcopy(cloned.value.rhs))
        elif isinstance(cloned, CompoundAssign):
            if cloned.op is Compound.ADD:
                cloned.op = Compound.SUB
            elif cloned.op is Compound.BXOR:
                cloned.value = BinOp(T.u32, op=Arith.ADD,
                                     lhs=copy.deepcopy(cloned.value),
                                     rhs=_uint(1))
        elif isinstance(cloned, ExprStmt) and isinstance(cloned.expr, BinOp):
            if cloned.expr.op is Arith.ADD:
                cloned.expr = BinOp(cloned.expr.ty, op=Arith.SUB,
                                    lhs=copy.deepcopy(cloned.expr.lhs),
                                    rhs=copy.deepcopy(cloned.expr.rhs))
            elif cloned.expr.op is Arith.BAND:
                cloned.expr = BinOp(cloned.expr.ty, op=Arith.MUL,
                                    lhs=copy.deepcopy(cloned.expr.lhs),
                                    rhs=copy.deepcopy(cloned.expr.rhs))
        return cloned

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None or kernel.body is None:
            return prog
        case_count = param if param is not None else rng.choice(self.sweep_space)

        # Candidate statements: no declarations (case-kx_scope would orphan later
        # uses), no barriers (divergent sync deadlocks), no switches/returns.
        candidates = []
        for block in blocks(kernel):
            for idx, stmt in enumerate(block.stmts):
                if isinstance(stmt, (Declare, Switch, Barrier)):
                    continue
                if collect(stmt, lambda n: isinstance(n, Barrier)):
                    continue
                candidates.append((block, idx, stmt))
        if not candidates:
            return prog
        block, idx, stmt = rng.choice(candidates)

        selector = BinOp(T.u32, op=Arith.BAND,
                         lhs=_tid(rng.choice(["x", "y"])),
                         rhs=_uint(max(1, case_count - 1)))
        cases = [Case(value=_uint(0),
                      body=Block([copy.deepcopy(stmt)]))]
        for case_idx in range(1, case_count):
            if case_idx == case_count - 1:
                arm = copy.deepcopy(stmt)
            else:
                arm = self._kx_tweak_stmt(stmt, rng)
            cases.append(Case(value=_uint(case_idx), body=Block([arm])))
        if case_count > 2:
            cases.append(Case(value=None,
                              body=Block([self._kx_tweak_stmt(stmt, rng)])))
        block.stmts[idx] = Switch(expr=selector, cases=cases)
        return prog

    def perturb_opt(self, program, rng, param=None) -> Program:
        """Opt direction: collapse an existing switch to its first arm."""
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None or kernel.body is None:
            return prog
        for block in blocks(kernel):
            for i, stmt in enumerate(block.stmts):
                if isinstance(stmt, Switch) and stmt.cases:
                    block.stmts[i:i + 1] = list(stmt.cases[0].body.stmts)
                    return prog
        return prog

    def perturb_deopt(self, program, rng, param=None) -> Program:
        """Deopt direction: build the divergent switch (base perturb)."""
        return self.perturb(program, rng, param)


# ---------------------------------------------------------------------------
# EXT: barrier relocation
# ---------------------------------------------------------------------------

class ExtBarrierRelocation(Mutation):
    """Move a barrier within its uniform loop body (reconvergence stress)."""

    name = "EXT_barrier_relocation"
    category = "divergence"

    @staticmethod
    def _accesses_shared(stmt: Stmt, shared_names: Set[str]) -> bool:
        """True if *stmt* reads or writes any __shared__ variable."""
        for sub in collect(stmt, lambda n: isinstance(n, Subscript)):
            if isinstance(sub.array, Var) and sub.array.name in shared_names:
                return True
        for v in collect(stmt, lambda n: isinstance(n, Var)):
            if v.name in shared_names and v.space is MemSpace.SHARED:
                return True
        return False

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None or kernel.body is None:
            return prog
        shared_names = {d.name for d in declarations(kernel)
                        if d.space is MemSpace.SHARED}
        for loop in mutable_for_loops(kernel):
            if loop.body is None:
                continue
            idxs = [i for i, s in enumerate(loop.body.stmts)
                    if isinstance(s, Barrier)]
            if not idxs or not barrier_ok_in_loop(kernel.body, loop):
                continue
            i = rng.choice(idxs)
            loop.body.stmts.pop(i)
            # Reinsert at a random UNIFORM position (top-level of the body).
            pos = rng.randint(0, len(loop.body.stmts))
            # Abort if any statement between the old and new position accesses
            # shared memory -- reordering a barrier past a shared access breaks
            # the happens-before edge and races the slots.
            lo, hi = min(i, pos), max(i, pos)
            if any(self._accesses_shared(s, shared_names)
                   for s in loop.body.stmts[lo:hi]):
                loop.body.stmts.insert(i, Barrier())
                return prog
            loop.body.stmts.insert(pos, Barrier())
            return prog
        return prog


# ---------------------------------------------------------------------------
# EXT: CUDA warp collectives
# ---------------------------------------------------------------------------

class ExtCudaWarpCollectives(Mutation):
    """Inject a CUDA warp match/reduce collective (match on sm_70+, reduce on
    sm_80+, or a hybrid of both) folded through the live-value sink."""

    name = "EXT_cuda_warp_collectives"
    category = "divergence"
    sweep_space = ["match", "reduce", "hybrid"]

    def _kx_build_collective(self, prog: Program, rng: random.Random,
                          mode: str) -> Expr:
        # __activemask() names exactly the participating lanes; a hardcoded
        # 0xFFFFFFFF mask is unsafe under divergent control flow.
        active = Call(T.u32, fn="__activemask", args=[], result_ty=T.u32)
        lane = BinOp(T.u32, op=Arith.BAND,
                     lhs=Var(T.u32, name="tid"), rhs=_uint(31))
        match = Call(T.u32, fn="__match_any_sync",
                     args=[active, lane], result_ty=T.u32)
        if mode == "match" or cuda_sm_version(prog) < 80:
            return match
        reduce_fn = rng.choice(["__reduce_add_sync", "__reduce_xor_sync",
                                "__reduce_or_sync"])
        reduce_val = BinOp(T.u32, op=Arith.BXOR,
                           lhs=Var(T.u32, name="tid"),
                           rhs=_uint(rng.choice([1, 3, 7, 15])))
        active2 = Call(T.u32, fn="__activemask", args=[], result_ty=T.u32)
        reduce = Call(T.u32, fn=reduce_fn,
                      args=[active2, reduce_val], result_ty=T.u32)
        if mode == "reduce":
            return reduce
        return BinOp(T.u32, op=Arith.BXOR, lhs=match, rhs=reduce)

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        if prog.launch.kx_is_2d or not kx_is_cuda(prog, 70):
            return prog
        kernel = kernel_of(prog)
        if kernel is None or kernel.body is None:
            return prog
        mode = str(param) if param is not None else rng.choice(self.sweep_space)
        collective = self._kx_build_collective(prog, rng, mode)
        keep_value_live(kernel, collective, base="cuda_warp",
                        allow_volatile=getattr(prog.launch, "allow_volatile", False))
        return prog

    def perturb_opt(self, program, rng, param=None) -> Program:
        """Opt direction: favor warp reductions."""
        mode = "reduce" if cuda_sm_version(program) >= 80 else "match"
        return self.perturb(program, rng, param=mode)

    def perturb_deopt(self, program, rng, param=None) -> Program:
        """Deopt direction: favor hybrid match+reduce pressure."""
        return self.perturb(program, rng, param="hybrid")


# ---------------------------------------------------------------------------
# EXT: asymmetric divergence (heavy then / light else)
# ---------------------------------------------------------------------------

class ExtAsymmetricDivergence(Mutation):
    """Convert an if/else so the then-branch carries 4+ live mixed vars while
    the else-branch is a single xor."""

    name = "EXT_asymmetric_divergence"
    category = "divergence"

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None or kernel.body is None:
            return prog
        taken = all_var_names(kernel)
        # Only kernel-body top-level uint vars are in kx_scope at the insertion
        # point (before the trailing output region); loop-nested vars (e.g.
        # the reduction's sh_* shuffle temps) are not.
        uints = sorted({s.name for s in kernel.body.stmts
                        if isinstance(s, Declare)
                        and isinstance(s.gtype, TUInt)
                        and s.name not in ("tid", "_fv")})
        if not uints:
            return prog
        base_name = rng.choice(uints)
        # The accumulator is declared BEFORE the if (at body level) so the
        # trailing fold can reference it (declaring inside the then-block
        # would leave it out of kx_scope for the liveness fold).
        acc = kx_fresh_name("asym", taken)
        taken.add(acc)
        heavy: List[Stmt] = []
        for k in range(4):
            heavy.append(Assign(
                Var(T.u32, name=acc),
                BinOp(T.u32, op=Arith.BXOR, lhs=Var(T.u32, name=acc),
                      rhs=BinOp(T.u32, op=Arith.MUL,
                                lhs=Var(T.u32, name=acc),
                                rhs=_uint(0x9E3779B9 + k)))))
        light = Assign(Var(T.u32, name=acc),
                       BinOp(T.u32, op=Arith.BXOR, lhs=Var(T.u32, name=acc),
                             rhs=_uint(0x5BD1E995)))
        cond = RelOp(T.boolean, op=Cmp.LT,
                     lhs=_tid("x"), rhs=_uint(16))
        acc_decl = Declare(name=acc, gtype=T.u32,
                           init=Var(T.u32, name=base_name))
        ifs = If(cond=cond, then=Block(heavy), otherwise=Block([light]))
        # Insert at the end of the kernel body (before trailing if).
        stmts = kernel.body.stmts
        pos = len(stmts)
        for i in range(len(stmts) - 1, -1, -1):
            if isinstance(stmts[i], If):
                pos = i
                break
        stmts.insert(pos, acc_decl)
        stmts.insert(pos + 1, ifs)
        keep_value_live(kernel, Var(T.u32, name=acc), base="asym",
                        allow_volatile=getattr(prog.launch, "allow_volatile", False))
        return prog
