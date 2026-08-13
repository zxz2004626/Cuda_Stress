"""EMI-style mutation operators.

EMI dead code, true guards, fake divergence, and cross-kernel splicing.
All preserve the observable result: guards are provably never/always taken
at runtime but opaque to the compiler.
"""

from __future__ import annotations

import copy
import random
from typing import List, Optional

from generator_ir.ir.typedecl import (
    GType, T, MemSpace, Qual, TArray, TPointer, TInt, TUInt,
)
from generator_ir.ir.operators import Arith, Cmp, SafeOp
from generator_ir.ir.nodes import (
    Expr, Lit, Var, ThreadBuiltin, BinOp, RelOp, LogicOp, Cast, SafeBin,
    Subscript, Call, Atomic,
)
from generator_ir.ir.statements import (
    Stmt, Assign, CompoundAssign, ExprStmt, Declare, Block, If, For,
    While, DoWhile, Switch, Barrier, RawCode,
)
from generator_ir.ir.program import Func, Program
from generator_ir.ir import traversal as traverse
from generator_ir.ir.traversal import (
    clone_program, kernel_of, collect, mutable_for_loops, declarations,
    if_statements, blocks, all_var_names, kx_fresh_name, keep_value_live,
)
from generator_ir.mutate.foundation import Mutation


def _uint(v: int) -> Lit:
    return Lit(T.u32, value=v)


def _tid(axis: str = "x") -> ThreadBuiltin:
    return ThreadBuiltin(T.u32, axis=axis, source="threadIdx")


def _parent_block(root, target: Stmt) -> Optional[Block]:
    for block in blocks(root):
        for s in block.stmts:
            if s is target:
                return block
    return None


def _opaque_flag_decl(kernel: Func, value: int, base: str,
                      allow_volatile: bool) -> tuple:
    """``[volatile] unsigned int <base> = <opaque value>;``

    The initializer derives from the runtime ``n`` parameter so the compiler
    cannot constant-fold it, but always evaluates to *value* at runtime:
    ``((unsigned)n >> 31) | value`` is value for n < 2^31 (always true)."""
    taken = all_var_names(kernel)
    name = kx_fresh_name(base, taken)
    n_shift = BinOp(T.u32, op=Arith.SHR,
                    lhs=Cast(T.u32, to=T.u32, arg=Var(T.i32, name="n")),
                    rhs=_uint(31))
    init = BinOp(T.u32, op=Arith.BOR, lhs=n_shift, rhs=_uint(value))
    quals = [Qual.VOLATILE] if allow_volatile else []
    decl = Declare(name=name, gtype=T.u32, init=init, quals=quals)
    return name, decl


class EmiDeadCode(Mutation):
    """Wrap a for/if in provably-dead code: if (flag != 0) { stmt } where
    flag is runtime-0 but compiler-opaque."""

    name = "EXT_emi_dead_code"
    category = "divergence"

    @staticmethod
    def _writes_shared(candidate, shared_names: set) -> bool:
        """True if *candidate* writes to a __shared__ variable (assign /
        compound-assign / atomic)."""
        writes = list(traverse.assignments(candidate)) + \
            list(traverse.compound_assignments(candidate))
        for a in writes:
            t = a.target
            if isinstance(t, Var) and t.name in shared_names:
                return True
            if (isinstance(t, Subscript) and isinstance(t.array, Var)
                    and t.array.name in shared_names):
                return True
        for atom in collect(candidate, lambda n: isinstance(n, Atomic)):
            addr = atom.addr
            if isinstance(addr, Var) and addr.name in shared_names:
                return True
            if (isinstance(addr, Subscript) and isinstance(addr.array, Var)
                    and addr.array.name in shared_names):
                return True
        return False

    @staticmethod
    def _names_assigned(candidate) -> set:
        names: set = set()
        for a in list(traverse.assignments(candidate)) + \
                list(traverse.compound_assignments(candidate)):
            t = a.target
            if isinstance(t, Var):
                names.add(t.name)
            elif isinstance(t, Subscript) and isinstance(t.array, Var):
                names.add(t.array.name)
        return names

    @staticmethod
    def _read_elsewhere(kernel: Func, candidate, names: set) -> bool:
        if not names:
            return False
        for s in kernel.body.stmts:
            if s is candidate:
                continue
            for v in collect(s, lambda n: isinstance(n, Var)):
                if v.name in names:
                    return True
        return False

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None or kernel.body is None:
            return prog
        allow_vol = getattr(prog.launch, "allow_volatile", False)
        shared_names = {d.name for d in declarations(kernel)
                        if d.space is MemSpace.SHARED}
        # Eligible top-level For/If only: must not contain a barrier, must not
        # write to shared memory, and must not assign to a variable referenced
        # elsewhere (wrapping in dead code would leave those reads stale).
        cands = []
        for s in kernel.body.stmts:
            if not isinstance(s, (For, If)):
                continue
            if collect(s, lambda n: isinstance(n, Barrier)):
                continue
            if self._writes_shared(s, shared_names):
                continue
            assigned = self._names_assigned(s)
            if self._read_elsewhere(kernel, s, assigned):
                continue
            cands.append(s)
        if not cands:
            return prog
        target = rng.choice(cands)
        flag, decl = _opaque_flag_decl(kernel, 0, "emi_flag", allow_vol)
        cond = RelOp(T.boolean, op=Cmp.NE,
                     lhs=Var(T.u32, name=flag), rhs=_uint(0))
        wrapped = If(cond=cond, then=Block([target]))
        stmts = kernel.body.stmts
        for i, s in enumerate(stmts):
            if s is target:
                stmts[i:i + 1] = [decl, wrapped]
                break
        return prog

    def perturb_deopt(self, program, rng, param=None) -> Program:
        """Deopt direction: add the dead wrapper (base perturb)."""
        return self.perturb(program, rng, param)

    def perturb_opt(self, program, rng, param=None) -> Program:
        """Opt direction: remove a previously-injected EMI wrapper -- unwrap
        the if's then-branch and delete the emi_flag declaration."""
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None or kernel.body is None:
            return prog
        wrappers = []
        for block in blocks(kernel):
            for i, s in enumerate(block.stmts):
                if not isinstance(s, If):
                    continue
                flag_names = {v.name for v in collect(
                    s.cond, lambda n: isinstance(n, Var))
                    if v.name.startswith("emi_flag")}
                if flag_names:
                    wrappers.append((block, i, s, flag_names))
        if not wrappers:
            return prog
        block, idx, if_stmt, flag_names = rng.choice(wrappers)
        inner = list(if_stmt.then.stmts) if if_stmt.then else []
        block.stmts[idx:idx + 1] = inner
        # Drop the flag declaration(s) in the same block.
        block.stmts = [s for s in block.stmts
                       if not (isinstance(s, Declare)
                               and s.name in flag_names)]
        return prog


class EmiTrueGuard(Mutation):
    """Wrap statements in an always-true but compiler-opaque guard:
    if (flag != 0) { stmts } where flag is runtime-1."""

    name = "EXT_true_guard"
    category = "divergence"

    @staticmethod
    def _select_stmt_execute(block: Block, rng: random.Random,
                         min_run: int = 2, max_run: int = 4):
        """(start, end) of a wrap-safe run: no Declare (kx_scope) and no
        Barrier (must not move into guards)."""
        n = len(block.stmts)
        if n < min_run:
            return None
        start_candidates = [
            i for i in range(n)
            if not isinstance(block.stmts[i], (Declare, Barrier))
            and (n - i) >= min_run
        ]
        if not start_candidates:
            return None
        start = rng.choice(start_candidates)
        max_possible = min(max_run, n - start)
        if max_possible < min_run:
            return None
        run_len = rng.randint(min_run, max_possible)
        end = start + run_len
        if any(isinstance(s, (Declare, Barrier))
               for s in block.stmts[start:end]):
            return None
        return (start, end)

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None or kernel.body is None:
            return prog
        allow_vol = getattr(prog.launch, "allow_volatile", False)

        block = kernel.body
        result = self._select_stmt_execute(block, rng)
        if result is None:
            # Fallback: wrap inside a top-level for-loop body instead.
            for s in kernel.body.stmts:
                if isinstance(s, For) and isinstance(s.body, Block):
                    result = self._select_stmt_execute(s.body, rng,
                                                   min_run=1, max_run=3)
                    if result is not None:
                        block = s.body
                        break
        if result is None:
            return prog

        start, end = result
        chunk = block.stmts[start:end]
        flag, decl = _opaque_flag_decl(kernel, 1, "tg_flag", allow_vol)
        cond = RelOp(T.boolean, op=Cmp.NE,
                     lhs=Var(T.u32, name=flag), rhs=_uint(0))
        wrapped = If(cond=cond, then=Block(chunk))
        block.stmts[start:end] = [decl, wrapped]
        return prog

    def perturb_opt(self, program, rng, param=None) -> Program:
        """Opt direction: unwrap an existing true guard -- replace the
        [tg_flag decl, guarded if] pair with the if's body statements (drop
        the redundant branch overhead)."""
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None or kernel.body is None:
            return prog
        for block in blocks(kernel):
            new_stmts = []
            i = 0
            while i < len(block.stmts):
                s = block.stmts[i]
                if (isinstance(s, Declare) and s.name.startswith("tg_flag")
                        and i + 1 < len(block.stmts)):
                    nxt = block.stmts[i + 1]
                    if (isinstance(nxt, If)
                            and isinstance(nxt.cond, RelOp)
                            and isinstance(nxt.cond.lhs, Var)
                            and nxt.cond.lhs.name == s.name
                            and nxt.then is not None
                            and (nxt.otherwise is None
                                 or not nxt.otherwise.stmts)):
                        new_stmts.extend(nxt.then.stmts)
                        i += 2
                        continue
                new_stmts.append(s)
                i += 1
            block.stmts = new_stmts
        return prog

    def perturb_deopt(self, program, rng, param=None) -> Program:
        """Deopt direction: add a true guard (a provably-taken but
        compiler-opaque branch -- redundant control-flow overhead)."""
        return self.perturb(program, rng, param)


class EmiFakeDivergence(Mutation):
    """Add thread-dependent terms that cancel to zero:
    expr + (threadIdx.x - threadIdx.x)."""

    name = "EXT_fake_divergence"
    category = "divergence"
    sweep_space = [1, 2, 3]

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None:
            return prog
        count = param if param is not None else rng.choice(self.sweep_space)
        assigns = [a for a in traverse.assignments(kernel)
                   if isinstance(a.value, BinOp)
                   and a.value.op in (Arith.ADD, Arith.BXOR, Arith.MUL)]
        if not assigns:
            return prog
        for a in rng.sample(assigns, min(count, len(assigns))):
            tid = _tid(rng.choice(["x", "y"]))
            zero = BinOp(T.u32, op=Arith.SUB, lhs=tid,
                         rhs=ThreadBuiltin(T.u32, axis=tid.axis,
                                           source="threadIdx"))
            old = a.value
            if old.op is Arith.MUL:
                # x * y -> x * (y + zero)
                a.value = BinOp(old.ty, op=Arith.MUL, lhs=old.lhs,
                                rhs=BinOp(old.rhs.ty, op=Arith.ADD,
                                          lhs=old.rhs, rhs=zero))
            else:
                a.value = BinOp(old.ty, op=old.op, lhs=old.lhs,
                                rhs=BinOp(old.rhs.ty, op=old.op,
                                          lhs=old.rhs, rhs=zero))
        return prog


class ExtSplice(Mutation):
    """Transplant a statement slice from a donor kernel (fresh or corpus).

    The slice is renamed to avoid collisions and kept live through the
    standard sink machinery so it cannot corrupt the recipient's output.
    """

    name = "EXT_splice"
    category = "compute"
    sweep_space = [2, 4, 6]

    def __init__(self) -> None:
        self._donor_corpus = None

    def kx_set_donor_corpus(self, corpus) -> None:
        self._donor_corpus = corpus

    def _donor_program(self, rng: random.Random) -> Optional[Program]:
        if self._donor_corpus is not None:
            donor = self._donor_corpus.kx_random_program(rng)
            if donor is not None:
                return donor
        # Fallback: kx_generate a fresh donor.
        try:
            from generator_ir.gen.options import GenConfig
            from generator_ir.gen.assembler import build_kernel
            return build_kernel(GenConfig(seed=rng.randint(0, 1 << 30),
                                             complexity="low"))
        except Exception:
            return None

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None or kernel.body is None:
            return prog
        max_len = param if param is not None else rng.choice(self.sweep_space)

        donor = self._donor_program(rng)
        if donor is None:
            return prog
        donor_kernel = kernel_of(donor)
        if donor_kernel is None or donor_kernel.body is None:
            return prog

        # Collect simple donor statements (assignments between uint scalars).
        simple: List[Stmt] = []
        for a in traverse.assignments(donor_kernel.body):
            if (isinstance(a.target, Var) and isinstance(a.value, Expr)
                    and not collect(a, lambda n: isinstance(n, Subscript))
                    and not collect(a, lambda n: isinstance(n, ThreadBuiltin))):
                simple.append(a)
        if not simple:
            return prog

        take = simple[:max_len]
        # Rename all donor vars to fresh sp_* names.
        taken = all_var_names(kernel)
        rename: dict = {}
        for s in take:
            for v in collect(s, lambda n: isinstance(n, Var)):
                if v.name not in rename:
                    rename[v.name] = kx_fresh_name("sp", taken)
                    taken.add(rename[v.name])

        slice_stmts: List[Stmt] = []
        for name in set(rename.values()):
            slice_stmts.append(Declare(name=name, gtype=T.u32,
                                       init=_uint(rng.randint(1, 255))))
        for s in take:
            s2 = copy.deepcopy(s)
            for old, new in rename.items():
                from generator_ir.ir.traversal import replace_var_everywhere
                replace_var_everywhere(s2, old, lambda: Var(T.u32, name=new))
            # All values cast to u32.
            if isinstance(s2, Assign):
                s2.value = Cast(T.u32, to=T.u32, arg=s2.value)
                s2.target = Var(T.u32, name=rename.get(
                    getattr(s.target, "name", ""), kx_fresh_name("sp", taken)))
            slice_stmts.append(s2)

        # Insert before the trailing output region.
        stmts = kernel.body.stmts
        pos = len(stmts)
        for i in range(len(stmts) - 1, -1, -1):
            if isinstance(stmts[i], If):
                pos = i
                break
        for j, s in enumerate(slice_stmts):
            stmts.insert(pos + j, s)

        # Keep one splice variable live so the slice is not DCE'd.
        last = list(rename.values())[-1]
        folded: Expr = Var(T.u32, name=last)
        keep_value_live(kernel, folded, base="sp",
                        allow_volatile=getattr(prog.launch, "allow_volatile", False))
        return prog
