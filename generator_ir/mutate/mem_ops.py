"""Memory-hierarchy mutation operators.

Paper operators (MEM1..MEM6) plus extension operators (access-pattern scan,
shared staging, bank conflict, cache hints, shuffle/shared swap,
vectorization width, prefetch distance, kernel fusion/fission).
"""

from __future__ import annotations

import copy
import random
from typing import Dict, List, Optional, Set

from generator_ir.ir.typedecl import (
    GType, T, MemSpace, Qual, TArray, TPointer, TQualified,
    TInt, TUInt, TFloat,
)
from generator_ir.ir.operators import Arith, Cmp, SafeOp, UnaOp
from generator_ir.ir.nodes import (
    Expr, Lit, Var, ThreadBuiltin, BinOp, RelOp, UnOp, Cast, SafeBin,
    Subscript, Call, Atomic, WarpShuffle,
)
from generator_ir.ir.statements import (
    Stmt, Assign, CompoundAssign, ExprStmt, Declare, Block, If, For,
    Barrier, UnrollHint,
)
from generator_ir.ir.program import Func, Program
from generator_ir.ir.access_pattern import AccessPattern, AccessPatternAnalyzer
# traverse.* calls reference functions across traversal + safety_checks.
from generator_ir.ir import traversal as traverse
from generator_ir.ir.safety_checks import (
    block_is_uniform, kx_loop_body_escapes, kernel_has_fv_write,
)
from generator_ir.ir.traversal import (
    clone_program, kernel_of, collect, mutable_for_loops, assignments,
    subscripts, declarations, if_statements, atomics, warp_shuffles, blocks,
    all_var_names, kx_fresh_name, replace_var_everywhere, is_pow2, walk_parent_block,
    barrier_ok_in_loop, keep_value_live, thread_builtins,
)
from generator_ir.mutate.foundation import Mutation, cuda_sm_version, kx_is_cuda


def _uint(v: int) -> Lit:
    return Lit(T.u32, value=v)


def _int(v: int) -> Lit:
    return Lit(T.i32, value=v)


def _tid(axis: str = "x") -> ThreadBuiltin:
    return ThreadBuiltin(T.u32, axis=axis, source="threadIdx")


# ---------------------------------------------------------------------------
# MEM1: register -> local promotion with opaque indexing
# ---------------------------------------------------------------------------




class MemRegisterToLocal(Mutation):
    """Tile a group of same-type scalar locals into a local array indexed by
    an opaque zero (``max(-threadIdx.x * threadIdx.y, 0)``), so the array
    cannot be scalarized back into registers."""

    name = "MEM1_register_to_local"
    category = "memory"
    sweep_space = [2, 3, 4, 6, 8]

    def _find_scalar_groups(self, kernel: Func) -> List[List[Declare]]:
        if kernel.body is None:
            return []
        groups: List[List[Declare]] = []
        for block in blocks(kernel):
            current: List[Declare] = []
            current_type = None
            for s in block.stmts:
                ok = (isinstance(s, Declare)
                      and not isinstance(s.gtype, (TArray, TPointer))
                      and s.space in (MemSpace.GENERIC, MemSpace.REGISTER)
                      and s.gtype is not None
                      and (s.gtype.is_int or s.gtype.is_fp))
                if not ok:
                    if len(current) >= 2:
                        groups.append(current)
                    current, current_type = [], None
                    continue
                key = type(s.gtype)
                if current_type is None or current_type is key:
                    current.append(s)
                    current_type = key
                else:
                    if len(current) >= 2:
                        groups.append(current)
                    current, current_type = [s], key
            if len(current) >= 2:
                groups.append(current)
        return groups

    def _opaque_zero(self, taken: Set[str]):
        """int i = max(-threadIdx.x * threadIdx.y, 0)  (always 0, unprovable)."""
        idx_name = kx_fresh_name("opaque_i", taken)
        taken.add(idx_name)
        neg_x = UnOp(T.i32, op=UnaOp.NEG,
                     arg=Cast(T.i32, to=T.i32, arg=_tid("x")))
        mul = BinOp(T.i32, op=Arith.MUL, lhs=neg_x,
                    rhs=Cast(T.i32, to=T.i32, arg=_tid("y")))
        max_call = Call(T.i32, fn="max", args=[mul, _int(0)], result_ty=T.i32)
        return idx_name, Declare(name=idx_name, gtype=T.i32, init=max_call)

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None or kernel.body is None:
            return prog
        tile_size = param if param is not None else rng.choice(self.sweep_space)

        groups = self._find_scalar_groups(kernel)
        if not groups:
            return prog
        group = rng.choice(groups)
        n_vars = min(tile_size, len(group))
        selected = group[:n_vars]
        elem_type = selected[0].gtype

        taken = all_var_names(kernel)
        tile_name = kx_fresh_name("tile", taken)
        taken.add(tile_name)
        idx_name, idx_decl = self._opaque_zero(taken)
        tile_decl = Declare(name=tile_name, gtype=TArray(elem_type, n_vars),
                            space=MemSpace.GENERIC)

        parent = walk_parent_block(kernel, selected[0])
        if parent is None:
            return prog

        # Build the initialization assignments tile[i+off] = init.
        init_assigns: List[Stmt] = []
        var_to_offset: Dict[str, int] = {}
        for off, decl in enumerate(selected):
            var_to_offset[decl.name] = off
            if decl.init is not None:
                idx_expr = BinOp(T.i32, op=Arith.ADD,
                                 lhs=Var(T.i32, name=idx_name), rhs=_int(off))
                target = Subscript(elem_type,
                                   array=Var(TArray(elem_type, n_vars), name=tile_name),
                                   index=idx_expr)
                init_assigns.append(Assign(target, copy.deepcopy(decl.init)))

        selected_ids = {id(s) for s in selected}
        new_stmts: List[Stmt] = []
        inserted = False
        for s in parent.stmts:
            if id(s) in selected_ids:
                if not inserted:
                    new_stmts.append(tile_decl)
                    new_stmts.append(idx_decl)
                    new_stmts.extend(init_assigns)
                    inserted = True
                continue
            new_stmts.append(s)
        parent.stmts = new_stmts

        for var_name, off in var_to_offset.items():
            def factory(off=off):
                return Subscript(
                    elem_type,
                    array=Var(TArray(elem_type, n_vars), name=tile_name),
                    index=BinOp(T.i32, op=Arith.ADD,
                                lhs=Var(T.i32, name=idx_name), rhs=_int(off)))
            replace_var_everywhere(kernel.body, var_name, factory)
        return prog

    def perturb_opt(self, program, rng, param=None) -> Program:
        return self.perturb(program, rng, param)

    def perturb_deopt(self, program, rng, param=None) -> Program:
        """Register-pressure locals kept live (volatile only when allowed)."""
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None or kernel.body is None:
            return prog
        n_pressure = param if param is not None else rng.choice(self.sweep_space)
        allow_vol = getattr(prog.launch, "allow_volatile", False)
        taken = all_var_names(kernel)

        names: List[str] = []
        decls: List[Stmt] = []
        for i in range(n_pressure):
            vn = kx_fresh_name("reg_pressure", taken)
            taken.add(vn)
            names.append(vn)
            xor = BinOp(T.u32, op=Arith.BXOR,
                        lhs=BinOp(T.u32, op=Arith.BXOR,
                                  lhs=Cast(T.u32, to=T.u32, arg=_tid("x")),
                                  rhs=Cast(T.u32, to=T.u32, arg=_tid("y"))),
                        rhs=_uint(i))
            decls.append(Declare(
                name=vn, gtype=T.u32, space=MemSpace.REGISTER, init=xor,
                quals=[Qual.VOLATILE] if allow_vol else []))
        kernel.body.stmts = decls + list(kernel.body.stmts)

        if not allow_vol and names:
            mixed: Expr = Var(T.u32, name=names[0])
            for vn in names[1:]:
                mixed = BinOp(T.u32, op=Arith.BXOR, lhs=mixed,
                              rhs=Var(T.u32, name=vn))
            keep_value_live(kernel, mixed, base="reg_pressure_sink",
                            allow_volatile=False)
        return prog


class MemPromotion(Mutation):
    """Promote a write-once uniform scalar to __shared__ (single-lane write +
    barrier), or demote a shared variable back to a register local."""

    name = "MEM2_memory_promotion"
    category = "memory"
    sweep_space = [MemSpace.SHARED, MemSpace.REGISTER]

    def _promotable(self, kernel: Func) -> List[Declare]:
        assigned: Set[str] = set()
        for a in assignments(kernel):
            if isinstance(a.target, Var):
                assigned.add(a.target.name)
        for ca in traverse.compound_assignments(kernel):
            if isinstance(ca.target, Var):
                assigned.add(ca.target.name)
        out = []
        for d in declarations(kernel):
            if d.name in assigned:
                continue
            if d.space in (MemSpace.GENERIC, MemSpace.REGISTER, MemSpace.LOCAL):
                out.append(d)
        return out

    @staticmethod
    def _per_thread_names(kernel: Func) -> Set[str]:
        """Vars whose initializer (transitively) reads threadIdx/blockIdx."""
        per_thread: Set[str] = set()
        decls = declarations(kernel)
        changed = True
        while changed:
            changed = False
            for d in decls:
                if d.name in per_thread or d.init is None:
                    continue
                if any(t.source in ("threadIdx", "blockIdx")
                       for t in collect(d.init, lambda n: isinstance(n, ThreadBuiltin))):
                    per_thread.add(d.name)
                    changed = True
                    continue
                if any(v.name in per_thread
                       for v in collect(d.init, lambda n: isinstance(n, Var))):
                    per_thread.add(d.name)
                    changed = True
        return per_thread

    @staticmethod
    def _shared_safe(d: Declare, per_thread: Set[str]) -> bool:
        if d.init is None:
            return False
        if not isinstance(d.gtype, (TInt, TUInt, TFloat)):
            return False
        if d.name in per_thread:
            return False
        if any(t.source in ("threadIdx", "blockIdx")
               for t in collect(d.init, lambda n: isinstance(n, ThreadBuiltin))):
            return False
        if collect(d.init, lambda n: isinstance(n, (WarpShuffle, Atomic))):
            return False
        warp_fns = ("__shfl", "__ballot", "__any", "__all", "__match",
                    "__reduce", "__activemask", "__syncwarp")
        for c in collect(d.init, lambda n: isinstance(n, Call)):
            if any(c.fn.startswith(w) for w in warp_fns):
                return False
        return True

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None or kernel.body is None:
            return prog
        target = param if param is not None else rng.choice(self.sweep_space)
        if target is MemSpace.REGISTER:
            return self.perturb_deopt(prog, rng, param)

        decls = self._promotable(kernel)
        if not decls:
            return prog
        body_stmts = kernel.body.stmts
        per_thread = self._per_thread_names(kernel)
        cands = [d for d in decls
                 if any(s is d for s in body_stmts)
                 and self._shared_safe(d, per_thread)]
        if not cands:
            return prog
        decl = rng.choice(cands)
        idx = next((i for i, s in enumerate(body_stmts) if s is decl), None)
        if idx is None:
            return prog
        # __shared__ T x;  if (threadIdx.x == 0) x = init;  __syncthreads();
        decl.space = MemSpace.SHARED
        if Qual.SHARED_MEM not in decl.quals:
            decl.quals.append(Qual.SHARED_MEM)
        init_val = decl.init
        decl.init = None
        assign = Assign(Var(decl.gtype, name=decl.name, space=MemSpace.SHARED),
                        init_val)
        guarded = If(cond=RelOp(T.u32, op=Cmp.EQ, lhs=_tid("x"), rhs=_uint(0)),
                     then=Block([assign]))
        body_stmts.insert(idx + 1, guarded)
        body_stmts.insert(idx + 2, Barrier())
        return prog

    def perturb_opt(self, program, rng, param=None) -> Program:
        return self.perturb(program, rng, MemSpace.SHARED)

    def perturb_deopt(self, program, rng, param=None) -> Program:
        """Demote a shared variable to a plain local.

        Only a shared array whose every access is indexed by the *same
        thread's* lane id is safe to demote: then each thread owns its slot
        and a per-thread local is observationally identical.  Demoting a
        cross-thread-shared value would leave readers of another lane's slot
        with uninitialized local memory (nondeterministic), so those are
        never demoted."""
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None:
            return prog
        shared = [d for d in declarations(kernel) if d.space is MemSpace.SHARED]
        safe = []
        for d in shared:
            accesses = [a for a in subscripts(kernel)
                        if isinstance(a.array, Var) and a.array.name == d.name]
            if not accesses:
                # a scalar shared var (declared, no subscript) is written by
                # one lane and read by all -> NOT safe to demote.
                continue
            ok = True
            for a in accesses:
                idx = a.index
                if not (isinstance(idx, ThreadBuiltin) and idx.source == "threadIdx"):
                    ok = False
                    break
            if ok:
                safe.append(d)
        if not safe:
            return prog
        decl = rng.choice(safe)
        decl.space = MemSpace.GENERIC
        if Qual.SHARED_MEM in decl.quals:
            decl.quals.remove(Qual.SHARED_MEM)
        return prog

    def scan(self, program, rng) -> List[Program]:
        return [self.perturb(clone_program(program), rng, p)
                for p in self.sweep_space]


# ---------------------------------------------------------------------------
# MEM3: stride alteration
# ---------------------------------------------------------------------------

class MemStrideAlteration(Mutation):
    """Set the stride annotation on global-memory accesses (the emitter wraps
    strided input reads in a bounds-safe modulo)."""

    name = "MEM3_stride_alteration"
    category = "memory"
    sweep_space = [1, 2, 4, 8, 32]

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None or kernel.body is None:
            return prog
        stride = param if param is not None else rng.choice(self.sweep_space)
        # Only `input` reads honor the stride annotation (the emitter wraps
        # those in a bounds-safe modulo); setting stride on other arrays would
        # be a silent no-op.
        for acc in subscripts(kernel):
            if acc.locked:
                continue
            if not (isinstance(acc.array, Var) and acc.array.name == "input"):
                continue
            acc.stride = stride
        return prog

    def perturb_opt(self, program, rng, param=None) -> Program:
        return self.perturb(program, rng, 1)

    def perturb_deopt(self, program, rng, param=None) -> Program:
        return self.perturb(program, rng, 32)


# ---------------------------------------------------------------------------
# MEM4: volatile qualifier (disabled by default -- volatile+atomicAdd
# interaction breaks compilation on a subset of seeds; opt-in via the
# pipeline's disabled-operator registry)
# ---------------------------------------------------------------------------

class MemVolatileQualifier(Mutation):
    """Add volatile to N variable declarations (never to atomic targets)."""

    name = "MEM4_volatile_qualifier"
    category = "memory"
    sweep_space = [1, 2, 4]
    enabled = False  # known-bad interaction with atomicAdd; opt-in only

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None:
            return prog
        count = param if param is not None else rng.choice(self.sweep_space)
        atomic_targets: Set[str] = set()
        for a in atomics(kernel):
            if isinstance(a.addr, Subscript) and isinstance(a.addr.array, Var):
                atomic_targets.add(a.addr.array.name)
            elif isinstance(a.addr, Var):
                atomic_targets.add(a.addr.name)
        cands = [d for d in declarations(kernel)
                 if Qual.VOLATILE not in d.quals and d.name not in atomic_targets]
        if not cands:
            return prog
        for d in rng.sample(cands, min(count, len(cands))):
            d.quals.append(Qual.VOLATILE)
        return prog

    def perturb_opt(self, program, rng, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None:
            return prog
        for d in declarations(kernel):
            if Qual.VOLATILE in d.quals:
                d.quals.remove(Qual.VOLATILE)
        return prog

    def perturb_deopt(self, program, rng, param=None) -> Program:
        return self.perturb(program, rng, param)


# ---------------------------------------------------------------------------
# MEM5: __restrict__
# ---------------------------------------------------------------------------

class MemRestrictQualifier(Mutation):
    """Add __restrict__ to pointer parameters (opt) / remove it (deopt)."""

    name = "MEM5_restrict_qualifier"
    category = "memory"
    sweep_space = [1, 2, 4]

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None:
            return prog
        count = param if param is not None else rng.choice(self.sweep_space)
        cands = [p for p in kernel.params
                 if isinstance(p.gtype, TPointer) and Qual.RESTRICT not in p.quals]
        if not cands:
            return prog
        for p in rng.sample(cands, min(count, len(cands))):
            p.quals.append(Qual.RESTRICT)
        return prog

    def perturb_opt(self, program, rng, param=None) -> Program:
        return self.perturb(program, rng, param)

    def perturb_deopt(self, program, rng, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None:
            return prog
        for p in kernel.params:
            if Qual.RESTRICT in p.quals:
                p.quals.remove(Qual.RESTRICT)
        return prog


# ---------------------------------------------------------------------------
# MEM6: padding
# ---------------------------------------------------------------------------

class MemPadding(Mutation):
    """Grow a shared array to a bank-conflict-avoiding size (never shrinks)."""

    name = "MEM6_padding"
    category = "memory"

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None:
            return prog
        arrays = [d for d in declarations(kernel) if isinstance(d.gtype, TArray)]
        if not arrays:
            return prog
        decl = rng.choice(arrays)
        at = decl.gtype
        size = at.count
        if is_pow2(size):
            decl.gtype = TArray(at.elem, size + 1)
        else:
            p2 = 1
            while p2 < size:
                p2 <<= 1
            decl.gtype = TArray(at.elem, p2 + 1)
        return prog


# ---------------------------------------------------------------------------
# EXT: access pattern scan
# ---------------------------------------------------------------------------



class ExtAccessPatternSweep(Mutation):
    """Rewrite input reads into coalesced / misaligned / column-major-ish
    shapes (all bounds-wrapped)."""

    name = "EXT_access_pattern_sweep"
    category = "memory"
    sweep_space = ["coalesced", "misaligned", "column_major"]

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None:
            return prog
        pattern = param if param is not None else rng.choice(self.sweep_space)
        analyzer = AccessPatternAnalyzer(kernel)
        reads = [r.subscript for r in analyzer.records()
                 if r.array_name == "input"
                 and not r.subscript.locked
                 and r.pattern is not AccessPattern.REDUCTION]
        if not reads:
            return prog

        for acc in reads:
            idx = acc.index
            if pattern == "coalesced":
                acc.stride = 1
            elif pattern == "misaligned":
                # index + 1, but bounds-wrapped: the emitter only wraps
                # strided input reads, so an unwrapped +1 would read index n.
                n_u = Cast(T.u32, to=T.u32, arg=Var(T.i32, name="n"))
                acc.index = SafeBin(T.u32, op=SafeOp.SMOD,
                                    lhs=BinOp(T.u32, op=Arith.ADD,
                                              lhs=copy.deepcopy(idx), rhs=_uint(1)),
                                    rhs=n_u)
            else:  # column_major-ish: swap-ish stride pattern
                acc.stride = 32
        return prog

    def perturb_opt(self, program, rng, param=None) -> Program:
        return self.perturb(program, rng, "coalesced")

    def perturb_deopt(self, program, rng, param=None) -> Program:
        return self.perturb(program, rng, "column_major")


# ---------------------------------------------------------------------------
# EXT: shared staging
# ---------------------------------------------------------------------------

class ExtSharedStaging(Mutation):
    """Introduce a shared staging buffer in a global-reading loop:

    __shared__ T smem[N]; smem[tid.x] = arr[i % n]; __syncthreads(); use smem.
    """

    name = "EXT_shared_staging"
    category = "memory"

    @staticmethod
    def _stageable(kernel: Func) -> Set[str]:
        return {p.name for p in kernel.params
                if isinstance(p.gtype, TPointer) and p.name != "output"}

    def _loops(self, kernel: Func) -> List[For]:
        names = self._stageable(kernel)
        out = []
        for loop in mutable_for_loops(kernel):
            for acc in subscripts(loop):
                if isinstance(acc.array, Var) and acc.array.name in names:
                    out.append(loop)
                    break
        return out

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None or kernel.body is None:
            return prog
        candidates = [l for l in self._loops(kernel)
                      if barrier_ok_in_loop(kernel.body, l)
                      and not any(d.name.startswith("smem_")
                                  for d in declarations(l))]
        if not candidates:
            return prog
        loop = rng.choice(candidates)
        names = self._stageable(kernel)
        # Only stage u32 buffers (the generated kernels' input is u32); a
        # mismatched element type would corrupt or fail to assemble.
        elem_of: Dict[str, GType] = {}
        for p in kernel.params:
            if isinstance(p.gtype, TPointer):
                elem_of[p.name] = p.gtype.base
        target_name = None
        elem_ty: Optional[GType] = None
        for acc in subscripts(loop):
            if isinstance(acc.array, Var) and acc.array.name in names:
                base = elem_of.get(acc.array.name)
                if base is not None and base.is_int and base.width == 4:
                    target_name = acc.array.name
                    elem_ty = base
                    break
        if target_name is None or elem_ty is None:
            return prog

        taken = all_var_names(kernel)
        block_size = prog.launch.threads_per_block
        smem_name = kx_fresh_name(f"smem_{target_name}", taken)
        arr_ty = TArray(elem_ty, block_size)
        smem_decl = Declare(name=smem_name, gtype=arr_ty,
                            space=MemSpace.SHARED, quals=[Qual.SHARED_MEM])
        smem_var = Var(arr_ty, name=smem_name, space=MemSpace.SHARED)
        load = Assign(
            Subscript(elem_ty, array=smem_var, index=_tid("x")),
            Subscript(elem_ty,
                      array=Var(TPointer(elem_ty), name=target_name),
                      index=SafeBin(T.u32, op=SafeOp.SMOD,
                                    lhs=Var(T.i32, name=loop.var),
                                    rhs=Cast(T.u32, to=T.u32,
                                             arg=Var(T.i32, name="n")))))

        # Repoint reads at the staging buffer.
        # Skip Subscripts that are direct arguments to global-pointer cache
        # intrinsics (__ldg / __ldcg / __stcg / ...): those require a global
        # address and would not assemble (or would alias) pointing at __shared__.
        global_arg_ids: Set[int] = set()
        for call in collect(loop.body, lambda n: isinstance(n, Call)):
            if call.fn.startswith(("__ld", "__st")):
                for arg in call.args:
                    if isinstance(arg, Subscript):
                        global_arg_ids.add(id(arg))
        for acc in subscripts(loop.body):
            if id(acc) in global_arg_ids:
                continue
            if isinstance(acc.array, Var) and acc.array.name == target_name:
                acc.array = Var(arr_ty, name=smem_name, space=MemSpace.SHARED)
                acc.index = _tid("x")

        loop.body.stmts = [smem_decl, load, Barrier()] + list(loop.body.stmts) + [Barrier()]
        return prog


# ---------------------------------------------------------------------------
# EXT: bank conflict inject/remove
# ---------------------------------------------------------------------------

class ExtBankConflict(Mutation):
    """Multiply shared-memory READ indices by a warp-size factor (conflict).

    Only reads are remapped: multiplying a write index by a non-coprime
    factor would map two lanes to one slot (a write-write race).  Read-side
    remapping is race-free and still forces the bank-conflict access shape."""

    name = "EXT_bank_conflict"
    category = "memory"

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None:
            return prog
        shared_arrays = {d.name: d.gtype for d in declarations(kernel)
                         if d.space is MemSpace.SHARED
                         and isinstance(d.gtype, TArray)}
        if not shared_arrays:
            return prog
        # Arrays that are write targets (Assign/CompoundAssign targets into
        # them) are excluded: remapping a write index could put two lanes in
        # one slot.  Pure read-only arrays are safe to remap.
        written: Set[str] = set()
        for stmt in list(assignments(kernel)) + list(traverse.compound_assignments(kernel)):
            t = stmt.target
            if isinstance(t, Subscript) and isinstance(t.array, Var):
                written.add(t.array.name)
        read_only = [n for n in shared_arrays if n not in written]
        if not read_only:
            return prog
        name = rng.choice(sorted(read_only))
        size = shared_arrays[name].count
        accesses = [a for a in subscripts(kernel)
                    if isinstance(a.array, Var) and a.array.name == name]
        if not accesses:
            return prog
        factor = rng.choice([2, 4, 8, 16, 32])
        for acc in accesses:
            acc.index = SafeBin(T.u32, op=SafeOp.SMOD,
                                lhs=BinOp(T.u32, op=Arith.MUL,
                                          lhs=copy.deepcopy(acc.index),
                                          rhs=_uint(factor)),
                                rhs=_uint(size))
        return prog


# ---------------------------------------------------------------------------
# EXT: CUDA cache-hint scan (ld/st policy intrinsics)
# ---------------------------------------------------------------------------

class ExtCacheHintSweep(Mutation):
    """Inject a guarded __ldcs/__stcs-style load/store pair (CUDA only)."""

    name = "EXT_cuda_cache_hint_sweep"
    category = "memory"
    sweep_space = ["__ldg", "__ldcg", "__ldca", "__ldcs", "__ldlu"]

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        if not kx_is_cuda(prog):
            return prog
        kernel = kernel_of(prog)
        if kernel is None or kernel.body is None:
            return prog
        hint = param if param is not None else rng.choice(self.sweep_space)
        if hint == "__ldlu" and cuda_sm_version(prog) < 80:
            hint = "__ldg"
        taken = all_var_names(kernel)
        tmp = kx_fresh_name("cache_hint", taken)
        n_u = Cast(T.u32, to=T.u32, arg=Var(T.i32, name="n"))
        idx = SafeBin(T.u32, op=SafeOp.SMOD,
                      lhs=Var(T.u32, name="tid"), rhs=n_u)
        load = Call(T.u32, fn=hint,
                    args=[BinOp(TPointer(T.u32), op=Arith.ADD,
                                lhs=Var(TPointer(T.u32), name="input"),
                                rhs=idx)],
                    result_ty=T.u32)
        decl = Declare(name=tmp, gtype=T.u32, init=load)
        # Insert just before the trailing output region, where tid/n are in
        # kx_scope, then keep the loaded value live through the sink machinery.
        stmts = kernel.body.stmts
        pos = len(stmts)
        for i in range(len(stmts) - 1, -1, -1):
            if isinstance(stmts[i], If):
                pos = i
                break
        stmts.insert(pos, decl)
        keep_value_live(kernel, Var(T.u32, name=tmp),
                        base="cache_hint",
                        allow_volatile=getattr(prog.launch, "allow_volatile", False))
        return prog


# ---------------------------------------------------------------------------
# EXT: warp shuffle <-> shared memory swap
# ---------------------------------------------------------------------------

class ExtShuffleSharedSwap(Mutation):
    """Replace a __shfl_down_sync reduction tail with a shared-memory
    exchange (deterministic, barrier-bracketed)."""

    name = "EXT_shuffle_shared_swap"
    category = "memory"

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None or kernel.body is None:
            return prog
        shuffles = [w for w in warp_shuffles(kernel)
                    if w.kind == "shfl_down_sync"]
        if not shuffles:
            return prog
        taken = all_var_names(kernel)
        smem_name = kx_fresh_name("shswap", taken)
        size = prog.launch.threads_per_block
        arr_ty = TArray(T.u32, size)
        decl = Declare(name=smem_name, gtype=arr_ty,
                       space=MemSpace.SHARED, quals=[Qual.SHARED_MEM])
        smem = Var(arr_ty, name=smem_name, space=MemSpace.SHARED)

        # Stage each shuffle's value in shared memory and read the neighbor.
        w = shuffles[0]
        parent_block = None
        parent_idx = None
        for block in blocks(kernel):
            for i, s in enumerate(block.stmts):
                if isinstance(s, Declare) and s.init is not None:
                    if collect(s.init, lambda n: n is w):
                        parent_block, parent_idx = block, i
                        break
            if parent_block is not None:
                break
        if parent_block is None:
            return prog
        # The inserted barriers must be reached by every thread: only proceed
        # when the shuffle's block is at divergent depth 0 (not inside any
        # if/switch/while or non-uniform for).
        if not traverse.block_is_uniform(kernel.body, parent_block):
            return prog
        if traverse.kx_loop_body_escapes(parent_block):
            return prog

        lane = _tid("x")
        delta = w.delta if w.delta is not None else _int(1)
        nbr = SafeBin(T.u32, op=SafeOp.SMOD,
                      lhs=BinOp(T.u32, op=Arith.ADD, lhs=lane,
                                rhs=Cast(T.u32, to=T.u32, arg=delta)),
                      rhs=_uint(size))
        staging = [
            decl,
            Assign(Subscript(T.u32, array=smem, index=lane),
                   w.value if w.value is not None else _uint(0)),
            Barrier(),
        ]
        replacement = Subscript(T.u32, array=smem, index=nbr)
        # Replace the shuffle call with the shared read.
        for node in traverse.traverse(kernel):
            for f in ("init", "value", "lhs", "rhs"):
                cur = getattr(node, f, None)
                if cur is w:
                    setattr(node, f, replacement)
        for j, s in enumerate(staging):
            parent_block.stmts.insert(parent_idx + j, s)
        # Barrier after the read region to protect reuse.
        parent_block.stmts.insert(parent_idx + len(staging) + 1, Barrier())
        return prog


# ---------------------------------------------------------------------------
# EXT: vectorization width
# ---------------------------------------------------------------------------

class ExtVectorizationWidth(Mutation):
    """Widen a scalar input read into a folded read of N consecutive lanes
    (vector-load pressure on the memory pipeline)."""

    name = "EXT_vectorization_width"
    category = "memory"
    sweep_space = [1, 2, 4]

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None:
            return prog
        width = param if param is not None else rng.choice(self.sweep_space)
        if width <= 1:
            return prog
        # Find assignments/declarations whose value is exactly one input read.
        sites: List = []
        for node in list(assignments(kernel)) + list(declarations(kernel)):
            val = getattr(node, "value", None) or getattr(node, "init", None)
            if (isinstance(val, Subscript)
                    and isinstance(val.array, Var) and val.array.name == "input"
                    and not val.locked):
                sites.append(node)
        if not sites:
            return prog
        site = rng.choice(sites)
        is_assign = isinstance(site, Assign)
        read = site.value if is_assign else site.init
        base = copy.deepcopy(read.index)
        n_u = Cast(T.u32, to=T.u32, arg=Var(T.i32, name="n"))
        combined: Expr = Subscript(T.u32,
                                   array=Var(TPointer(T.u32), name="input"),
                                   index=base)
        for k in range(1, width):
            nxt_idx = SafeBin(T.u32, op=SafeOp.SMOD,
                              lhs=BinOp(T.u32, op=Arith.ADD,
                                        lhs=copy.deepcopy(base), rhs=_uint(k)),
                              rhs=n_u)
            combined = BinOp(T.u32, op=Arith.BXOR, lhs=combined,
                             rhs=Subscript(T.u32,
                                           array=Var(TPointer(T.u32), name="input"),
                                           index=nxt_idx))
        if is_assign:
            site.value = combined
        else:
            site.init = combined
        return prog


# ---------------------------------------------------------------------------
# EXT: prefetch distance
# ---------------------------------------------------------------------------

class ExtPrefetchDistance(Mutation):
    """Hoist an input read for iteration i+K ahead of the loop body
    (software prefetch shape)."""

    name = "EXT_prefetch_distance"
    category = "memory"
    sweep_space = [0, 1, 2, 4]

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None or kernel.body is None:
            return prog
        dist = param if param is not None else rng.choice(self.sweep_space)
        if dist == 0:
            return prog
        loops = [l for l in mutable_for_loops(kernel)
                 if subscripts(l)]
        if not loops:
            return prog
        loop = rng.choice(loops)
        reads = [a for a in subscripts(loop.body)
                 if isinstance(a.array, Var) and a.array.name == "input"]
        if not reads:
            return prog
        taken = all_var_names(kernel)
        # Fold the look-ahead load into the observable _fv so it survives DCE.
        # Skip when the kernel has no _fv sink to pin the load live.
        if "_fv" not in taken:
            return prog
        pf = kx_fresh_name("pf", taken)
        n_u = Cast(T.u32, to=T.u32, arg=Var(T.i32, name="n"))
        pf_idx = SafeBin(T.u32, op=SafeOp.SMOD,
                         lhs=BinOp(T.u32, op=Arith.ADD,
                                   lhs=Cast(T.u32, to=T.u32,
                                            arg=Var(T.i32, name=loop.var)),
                                   rhs=_uint(dist)),
                         rhs=n_u)
        decl = Declare(name=pf, gtype=T.u32,
                       init=Subscript(T.u32, array=Var(TPointer(T.u32), name="input"),
                                      index=pf_idx))
        loop.body.stmts.insert(0, decl)
        # Consume the prefetched value so the look-ahead load survives DCE:
        # XOR it into _fv (the observable sink).  Routing it through (pf & 0)
        # would fold to a constant 0 and let the target compiler drop the load,
        # making the prefetch a no-op.
        fold = Assign(Var(T.u32, name="_fv"),
                      BinOp(T.u32, op=Arith.BXOR, lhs=Var(T.u32, name="_fv"),
                            rhs=Var(T.u32, name=pf)))
        loop.body.stmts.insert(1, fold)
        return prog


# ---------------------------------------------------------------------------
# EXT: kernel fusion/fission (single-kernel shared handoff)
# ---------------------------------------------------------------------------

class ExtKernelFusionFission(Mutation):
    """Model a fission boundary inside one kernel: a deterministic shared
    handoff splits the body into two phases (barrier-separated)."""

    name = "EXT_kernel_fusion_fission"
    category = "memory"

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None or kernel.body is None:
            return prog
        if prog.launch.kx_is_2d:
            return prog
        taken = all_var_names(kernel)
        if "tid" not in taken or "output" not in taken:
            return prog
        smem_name = kx_fresh_name("fusion_smem", taken)
        size = prog.launch.threads_per_block
        arr_ty = TArray(T.u32, size)
        lane = _tid("x")
        n_u = Cast(T.u32, to=T.u32, arg=Var(T.i32, name="n"))
        handoff = [
            Declare(name=smem_name, gtype=arr_ty,
                    space=MemSpace.SHARED, quals=[Qual.SHARED_MEM]),
            Assign(Subscript(T.u32, array=Var(arr_ty, name=smem_name,
                                              space=MemSpace.SHARED),
                             index=lane),
                   SafeBin(T.u32, op=SafeOp.SMOD,
                           lhs=Subscript(T.u32,
                                         array=Var(TPointer(T.u32), name="input"),
                                         index=SafeBin(T.u32, op=SafeOp.SMOD,
                                                       lhs=Var(T.u32, name="tid"),
                                                       rhs=n_u)),
                           rhs=_uint(size))),
            Barrier(),
            Assign(Var(T.u32, name="_fv"),
                   BinOp(T.u32, op=Arith.BXOR,
                         lhs=Var(T.u32, name="_fv"),
                         rhs=Subscript(T.u32,
                                       array=Var(arr_ty, name=smem_name,
                                                 space=MemSpace.SHARED),
                                       index=SafeBin(T.u32, op=SafeOp.SMOD,
                                                     lhs=BinOp(T.u32, op=Arith.BXOR,
                                                               lhs=lane,
                                                               rhs=_uint(1)),
                                                     rhs=_uint(size))))),
            Barrier(),
        ]
        if not traverse.kernel_has_fv_write(kernel):
            return prog
        # Insert before the final output write If.
        stmts = kernel.body.stmts
        pos = len(stmts)
        for i in range(len(stmts) - 1, -1, -1):
            if isinstance(stmts[i], If):
                pos = i
                break
        for j, s in enumerate(handoff):
            stmts.insert(pos + j, s)
        return prog
