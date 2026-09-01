"""Stress-family templates: targeted compiler-pressure microkernels.

Each template is built from IR (not raw text), so mutations can still rewrite
them.  All are deterministic and barrier-safe.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from generator_ir.ir.typedecl import (
    T, MemSpace, Qual, TArray,
)
from generator_ir.ir.operators import Arith, Cmp, IncDec, SafeOp
from generator_ir.ir.nodes import (
    Expr, Lit, Var, ThreadBuiltin, BinOp, RelOp, Cast, SafeBin, Subscript,
)
from generator_ir.ir.statements import (
    Stmt, Assign, Declare, Block, If, For, While, DoWhile, Continue,
    IncDecStmt, Barrier, Fence, Switch, Case,
)
from generator_ir.templates.skeleton import KernelTemplate
from generator_ir.gen.context import GenContext
from generator_ir.gen.expr_builder import ExprGen


def _u(v: Expr) -> Cast:
    return Cast(T.u32, to=T.u32, arg=v)


def _mod(e: Expr, m: Expr) -> SafeBin:
    return SafeBin(T.u32, op=SafeOp.SMOD, lhs=e, rhs=m)


def _tid() -> ThreadBuiltin:
    return ThreadBuiltin(T.u32, axis="x", source="threadIdx")


class _SharedTemplate(KernelTemplate):
    """Base for 1D templates with a shared scratch array."""

    _block_size = 256

    def preferred_block(self) -> Tuple[int, int, int]:
        return (self._block_size, 1, 1)

    def kx_required_shared_mem(self) -> int:
        return self._block_size * 4

    # Shared helpers -----------------------------------------------------

    def _scope(self, ctx: GenContext):
        tid = ctx.kx_scope.lookup("tid")
        n_var = ctx.kx_scope.lookup("n")
        return (tid, n_var, ctx.kx_scope.lookup("output"), ctx.kx_scope.lookup("input"),
                _u(n_var))

    def _declare_shared(self, ctx: GenContext, prefix: str, size: int):
        name = ctx.kx_fresh_name(prefix)
        var = Var(TArray(T.u32, size), name=name, space=MemSpace.SHARED)
        ctx.kx_declare(var)
        decl = Declare(name=name, gtype=TArray(T.u32, size),
                       space=MemSpace.SHARED, quals=[Qual.SHARED_MEM])
        return name, var, decl

    def _flat_sink(self, ctx: GenContext, pool_snapshot: Set[str],
                   value: Expr) -> List[Stmt]:
        """observable_fold result routing: sample the sink only from the
        pre-template pool (never this template's own scratch)."""
        saved = ctx._visible_names
        ctx._visible_names = pool_snapshot
        try:
            sink, sink_decl = ctx.kx_sample_or_create(T.u32, for_write=True)
        finally:
            ctx._visible_names = saved
        stmts: List[Stmt] = []
        if sink_decl is not None:
            stmts.append(sink_decl)
        stmts.append(Assign(sink, value))
        return stmts


# ---------------------------------------------------------------------------
# Bank conflict
# ---------------------------------------------------------------------------



class BankConflictTemplate(_SharedTemplate):
    """Shared-memory layout stress: stage `passes` block-sized slices, then
    re-read through a guaranteed bank-conflicting transpose index
    (lane*32 + warp) with a read/modify/write on the same slot."""

    def __init__(self) -> None:
        self._block_size = 256
        self._passes = 4

    @property
    def name(self) -> str:
        return "smem_bank_stress"

    @property
    def features(self) -> Set[str]:
        return {"shared_mem", "sync", "strided_access"}

    def fetch_params(self) -> Dict[str, Any]:
        return {"block_size": self._block_size, "passes": self._passes}

    def kx_required_shared_mem(self) -> int:
        # `passes` full block-sized slices.
        return self._block_size * self._passes * 4

    def _stage_passes(self, ctx: GenContext, smem: Var, tid: Expr, n_u: Expr,
                      input_var: Var) -> Stmt:
        """for p in [0, passes): smem[p*bs + threadIdx.x] =
        input[(tid + p*17) % n] ^ (p * 0x9E3779B9)."""
        p_name = ctx.kx_fresh_name("p")
        p_u = Cast(T.u32, to=T.u32, arg=Var(T.i32, name=p_name))
        base_idx = BinOp(T.u32, op=Arith.MUL, lhs=p_u,
                         rhs=Lit(T.u32, value=self._block_size))
        src_idx = _mod(BinOp(T.u32, op=Arith.ADD, lhs=tid,
                             rhs=BinOp(T.u32, op=Arith.MUL, lhs=p_u,
                                       rhs=Lit(T.u32, value=17))), n_u)
        src_val = BinOp(T.u32, op=Arith.BXOR,
                        lhs=Subscript(T.u32, array=input_var, index=src_idx),
                        rhs=BinOp(T.u32, op=Arith.MUL, lhs=p_u,
                                  rhs=Lit(T.u32, value=0x9E3779B9)))
        slot = BinOp(T.u32, op=Arith.ADD, lhs=base_idx, rhs=_tid())
        return For(var=p_name, start=Lit(T.i32, value=0),
                   end=Lit(T.i32, value=self._passes),
                   step=Lit(T.i32, value=1),
                   body=Block([Assign(Subscript(T.u32, array=smem, index=slot),
                                      src_val)]))

    def build_body(self, ctx: GenContext) -> List[Stmt]:
        tid, n_var, output_var, input_var, n_u = self._scope(ctx)
        bs = self._block_size
        flat = self.kx_flat_mode(ctx)
        # Snapshot the shared pool BEFORE declaring template scratch so the
        # flat result sink combines with the generated pool, never with the
        # template's own lane/warp/mix temporaries.
        pool_snapshot = ctx.kx_visible_declared_names() if flat else None
        name, smem, decl = self._declare_shared(ctx, "bc_smem",
                                                bs * self._passes)

        stmts: List[Stmt] = [decl]
        c32 = Lit(T.u32, value=32)
        if flat:
            # lane / warp sampled from the shared pool: they may alias each
            # other, but both stay block-local (lane in [0,31], warp in
            # [0,7], or aliased <= 7), so lane*32 + warp is always < slots.
            lane_init = _mod(_tid(), c32)
            lane, lane_decl = ctx.kx_sample_or_create(T.u32, for_write=True,
                                                   init=lane_init)
            stmts.append(lane_decl if lane_decl is not None
                         else Assign(lane, lane_init))
            warp_init = BinOp(T.u32, op=Arith.DIV, lhs=_tid(), rhs=c32)
            warp, warp_decl = ctx.kx_sample_or_create(T.u32, for_write=True,
                                                   init=warp_init)
            stmts.append(warp_decl if warp_decl is not None
                         else Assign(warp, warp_init))
        else:
            lane_name = ctx.kx_fresh_name("lane")
            warp_name = ctx.kx_fresh_name("warp")
            lane = Var(T.u32, name=lane_name)
            warp = Var(T.u32, name=warp_name)
            ctx.kx_declare(lane)
            ctx.kx_declare(warp)
            # Block-local warp index: scratch is a per-block __shared__ array,
            # so the transpose index must be block-local.
            stmts.append(Declare(name=lane_name, gtype=T.u32,
                                 init=_mod(_tid(), c32)))
            stmts.append(Declare(name=warp_name, gtype=T.u32,
                                 init=BinOp(T.u32, op=Arith.DIV,
                                            lhs=_tid(), rhs=c32)))

        # Stage the passes; keep the data live across a fence + barrier.
        stmts.append(self._stage_passes(ctx, smem, tid, n_u, input_var))
        stmts.append(Fence(kx_scope="block"))
        stmts.append(Barrier())

        # Guaranteed conflict: every lane of a warp hits the same bank
        # ((lane*32 + warp) % 32 == warp).  `mix` is a DEDICATED fresh var
        # (never sampled): the RMW below recomputes lane*32 + warp for its
        # index, so an aliased mix could clobber the index and send the
        # shared write out of bounds.
        conflict_idx = BinOp(T.u32, op=Arith.ADD,
                             lhs=BinOp(T.u32, op=Arith.MUL, lhs=lane, rhs=c32),
                             rhs=warp)
        mix_expr = BinOp(
            T.u32, op=Arith.BXOR,
            lhs=Subscript(T.u32, array=smem, index=conflict_idx),
            rhs=BinOp(T.u32, op=Arith.BXOR,
                      lhs=_mod(BinOp(T.u32, op=Arith.ADD, lhs=tid,
                                     rhs=Lit(T.u32, value=13)), n_u),
                      rhs=BinOp(T.u32, op=Arith.BXOR, lhs=lane, rhs=warp)))
        mix_name = ctx.kx_fresh_name("bc_mix")
        mix = Var(T.u32, name=mix_name)
        ctx.kx_declare(mix)
        stmts.append(Declare(name=mix_name, gtype=T.u32, init=mix_expr))
        # Read/modify/write the same conflictful slot to exercise both paths.
        stmts.append(Assign(Subscript(T.u32, array=smem, index=conflict_idx),
                            Var(T.u32, name=mix_name)))

        if flat:
            stmts.extend(self._flat_sink(ctx, pool_snapshot,
                                         Var(T.u32, name=mix_name)))
            # Make the scratch safe to re-stage next repeat-loop iteration.
            stmts.append(Barrier())
            return stmts
        stmts.append(If(cond=RelOp(T.boolean, op=Cmp.LT, lhs=tid, rhs=n_u),
                        then=Block([Assign(
                            Subscript(T.u32, array=output_var, index=tid),
                            Var(T.u32, name=mix_name))])))
        stmts.append(Barrier())
        return stmts


# ---------------------------------------------------------------------------
# Barrier pressure
# ---------------------------------------------------------------------------



class BarrierPressureTemplate(_SharedTemplate):
    """Live-range stress: a chain of live scalars built before a barrier,
    anchored through shared memory, consumed (and extended) after it."""

    def __init__(self) -> None:
        self._block_size = 256
        self._live_count = 6

    @property
    def name(self) -> str:
        return "sync_span_live"

    @property
    def features(self) -> Set[str]:
        return {"shared_mem", "sync", "strided_access"}

    def fetch_params(self) -> Dict[str, Any]:
        return {"block_size": self._block_size,
                "live_count": self._live_count}

    def build_body(self, ctx: GenContext) -> List[Stmt]:
        tid, n_var, output_var, input_var, n_u = self._scope(ctx)
        bs = self._block_size
        flat = self.kx_flat_mode(ctx)
        # Snapshot the pre-template pool so the flat sink combines with the
        # generated pool rather than the template's own live_*/mix chain.
        pool_snapshot = ctx.kx_visible_declared_names() if flat else None
        name, smem, decl = self._declare_shared(ctx, "bp_smem", bs)
        lane = _tid()
        bs_c = Lit(T.u32, value=bs)
        expr_gen = ExprGen(ctx)

        stmts: List[Stmt] = [decl]
        # Chained live scalars from distinct bounded reads: live_i folds the
        # previous value (alternating +/^) with input[(tid + 3i + 1) % n].
        live: List[Var] = []
        prev: Optional[Expr] = None
        for i in range(self._live_count):
            vn = ctx.kx_fresh_name("live")
            v = Var(T.u32, name=vn)
            ctx.kx_declare(v)
            idx = _mod(BinOp(T.u32, op=Arith.ADD, lhs=tid,
                             rhs=Lit(T.u32, value=3 * i + 1)), n_u)
            load: Expr = Subscript(T.u32, array=input_var, index=idx)
            if prev is not None:
                load = BinOp(T.u32, op=Arith.ADD if (i & 1) else Arith.BXOR,
                             lhs=prev, rhs=load)
            stmts.append(Declare(name=vn, gtype=T.u32, init=load))
            live.append(v)
            prev = v

        # Anchor one chained value in shared memory before the barrier.
        stmts.append(Assign(Subscript(T.u32, array=smem, index=lane),
                            BinOp(T.u32, op=Arith.BXOR,
                                  lhs=live[0], rhs=live[1])))
        stmts.append(Fence(kx_scope="block"))
        stmts.append(Barrier())

        # Consume after the barrier: block-local neighbour + live[2:].
        neighbor = Subscript(T.u32, array=smem, index=_mod(
            BinOp(T.u32, op=Arith.ADD, lhs=lane, rhs=Lit(T.u32, value=1)),
            bs_c))
        mix_name = ctx.kx_fresh_name("mix")
        mix = Var(T.u32, name=mix_name)
        ctx.kx_declare(mix)
        mix_expr: Expr = neighbor
        for v in live[2:]:
            mix_expr = BinOp(T.u32, op=Arith.BXOR, lhs=mix_expr, rhs=v)
        stmts.append(Declare(name=mix_name, gtype=T.u32, init=mix_expr))

        # A short post-barrier loop keeps the live range active a bit longer.
        r_name = ctx.kx_fresh_name("r")
        r_u = Cast(T.u32, to=T.u32, arg=Var(T.i32, name=r_name))
        r_idx = _mod(BinOp(T.u32, op=Arith.ADD, lhs=tid,
                           rhs=BinOp(T.u32, op=Arith.MUL, lhs=r_u,
                                     rhs=Lit(T.u32, value=17))), n_u)
        stmts.append(For(var=r_name, start=Lit(T.i32, value=0),
                         end=Lit(T.i32, value=2), step=Lit(T.i32, value=1),
                         body=Block([Assign(mix, BinOp(
                             T.u32, op=Arith.BXOR, lhs=mix,
                             rhs=Subscript(T.u32, array=input_var,
                                           index=r_idx)))])))

        if flat:
            result = BinOp(T.u32, op=Arith.BXOR, lhs=mix,
                           rhs=expr_gen.kx_generate(T.u32))
            stmts.extend(self._flat_sink(ctx, pool_snapshot, result))
            # Make scratch safe to reuse on the next repeat-loop iteration.
            stmts.append(Barrier())
            return stmts
        stmts.append(If(cond=RelOp(T.boolean, op=Cmp.LT, lhs=tid, rhs=n_u),
                        then=Block([Assign(
                            Subscript(T.u32, array=output_var, index=tid),
                            mix)])))
        stmts.append(Barrier())
        return stmts


# ---------------------------------------------------------------------------
# Switch divergence
# ---------------------------------------------------------------------------



class SwitchDivergenceTemplate(_SharedTemplate):
    """Per-thread switch key, distinct loop shapes per arm (while, do-while,
    for, fence-heavy, while+continue), barrier reconverge, cross-lane mix."""

    def __init__(self) -> None:
        self._block_size = 256
        self._cases = 5

    @property
    def name(self) -> str:
        return "cf_switch_tree"

    @property
    def features(self) -> Set[str]:
        return {"shared_mem", "sync", "strided_access"}

    def fetch_params(self) -> Dict[str, Any]:
        return {"block_size": self._block_size, "cases": self._cases}

    def _build_cases(self, ctx: GenContext, tid: Expr, n_u: Expr,
                     input_var: Var, acc_name: str, seed_name: str
                     ) -> List[Case]:
        acc = Var(T.u32, name=acc_name)

        def bounded_fetch(idx: Expr) -> Subscript:
            return Subscript(T.u32, array=input_var, index=_mod(idx, n_u))

        def ivar(name: str) -> Var:
            return Var(T.i32, name=name)

        cases: List[Case] = []

        # Case 0: while loop with explicit increment.
        i_name = ctx.kx_fresh_name("i")
        i_u = Cast(T.u32, to=T.u32, arg=ivar(i_name))
        case0: List[Stmt] = [
            Declare(name=i_name, gtype=T.i32, init=Lit(T.i32, value=0)),
            While(cond=RelOp(T.boolean, op=Cmp.LT, lhs=ivar(i_name),
                             rhs=Lit(T.i32, value=2)),
                  body=Block([
                      Assign(acc, BinOp(T.u32, op=Arith.BXOR, lhs=acc,
                                        rhs=bounded_fetch(BinOp(
                                            T.u32, op=Arith.ADD, lhs=tid,
                                            rhs=i_u)))),
                      IncDecStmt(ivar(i_name), op=IncDec.INC, prefix=True),
                  ])),
        ]
        cases.append(Case(value=Lit(T.i32, value=0), body=Block(case0)))

        # Case 1: do-while loop.
        j_name = ctx.kx_fresh_name("j")
        j_u = Cast(T.u32, to=T.u32, arg=ivar(j_name))
        case1: List[Stmt] = [
            Declare(name=j_name, gtype=T.i32, init=Lit(T.i32, value=0)),
            DoWhile(body=Block([
                        Assign(acc, BinOp(
                            T.u32, op=Arith.ADD, lhs=acc,
                            rhs=bounded_fetch(BinOp(
                                T.u32, op=Arith.ADD, lhs=tid,
                                rhs=BinOp(T.u32, op=Arith.MUL, lhs=j_u,
                                          rhs=Lit(T.u32, value=3)))))),
                        IncDecStmt(ivar(j_name), op=IncDec.INC, prefix=True),
                    ]),
                    cond=RelOp(T.boolean, op=Cmp.LT, lhs=ivar(j_name),
                               rhs=Lit(T.i32, value=3))),
        ]
        cases.append(Case(value=Lit(T.i32, value=1), body=Block(case1)))

        # Case 2: for-loop with a strided gather.
        k_name = ctx.kx_fresh_name("k")
        k_u = Cast(T.u32, to=T.u32, arg=ivar(k_name))
        case2: List[Stmt] = [
            For(var=k_name, start=Lit(T.i32, value=0),
                end=Lit(T.i32, value=3), step=Lit(T.i32, value=1),
                body=Block([
                    Assign(acc, BinOp(
                        T.u32, op=Arith.BXOR, lhs=acc,
                        rhs=bounded_fetch(BinOp(
                            T.u32, op=Arith.ADD,
                            lhs=BinOp(T.u32, op=Arith.ADD, lhs=tid,
                                      rhs=BinOp(T.u32, op=Arith.MUL, lhs=k_u,
                                                rhs=Lit(T.u32, value=5))),
                            rhs=Lit(T.u32, value=1))))),
                ])),
        ]
        cases.append(Case(value=Lit(T.i32, value=2), body=Block(case2)))

        # Case 3: fence-heavy branch.  __threadfence_block is NOT a barrier
        # (it does not block on other threads), so it is safe inside a
        # divergent switch arm.
        case3: List[Stmt] = [
            Fence(kx_scope="block"),
            Assign(acc, BinOp(T.u32, op=Arith.BXOR, lhs=acc,
                              rhs=BinOp(T.u32, op=Arith.ADD,
                                        lhs=Var(T.u32, name=seed_name),
                                        rhs=Lit(T.u32, value=0x7F4A7C15)))),
        ]
        cases.append(Case(value=Lit(T.i32, value=3), body=Block(case3)))

        # Default: while loop with a parity-skip continue.
        m_name = ctx.kx_fresh_name("m")
        m_u = Cast(T.u32, to=T.u32, arg=ivar(m_name))
        case_default: List[Stmt] = [
            Declare(name=m_name, gtype=T.i32, init=Lit(T.i32, value=0)),
            While(cond=RelOp(T.boolean, op=Cmp.LT, lhs=ivar(m_name),
                             rhs=Lit(T.i32, value=4)),
                  body=Block([
                      If(cond=RelOp(T.boolean, op=Cmp.EQ,
                                    lhs=BinOp(T.u32, op=Arith.BAND, lhs=m_u,
                                              rhs=Lit(T.u32, value=1)),
                                    rhs=Lit(T.u32, value=0)),
                         then=Block([
                             IncDecStmt(ivar(m_name), op=IncDec.INC,
                                        prefix=True),
                             Continue(),
                         ])),
                      Assign(acc, BinOp(
                          T.u32, op=Arith.ADD, lhs=acc,
                          rhs=bounded_fetch(BinOp(
                              T.u32, op=Arith.ADD, lhs=tid,
                              rhs=BinOp(T.u32, op=Arith.MUL, lhs=m_u,
                                        rhs=Lit(T.u32, value=11)))))),
                      IncDecStmt(ivar(m_name), op=IncDec.INC, prefix=True),
                  ])),
        ]
        cases.append(Case(value=None, body=Block(case_default)))
        return cases

    def build_body(self, ctx: GenContext) -> List[Stmt]:
        tid, n_var, output_var, input_var, n_u = self._scope(ctx)
        bs = self._block_size
        flat = self.kx_flat_mode(ctx)
        # Snapshot the pool BEFORE declaring scratch/accumulators so the flat
        # sink combines with the generated pool, not the template's own
        # seed/bias/acc state.
        pool_snapshot = ctx.kx_visible_declared_names() if flat else None
        name, smem, decl = self._declare_shared(ctx, "sd_smem", bs)
        lane = _tid()

        stmts: List[Stmt] = [decl]

        # seed / bias stay live across the divergent switch (sampled from the
        # pool in flat mode).
        seed_load = Subscript(T.u32, array=input_var, index=_mod(tid, n_u))
        bias_load = Subscript(T.u32, array=input_var, index=_mod(
            BinOp(T.u32, op=Arith.ADD, lhs=tid, rhs=Lit(T.u32, value=7)), n_u))
        if flat:
            seed, seed_decl = ctx.kx_sample_or_create(T.u32, for_write=True,
                                                   init=seed_load)
            stmts.append(seed_decl if seed_decl is not None
                         else Assign(seed, seed_load))
            bias, bias_decl = ctx.kx_sample_or_create(T.u32, for_write=True,
                                                   init=bias_load)
            stmts.append(bias_decl if bias_decl is not None
                         else Assign(bias, bias_load))
            seed_name, bias_name = seed.name, bias.name
        else:
            seed_name = ctx.kx_fresh_name("seed")
            bias_name = ctx.kx_fresh_name("bias")
            ctx.kx_declare(Var(T.u32, name=seed_name))
            ctx.kx_declare(Var(T.u32, name=bias_name))
            stmts.append(Declare(name=seed_name, gtype=T.u32, init=seed_load))
            stmts.append(Declare(name=bias_name, gtype=T.u32, init=bias_load))

        # acc is a DEDICATED fresh var (never sampled): it is mutated inside
        # the divergent arms, so it must not alias a pool var that generated
        # statements also read.
        acc_name = ctx.kx_fresh_name("sd_acc")
        acc = Var(T.u32, name=acc_name)
        ctx.kx_declare(acc)
        stmts.append(Declare(name=acc_name, gtype=T.u32,
                             init=BinOp(T.u32, op=Arith.BXOR,
                                        lhs=Var(T.u32, name=seed_name),
                                        rhs=Var(T.u32, name=bias_name))))

        # Deterministic shared anchor (block-local slot) before the switch --
        # unconditional, uniform shared write.
        stmts.append(Assign(Subscript(T.u32, array=smem, index=lane),
                            BinOp(T.u32, op=Arith.BXOR,
                                  lhs=Var(T.u32, name=acc_name), rhs=tid)))

        # Switch key: per-thread, `cases`-way divergence (sampled in flat).
        mode_init = _mod(tid, Lit(T.u32, value=self._cases))
        if flat:
            mode, mode_decl = ctx.kx_sample_or_create(T.u32, for_write=True,
                                                   init=mode_init)
            stmts.append(mode_decl if mode_decl is not None
                         else Assign(mode, mode_init))
            mode_name = mode.name
        else:
            mode_name = ctx.kx_fresh_name("mode")
            ctx.kx_declare(Var(T.u32, name=mode_name))
            stmts.append(Declare(name=mode_name, gtype=T.u32, init=mode_init))

        stmts.append(Switch(expr=Var(T.u32, name=mode_name),
                            cases=self._build_cases(ctx, tid, n_u, input_var,
                                                    acc_name, seed_name)))
        # Post-switch reconvergence (top level -> uniform barrier).
        stmts.append(Fence(kx_scope="block"))
        stmts.append(Barrier())

        # Cross-lane shared read after reconvergence (block-local index).
        neighbor = Subscript(T.u32, array=smem, index=_mod(
            BinOp(T.u32, op=Arith.ADD, lhs=lane, rhs=Lit(T.u32, value=1)),
            Lit(T.u32, value=bs)))
        final = BinOp(
            T.u32, op=Arith.BXOR,
            lhs=BinOp(T.u32, op=Arith.BXOR, lhs=Var(T.u32, name=acc_name),
                      rhs=Var(T.u32, name=seed_name)),
            rhs=BinOp(T.u32, op=Arith.BXOR, lhs=Var(T.u32, name=bias_name),
                      rhs=neighbor))

        if flat:
            stmts.extend(self._flat_sink(ctx, pool_snapshot, final))
            # Make the scratch safe to re-anchor next repeat iteration.
            stmts.append(Barrier())
            return stmts
        stmts.append(If(cond=RelOp(T.boolean, op=Cmp.LT, lhs=tid, rhs=n_u),
                        then=Block([Assign(
                            Subscript(T.u32, array=output_var, index=tid),
                            final)])))
        stmts.append(Barrier())
        return stmts


# ---------------------------------------------------------------------------
# Gather / scatter
# ---------------------------------------------------------------------------



class GatherScatterTemplate(_SharedTemplate):
    """Irregular gather + deterministic permuted scatter through shared
    memory, with a fence and a write-back to the permuted slot."""

    def __init__(self) -> None:
        self._block_size = 256
        self._gathers = 3

    @property
    def name(self) -> str:
        return "irregular_io"

    @property
    def features(self) -> Set[str]:
        return {"shared_mem", "sync", "strided_access"}

    def fetch_params(self) -> Dict[str, Any]:
        return {"block_size": self._block_size, "gathers": self._gathers}

    def build_body(self, ctx: GenContext) -> List[Stmt]:
        tid, n_var, output_var, input_var, n_u = self._scope(ctx)
        bs = self._block_size
        flat = self.kx_flat_mode(ctx)
        # Snapshot the pre-template pool so the flat sink combines with the
        # generated pool rather than the gather/scatter temporaries.
        pool_snapshot = ctx.kx_visible_declared_names() if flat else None
        name, smem, decl = self._declare_shared(ctx, "gs_smem", bs)
        lane = _tid()

        stmts: List[Stmt] = [decl]
        # Gather: seeded with input[tid % n], then `gathers` irregular rounds
        # folding input[(tid + g*7 + 3) % n] (sampled from the pool in flat).
        seed_load = Subscript(T.u32, array=input_var, index=_mod(tid, n_u))
        if flat:
            gather, gather_decl = ctx.kx_sample_or_create(T.u32, for_write=True,
                                                       init=seed_load)
            stmts.append(gather_decl if gather_decl is not None
                         else Assign(gather, seed_load))
            gather_name = gather.name
        else:
            gather_name = ctx.kx_fresh_name("gather")
            ctx.kx_declare(Var(T.u32, name=gather_name))
            stmts.append(Declare(name=gather_name, gtype=T.u32,
                                 init=seed_load))

        g_name = ctx.kx_fresh_name("g")
        g_u = Cast(T.u32, to=T.u32, arg=Var(T.i32, name=g_name))
        g_idx = _mod(BinOp(T.u32, op=Arith.ADD,
                           lhs=BinOp(T.u32, op=Arith.ADD, lhs=tid,
                                     rhs=BinOp(T.u32, op=Arith.MUL, lhs=g_u,
                                               rhs=Lit(T.u32, value=7))),
                           rhs=Lit(T.u32, value=3)), n_u)
        stmts.append(For(var=g_name, start=Lit(T.i32, value=0),
                         end=Lit(T.i32, value=self._gathers),
                         step=Lit(T.i32, value=1),
                         body=Block([Assign(Var(T.u32, name=gather_name),
                             BinOp(T.u32, op=Arith.BXOR,
                                   lhs=Var(T.u32, name=gather_name),
                                   rhs=Subscript(T.u32, array=input_var,
                                                 index=g_idx)))])))

        # Stage the aggregate (block-local slot) across a fence + barrier.
        stmts.append(Assign(Subscript(T.u32, array=smem, index=lane),
                            Var(T.u32, name=gather_name)))
        stmts.append(Fence(kx_scope="block"))
        stmts.append(Barrier())

        # Deterministic permuted scatter (block-local index).
        scatter_idx = _mod(BinOp(T.u32, op=Arith.ADD,
                                 lhs=BinOp(T.u32, op=Arith.MUL, lhs=lane,
                                           rhs=Lit(T.u32, value=17)),
                                 rhs=Lit(T.u32, value=13)),
                           Lit(T.u32, value=bs))
        scatter_expr = BinOp(
            T.u32, op=Arith.BXOR,
            lhs=Subscript(T.u32, array=smem, index=scatter_idx),
            rhs=Var(T.u32, name=gather_name))
        if flat:
            scatter, scatter_decl = ctx.kx_sample_or_create(
                T.u32, for_write=True, init=scatter_expr)
            stmts.append(scatter_decl if scatter_decl is not None
                         else Assign(scatter, scatter_expr))
            scatter_name = scatter.name
        else:
            scatter_name = ctx.kx_fresh_name("scatter")
            ctx.kx_declare(Var(T.u32, name=scatter_name))
            stmts.append(Declare(name=scatter_name, gtype=T.u32,
                                 init=scatter_expr))

        # Write the scattered value back to the permuted slot too.
        stmts.append(Assign(Subscript(T.u32, array=smem, index=scatter_idx),
                            Var(T.u32, name=scatter_name)))

        if flat:
            result = BinOp(T.u32, op=Arith.BXOR,
                           lhs=Var(T.u32, name=scatter_name),
                           rhs=Var(T.u32, name=gather_name))
            stmts.extend(self._flat_sink(ctx, pool_snapshot, result))
            # Make the scratch safe to re-stage next repeat iteration.
            stmts.append(Barrier())
            return stmts
        stmts.append(If(cond=RelOp(T.boolean, op=Cmp.LT, lhs=tid, rhs=n_u),
                        then=Block([Assign(
                            Subscript(T.u32, array=output_var, index=tid),
                            Var(T.u32, name=scatter_name))])))
        stmts.append(Barrier())
        return stmts


# ---------------------------------------------------------------------------
# Warp communication fallback (shared-memory lane exchange)
# ---------------------------------------------------------------------------



class WarpCommFallbackTemplate(_SharedTemplate):
    """Warp-local lane exchange via shared memory (portable shuffle): stage a
    per-thread value, exchange with rotating peer lanes across `rounds`,
    then a fixed-neighbour tail exchange."""

    def __init__(self) -> None:
        self._block_size = 256
        self._rounds = 2

    @property
    def name(self) -> str:
        return "lane_exchange"

    @property
    def features(self) -> Set[str]:
        return {"shared_mem", "sync", "strided_access"}

    def fetch_params(self) -> Dict[str, Any]:
        return {"block_size": self._block_size, "rounds": self._rounds}

    def build_body(self, ctx: GenContext) -> List[Stmt]:
        tid, n_var, output_var, input_var, n_u = self._scope(ctx)
        bs = self._block_size
        flat = self.kx_flat_mode(ctx)
        # Snapshot the pre-template pool so the flat sink combines with the
        # generated pool rather than the communication temporaries.
        pool_snapshot = ctx.kx_visible_declared_names() if flat else None
        name, smem, decl = self._declare_shared(ctx, "wc_smem", bs)
        lane = _tid()
        c32 = Lit(T.u32, value=32)
        expr_gen = ExprGen(ctx)

        # Block-local lane / warp indices: scratch is a per-block __shared__
        # array, so peer slots must be warp-local (warp*32 + lane).
        lane_name = ctx.kx_fresh_name("lane")
        warp_name = ctx.kx_fresh_name("warp")
        base_name = ctx.kx_fresh_name("base")
        lane_v = Var(T.u32, name=lane_name)
        warp_v = Var(T.u32, name=warp_name)
        ctx.kx_declare(lane_v)
        ctx.kx_declare(warp_v)
        ctx.kx_declare(Var(T.u32, name=base_name))
        stmts: List[Stmt] = [
            decl,
            Declare(name=lane_name, gtype=T.u32, init=_mod(lane, c32)),
            Declare(name=warp_name, gtype=T.u32,
                    init=BinOp(T.u32, op=Arith.DIV, lhs=lane, rhs=c32)),
            # Per-thread seed from a bounded global load.
            Declare(name=base_name, gtype=T.u32,
                    init=BinOp(T.u32, op=Arith.BXOR,
                               lhs=Subscript(T.u32, array=input_var,
                                             index=_mod(tid, n_u)),
                               rhs=tid)),
        ]

        # Stage the seed into the per-thread slot (unconditional).
        stmts.append(Assign(Subscript(T.u32, array=smem, index=lane),
                            Var(T.u32, name=base_name)))
        stmts.append(Fence(kx_scope="block"))
        stmts.append(Barrier())

        # Exchange with a rotating peer lane inside the warp across `rounds`.
        r_name = ctx.kx_fresh_name("r")
        r_u = Cast(T.u32, to=T.u32, arg=Var(T.i32, name=r_name))
        peer_lane = _mod(BinOp(T.u32, op=Arith.ADD, lhs=lane_v,
                               rhs=BinOp(T.u32, op=Arith.ADD, lhs=r_u,
                                         rhs=Lit(T.u32, value=1))), c32)
        peer_idx = BinOp(T.u32, op=Arith.ADD,
                         lhs=BinOp(T.u32, op=Arith.MUL, lhs=warp_v, rhs=c32),
                         rhs=peer_lane)
        stmts.append(For(var=r_name, start=Lit(T.i32, value=0),
                         end=Lit(T.i32, value=self._rounds),
                         step=Lit(T.i32, value=1),
                         body=Block([
                             Assign(Subscript(T.u32, array=smem, index=lane),
                                    BinOp(T.u32, op=Arith.BXOR,
                                          lhs=BinOp(
                                              T.u32, op=Arith.BXOR,
                                              lhs=Subscript(
                                                  T.u32, array=smem,
                                                  index=lane),
                                              rhs=Subscript(
                                                  T.u32, array=smem,
                                                  index=peer_idx)),
                                          rhs=r_u)),
                             Barrier(),
                         ])))

        # Tail exchange: combine the own slot with a fixed neighbour lane.
        tail_lane = _mod(BinOp(T.u32, op=Arith.ADD, lhs=lane_v,
                               rhs=Lit(T.u32, value=31)), c32)
        tail_idx = BinOp(T.u32, op=Arith.ADD,
                         lhs=BinOp(T.u32, op=Arith.MUL, lhs=warp_v, rhs=c32),
                         rhs=tail_lane)
        final = BinOp(T.u32, op=Arith.BXOR,
                      lhs=Subscript(T.u32, array=smem, index=lane),
                      rhs=Subscript(T.u32, array=smem, index=tail_idx))

        if flat:
            result = BinOp(T.u32, op=Arith.BXOR, lhs=final,
                           rhs=expr_gen.kx_generate(T.u32))
            stmts.extend(self._flat_sink(ctx, pool_snapshot, result))
            # Make scratch safe to reuse on the next repeat-loop iteration.
            stmts.append(Barrier())
            return stmts
        stmts.append(If(cond=RelOp(T.boolean, op=Cmp.LT, lhs=tid, rhs=n_u),
                        then=Block([Assign(
                            Subscript(T.u32, array=output_var, index=tid),
                            final)])))
        stmts.append(Barrier())
        return stmts


# ---------------------------------------------------------------------------
# Permutation communication
# ---------------------------------------------------------------------------



class PermutationCommTemplate(_SharedTemplate):
    """Hash-permutation exchange: two multiplicative-hash permutation tables
    plus a comm buffer; two permuted write/read rounds, then an XOR-1
    near-stride-1 round as a compiler contrast target."""

    def __init__(self) -> None:
        self._block_size = 256

    @property
    def name(self) -> str:
        return "perm_round"

    @property
    def features(self) -> Set[str]:
        return {"shared_mem", "sync", "strided_access"}

    def fetch_params(self) -> Dict[str, Any]:
        return {"block_size": self._block_size}

    def kx_required_shared_mem(self) -> int:
        # Three shared arrays of block_size uints: perm1, perm2, comm.
        return self._block_size * 3 * 4

    def build_body(self, ctx: GenContext) -> List[Stmt]:
        tid, n_var, output_var, input_var, n_u = self._scope(ctx)
        bs = self._block_size
        flat = self.kx_flat_mode(ctx)
        # Snapshot the pre-template pool so the flat sink combines with the
        # generated pool rather than the template's own round temporaries.
        pool_snapshot = ctx.kx_visible_declared_names() if flat else None
        _, perm1, perm1_decl = self._declare_shared(ctx, "pc_perm1", bs)
        _, perm2, perm2_decl = self._declare_shared(ctx, "pc_perm2", bs)
        _, comm, comm_decl = self._declare_shared(ctx, "pc_comm", bs)
        lid = _tid()
        bs_c = Lit(T.u32, value=bs)

        stmts: List[Stmt] = [perm1_decl, perm2_decl, comm_decl]

        # Build both permutation tables: multiplicative hashes with odd
        # multipliers (bijective over [0, bs)).
        stmts.append(Assign(Subscript(T.u32, array=perm1, index=lid),
                            _mod(BinOp(T.u32, op=Arith.ADD,
                                       lhs=BinOp(T.u32, op=Arith.MUL, lhs=lid,
                                                 rhs=Lit(T.u32, value=137)),
                                       rhs=Lit(T.u32, value=59)), bs_c)))
        stmts.append(Assign(Subscript(T.u32, array=perm2, index=lid),
                            _mod(BinOp(T.u32, op=Arith.ADD,
                                       lhs=BinOp(T.u32, op=Arith.MUL, lhs=lid,
                                                 rhs=Lit(T.u32, value=173)),
                                       rhs=Lit(T.u32, value=97)), bs_c)))
        stmts.append(Barrier())

        val_name = ctx.kx_fresh_name("pc_val")
        r1_name = ctx.kx_fresh_name("pc_r1")
        r2_name = ctx.kx_fresh_name("pc_r2")
        r3_name = ctx.kx_fresh_name("pc_r3")
        r1 = Var(T.u32, name=r1_name)
        r2 = Var(T.u32, name=r2_name)
        r3 = Var(T.u32, name=r3_name)
        for vn in (val_name, r1_name, r2_name, r3_name):
            ctx.kx_declare(Var(T.u32, name=vn))

        # Round 1: write to the permuted slot, read back the own slot.
        stmts.append(Declare(name=val_name, gtype=T.u32, init=BinOp(
            T.u32, op=Arith.BXOR,
            lhs=Subscript(T.u32, array=input_var, index=_mod(tid, n_u)),
            rhs=BinOp(T.u32, op=Arith.MUL, lhs=lid,
                      rhs=Lit(T.u32, value=0x9E3779B9)))))
        stmts.append(Assign(Subscript(T.u32, array=comm,
                                      index=Subscript(T.u32, array=perm1,
                                                      index=lid)),
                            Var(T.u32, name=val_name)))
        stmts.append(Barrier())
        stmts.append(Declare(name=r1_name, gtype=T.u32,
                             init=Subscript(T.u32, array=comm, index=lid)))
        stmts.append(Barrier())

        # Round 2: different permutation -> different bank-conflict pattern.
        stmts.append(Assign(Subscript(T.u32, array=comm,
                                      index=Subscript(T.u32, array=perm2,
                                                      index=lid)),
                            BinOp(T.u32, op=Arith.BXOR, lhs=r1,
                                  rhs=Lit(T.u32, value=0xDEADBEEF))))
        stmts.append(Barrier())
        stmts.append(Declare(name=r2_name, gtype=T.u32,
                             init=Subscript(T.u32, array=comm, index=lid)))
        stmts.append(Barrier())

        # Round 3: XOR-1 neighbour exchange (near-stride-1 contrast).
        stmts.append(Assign(Subscript(T.u32, array=comm,
                                      index=BinOp(T.u32, op=Arith.BXOR,
                                                  lhs=lid,
                                                  rhs=Lit(T.u32, value=1))),
                            r2))
        stmts.append(Barrier())
        stmts.append(Declare(name=r3_name, gtype=T.u32,
                             init=Subscript(T.u32, array=comm, index=lid)))

        result = BinOp(T.u32, op=Arith.BXOR,
                       lhs=BinOp(T.u32, op=Arith.BXOR, lhs=r1, rhs=r2),
                       rhs=r3)

        if flat:
            stmts.extend(self._flat_sink(ctx, pool_snapshot, result))
            # Make the shared arrays safe to reuse on the next repeat
            # iteration.
            stmts.append(Barrier())
            return stmts
        stmts.append(If(cond=RelOp(T.boolean, op=Cmp.LT, lhs=tid, rhs=n_u),
                        then=Block([Assign(
                            Subscript(T.u32, array=output_var, index=tid),
                            result)])))
        # Trailing barrier: keeps the shared arrays reusable across the
        # repeat-loop iterations (matches the sibling templates).
        stmts.append(Barrier())
        return stmts


# ---------------------------------------------------------------------------
# Register pressure
# ---------------------------------------------------------------------------

_PRESSURE_LEVELS = {"low": 24, "medium": 48, "high": 80, "extreme": 128}


def _kx_mix_constant(i: int) -> int:
    return (0x9E3779B9 * (i + 1)) & 0xFFFFFFFF
