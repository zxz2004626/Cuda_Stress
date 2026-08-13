"""The seven classic kernel templates.

elementwise, reduction, stencil, matmul, scan, histogram, transpose —
each is a structural pattern with generator-filled expression holes.  All are
deterministic and barrier-safe by construction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

from generator_ir.ir.typedecl import (
    T, MemSpace, Qual, TArray,
)
from generator_ir.ir.operators import Arith, Cmp, SafeOp
from generator_ir.ir.nodes import (
    Expr, Lit, Var, ThreadBuiltin, BinOp, RelOp, Cast, SafeBin, Subscript,
    WarpShuffle,
)
from generator_ir.ir.statements import (
    Stmt, Assign, Declare, Block, If, Barrier,
)
from generator_ir.templates.skeleton import KernelTemplate
from generator_ir.gen.context import GenContext
from generator_ir.gen.expr_builder import ExprGen
from generator_ir.gen.stmt_builder import StmtGen


def _u(v: Expr) -> Cast:
    return Cast(T.u32, to=T.u32, arg=v)


# ---------------------------------------------------------------------------
# Elementwise
# ---------------------------------------------------------------------------


class ElementwiseTemplate(KernelTemplate):
    """output[tid] = f(input[tid]) -- coalesced, arithmetic-heavy."""

    def __init__(self) -> None:
        self._arithmetic_intensity = 4

    @property
    def name(self) -> str:
        return "map_transform"

    @property
    def features(self) -> Set[str]:
        return {"coalesced_access"}

    def fetch_params(self) -> Dict[str, Any]:
        return {"arithmetic_intensity": self._arithmetic_intensity}

    def build_body(self, ctx: GenContext) -> List[Stmt]:
        expr_gen = ExprGen(ctx)
        stmt_gen = StmtGen(ctx, expr_gen)
        tid = ctx.kx_scope.lookup("tid")
        n_var = ctx.kx_scope.lookup("n")
        output_var = ctx.kx_scope.lookup("output")
        input_var = ctx.kx_scope.lookup("input")
        n_u = _u(n_var)

        if self.kx_flat_mode(ctx):
            stmts: List[Stmt] = []
            load = Subscript(T.u32, array=input_var,
                             index=SafeBin(T.u32, op=SafeOp.SMOD, lhs=tid, rhs=n_u))
            val, val_decl = ctx.kx_sample_or_create(T.u32, for_write=True, init=load)
            if val_decl is not None:
                stmts.append(val_decl)
            else:
                stmts.append(Assign(val, load))
            for _ in range(self._arithmetic_intensity):
                stmts.extend(stmt_gen.build_stmts(1))
            result = BinOp(T.u32, op=Arith.BXOR, lhs=val,
                           rhs=expr_gen.kx_generate(T.u32))
            sink, sink_decl = ctx.kx_sample_or_create(T.u32, for_write=True)
            if sink_decl is not None:
                stmts.append(sink_decl)
            stmts.append(Assign(sink, result))
            return stmts

        # Structured path: guard the whole body (no early return; a return
        # would deadlock the repeat-loop barrier).
        val_name = ctx.kx_fresh_name("val")
        res_name = ctx.kx_fresh_name("result")
        val = Var(T.u32, name=val_name)
        res = Var(T.u32, name=res_name)
        ctx.kx_declare(val)
        ctx.kx_declare(res)
        stmts = [
            Declare(name=val_name, gtype=T.u32, init=Lit(T.u32, value=0)),
            Declare(name=res_name, gtype=T.u32, init=Lit(T.u32, value=0)),
        ]
        ctx.kx_push_scope()
        ctx.if_depth += 1
        guarded: List[Stmt] = [
            Assign(val, Subscript(T.u32, array=input_var, index=tid)),
        ]
        for _ in range(self._arithmetic_intensity):
            guarded.extend(stmt_gen.build_stmts(1))
        guarded.append(Assign(res, BinOp(T.u32, op=Arith.BXOR, lhs=val,
                                         rhs=expr_gen.kx_generate(T.u32))))
        guarded.append(Assign(Subscript(T.u32, array=output_var, index=tid), res))
        ctx.if_depth -= 1
        ctx.kx_pop_scope()
        stmts.append(If(cond=RelOp(T.boolean, op=Cmp.LT, lhs=tid, rhs=n_u),
                        then=Block(guarded)))
        return stmts


# ---------------------------------------------------------------------------
# Reduction
# ---------------------------------------------------------------------------

_REDUCE_OPS = {
    "add": Arith.ADD,
    "xor": Arith.BXOR,
    "or": Arith.BOR,
    "and": Arith.BAND,
}




class ReductionTemplate(KernelTemplate):
    """Shared-memory tree reduction + warp shuffle finish."""

    def __init__(self) -> None:
        self._reduce_op = "add"
        self._block_size = 256

    @property
    def name(self) -> str:
        return "tree_reduce"

    @property
    def features(self) -> Set[str]:
        return {"shared_mem", "warp_shuffle", "sync", "atomics", "reduction"}

    def fetch_params(self) -> Dict[str, Any]:
        return {"reduce_op": self._reduce_op, "block_size": self._block_size}

    def kx_required_shared_mem(self) -> int:
        return self._block_size * 4

    def preferred_block(self) -> Tuple[int, int, int]:
        return (self._block_size, 1, 1)

    def build_body(self, ctx: GenContext) -> List[Stmt]:
        bs = self._block_size
        op = _REDUCE_OPS.get(self._reduce_op, Arith.ADD)
        identity = 0xFFFFFFFF if self._reduce_op == "and" else 0
        expr_gen = ExprGen(ctx)

        tid = ctx.kx_scope.lookup("tid")
        n_var = ctx.kx_scope.lookup("n")
        input_var = ctx.kx_scope.lookup("input")
        n_u = _u(n_var)
        lane = ThreadBuiltin(T.u32, axis="x", source="threadIdx")

        sdata_name = ctx.kx_fresh_name("sdata")
        sdata = Var(TArray(T.u32, bs), name=sdata_name, space=MemSpace.SHARED)
        ctx.kx_declare(sdata)
        stmts: List[Stmt] = [Declare(
            name=sdata_name, gtype=TArray(T.u32, bs),
            space=MemSpace.SHARED, quals=[Qual.SHARED_MEM])]

        flat = self.kx_flat_mode(ctx)
        if flat:
            # Unconditional shared write keeps the barrier uniform.
            pool_snapshot = ctx.kx_scope.declared_names()
            load_idx = SafeBin(T.u32, op=SafeOp.SMOD, lhs=tid, rhs=n_u)
            load = BinOp(T.u32, op=Arith.BXOR,
                         lhs=Subscript(T.u32, array=input_var, index=load_idx),
                         rhs=expr_gen.kx_generate(T.u32, depth=2))
            stmts.append(Assign(Subscript(T.u32, array=sdata, index=lane), load))
            stmts.append(Barrier())
        else:
            stmts.append(Assign(Subscript(T.u32, array=sdata, index=lane),
                                Lit(T.u32, value=identity)))
            load = BinOp(T.u32, op=Arith.BXOR,
                         lhs=Subscript(T.u32, array=input_var, index=tid),
                         rhs=expr_gen.kx_generate(T.u32, depth=2))
            stmts.append(If(cond=RelOp(T.boolean, op=Cmp.LT, lhs=tid, rhs=n_u),
                            then=Block([Assign(
                                Subscript(T.u32, array=sdata, index=lane), load)])))
            stmts.append(Barrier())

        # Tree reduction down to 32 lanes.
        s = bs // 2
        while s >= 32:
            sc = Lit(T.u32, value=s)
            stmts.append(If(
                cond=RelOp(T.boolean, op=Cmp.LT, lhs=lane, rhs=sc),
                then=Block([Assign(
                    Subscript(T.u32, array=sdata, index=lane),
                    BinOp(T.u32, op=op,
                          lhs=Subscript(T.u32, array=sdata, index=lane),
                          rhs=Subscript(T.u32, array=sdata,
                                        index=BinOp(T.u32, op=Arith.ADD,
                                                    lhs=lane, rhs=sc))))])))
            stmts.append(Barrier())
            s //= 2

        # Warp shuffle finish.
        warp_val_name = ctx.kx_fresh_name("warp_val")
        ctx.kx_declare(Var(T.u32, name=warp_val_name))
        stmts.append(Declare(name=warp_val_name, gtype=T.u32,
                             init=Subscript(T.u32, array=sdata, index=lane)))
        full_mask = Lit(T.u32, value=0xFFFFFFFF)
        for delta in (16, 8, 4, 2, 1):
            sh_name = ctx.kx_fresh_name("sh")
            ctx.kx_declare(Var(T.u32, name=sh_name))
            stmts.append(Declare(
                name=sh_name, gtype=T.u32,
                init=WarpShuffle(T.u32, kind="shfl_down_sync", mask=full_mask,
                                 value=Var(T.u32, name=warp_val_name),
                                 delta=Lit(T.i32, value=delta))))
            stmts.append(Assign(
                Var(T.u32, name=warp_val_name),
                BinOp(T.u32, op=op,
                      lhs=Var(T.u32, name=warp_val_name),
                      rhs=Var(T.u32, name=sh_name))))

        if flat:
            # Fold the reduced value into a pooled variable.
            result = BinOp(T.u32, op=Arith.BXOR,
                           lhs=Var(T.u32, name=warp_val_name),
                           rhs=Subscript(T.u32, array=input_var,
                                         index=SafeBin(T.u32, op=SafeOp.SMOD,
                                                       lhs=tid, rhs=n_u)))
            saved_visible = ctx._visible_names
            ctx._visible_names = pool_snapshot
            try:
                sink, sink_decl = ctx.kx_sample_or_create(T.u32, for_write=True)
            finally:
                ctx._visible_names = saved_visible
            if sink_decl is not None:
                stmts.append(sink_decl)
            stmts.append(Assign(sink, result))
            # Reusable sdata on the next repeat-loop iteration.
            stmts.append(Barrier())
        else:
            # Consume the reduction result via a volatile sink (avoids
            # polluting the output array, which would be non-deterministic
            # due to block execution order).  Volatile is unconditional here
            # (mirrors perffuzzer): without it the whole reduction is dead
            # code and would be eliminated at -O3, killing the perf signal.
            sink_name = ctx.kx_fresh_name("red_sink")
            stmts.append(Declare(name=sink_name, gtype=T.u32,
                                 quals=[Qual.VOLATILE],
                                 init=Var(T.u32, name=warp_val_name)))
        return stmts


# ---------------------------------------------------------------------------
# Stencil
# ---------------------------------------------------------------------------



class StencilTemplate(KernelTemplate):
    """1D stencil with a shared-memory tile + halo."""

    def __init__(self) -> None:
        self._radius = 1
        self._tile_size = 256

    @property
    def name(self) -> str:
        return "neighbor_sum"

    @property
    def features(self) -> Set[str]:
        return {"shared_mem", "sync", "tiling", "stencil"}

    def fetch_params(self) -> Dict[str, Any]:
        return {"radius": self._radius, "tile_size": self._tile_size}

    def kx_required_shared_mem(self) -> int:
        return (self._tile_size + 2 * self._radius) * 4

    def preferred_block(self) -> Tuple[int, int, int]:
        return (self._tile_size, 1, 1)

    def build_body(self, ctx: GenContext) -> List[Stmt]:
        if self.kx_flat_mode(ctx):
            return self.kx_flat_body(ctx)
        r = self._radius
        ts = self._tile_size
        expr_gen = ExprGen(ctx)
        tid = ctx.kx_scope.lookup("tid")
        n_var = ctx.kx_scope.lookup("n")
        output_var = ctx.kx_scope.lookup("output")
        input_var = ctx.kx_scope.lookup("input")
        n_u = _u(n_var)
        lane = ThreadBuiltin(T.u32, axis="x", source="threadIdx")

        tile_name = ctx.kx_fresh_name("tile")
        tile = Var(TArray(T.u32, ts + 2 * r), name=tile_name, space=MemSpace.SHARED)
        ctx.kx_declare(tile)

        center = BinOp(T.u32, op=Arith.ADD, lhs=lane, rhs=Lit(T.u32, value=r))
        r_lit = Lit(T.u32, value=r)
        bdx = ThreadBuiltin(T.u32, axis="x", source="blockDim")
        stmts: List[Stmt] = [
            Declare(name=tile_name, gtype=TArray(T.u32, ts + 2 * r),
                    space=MemSpace.SHARED, quals=[Qual.SHARED_MEM]),
        ]
        # Center: zero-fill, then a guarded load (every slot initialised).
        stmts.append(Assign(Subscript(T.u32, array=tile, index=center),
                            Lit(T.u32, value=0)))
        stmts.append(If(
            cond=RelOp(T.boolean, op=Cmp.LT, lhs=tid, rhs=n_u),
            then=Block([Assign(Subscript(T.u32, array=tile, index=center),
                               Subscript(T.u32, array=input_var, index=tid))])))
        # Left halo: the r lowest lanes fill tile[lane].
        stmts.append(If(
            cond=RelOp(T.boolean, op=Cmp.LT, lhs=lane, rhs=r_lit),
            then=Block([
                Assign(Subscript(T.u32, array=tile, index=lane),
                       Lit(T.u32, value=0)),
                If(cond=RelOp(T.boolean, op=Cmp.GE, lhs=tid, rhs=r_lit),
                   then=Block([Assign(
                       Subscript(T.u32, array=tile, index=lane),
                       Subscript(T.u32, array=input_var,
                                 index=BinOp(T.u32, op=Arith.SUB,
                                             lhs=tid, rhs=r_lit)))])),
            ])))
        # Right halo: the r highest lanes fill tile[lane + 2r].
        tid_plus_r = BinOp(T.u32, op=Arith.ADD, lhs=tid, rhs=r_lit)
        local_plus_2r = BinOp(T.u32, op=Arith.ADD, lhs=lane,
                              rhs=Lit(T.u32, value=2 * r))
        stmts.append(If(
            cond=RelOp(T.boolean, op=Cmp.GE, lhs=lane,
                       rhs=BinOp(T.u32, op=Arith.SUB, lhs=bdx, rhs=r_lit)),
            then=Block([
                Assign(Subscript(T.u32, array=tile, index=local_plus_2r),
                       Lit(T.u32, value=0)),
                If(cond=RelOp(T.boolean, op=Cmp.LT, lhs=tid_plus_r, rhs=n_u),
                   then=Block([Assign(
                       Subscript(T.u32, array=tile, index=local_plus_2r),
                       Subscript(T.u32, array=input_var, index=tid_plus_r))])),
            ])))
        stmts.append(Barrier())

        # Weighted sum over [-r, +r]; neighbours weighted by abs(off) + 1.
        acc_name = ctx.kx_fresh_name("acc")
        acc = Var(T.u32, name=acc_name)
        ctx.kx_declare(acc)
        terms: Expr = Subscript(T.u32, array=tile, index=center)
        for off in range(-r, r + 1):
            if off == 0:
                continue
            if off > 0:
                idx = BinOp(T.u32, op=Arith.ADD, lhs=center,
                            rhs=Lit(T.u32, value=off))
            else:
                idx = BinOp(T.u32, op=Arith.SUB, lhs=center,
                            rhs=Lit(T.u32, value=-off))
            neighbor = Subscript(T.u32, array=tile, index=idx)
            weighted = BinOp(T.u32, op=Arith.MUL, lhs=neighbor,
                             rhs=Lit(T.u32, value=abs(off) + 1))
            terms = BinOp(T.u32, op=Arith.ADD, lhs=terms, rhs=weighted)
        mixed = BinOp(T.u32, op=Arith.BXOR, lhs=terms,
                      rhs=expr_gen.kx_generate(T.u32, depth=3))
        stmts.append(Declare(name=acc_name, gtype=T.u32, init=mixed))
        stmts.append(If(cond=RelOp(T.boolean, op=Cmp.LT, lhs=tid, rhs=n_u),
                        then=Block([Assign(
                            Subscript(T.u32, array=output_var, index=tid), acc)])))
        stmts.append(Barrier())
        return stmts

    def kx_flat_body(self, ctx: GenContext) -> List[Stmt]:
        """observable_fold: keep the REAL stencil structure (shared tile with a
        halo, barrier before the neighbour reads) but drop the `if(tid<n)`
        wrappers, fully initialise center + both halos with bounded loads, and
        fold the result into a sampled pre-template pool variable."""
        r = self._radius
        ts = self._tile_size
        expr_gen = ExprGen(ctx)
        tid = ctx.kx_scope.lookup("tid")
        n_var = ctx.kx_scope.lookup("n")
        input_var = ctx.kx_scope.lookup("input")
        n_u = _u(n_var)
        lane = ThreadBuiltin(T.u32, axis="x", source="threadIdx")
        bdx = ThreadBuiltin(T.u32, axis="x", source="blockDim")
        r_lit = Lit(T.u32, value=r)

        tile_name = ctx.kx_fresh_name("tile")
        tile = Var(TArray(T.u32, ts + 2 * r), name=tile_name, space=MemSpace.SHARED)
        ctx.kx_declare(tile)
        stmts: List[Stmt] = [
            Declare(name=tile_name, gtype=TArray(T.u32, ts + 2 * r),
                    space=MemSpace.SHARED, quals=[Qual.SHARED_MEM]),
        ]

        # Sample the result sink only from variables that existed before this
        # template (never its own scratch).
        pool_snapshot = ctx.kx_scope.declared_names()

        def bounded_fetch(idx_expr: Expr) -> Expr:
            return Subscript(T.u32, array=input_var,
                             index=SafeBin(T.u32, op=SafeOp.SMOD,
                                           lhs=idx_expr, rhs=n_u))

        center_idx = BinOp(T.u32, op=Arith.ADD, lhs=lane, rhs=r_lit)
        # Center: every thread, unconditional, bounded.
        stmts.append(Assign(Subscript(T.u32, array=tile, index=center_idx),
                            bounded_fetch(tid)))
        # Left halo: lane < r fills tile[lane] with input[(tid + n - r) % n].
        left_src = SafeBin(T.u32, op=SafeOp.SMOD,
                           lhs=BinOp(T.u32, op=Arith.SUB,
                                     lhs=BinOp(T.u32, op=Arith.ADD,
                                               lhs=tid, rhs=n_u),
                                     rhs=r_lit),
                           rhs=n_u)
        stmts.append(If(
            cond=RelOp(T.boolean, op=Cmp.LT, lhs=lane, rhs=r_lit),
            then=Block([Assign(Subscript(T.u32, array=tile, index=lane),
                               Subscript(T.u32, array=input_var,
                                         index=left_src))])))
        # Right halo: lane >= blockDim - r fills tile[lane + 2r] with
        # input[(tid + r) % n].
        local_plus_2r = BinOp(T.u32, op=Arith.ADD, lhs=lane,
                              rhs=Lit(T.u32, value=2 * r))
        right_src = SafeBin(T.u32, op=SafeOp.SMOD,
                            lhs=BinOp(T.u32, op=Arith.ADD, lhs=tid, rhs=r_lit),
                            rhs=n_u)
        stmts.append(If(
            cond=RelOp(T.boolean, op=Cmp.GE, lhs=lane,
                       rhs=BinOp(T.u32, op=Arith.SUB, lhs=bdx, rhs=r_lit)),
            then=Block([Assign(Subscript(T.u32, array=tile, index=local_plus_2r),
                               Subscript(T.u32, array=input_var,
                                         index=right_src))])))
        # Barrier: the whole tile (center + both halos) is now initialised.
        stmts.append(Barrier())

        # Weighted sum over [-r, +r] of tile neighbours.
        sum_expr: Expr = Subscript(T.u32, array=tile, index=center_idx)
        for off in range(-r, r + 1):
            if off == 0:
                continue
            if off > 0:
                idx = BinOp(T.u32, op=Arith.ADD, lhs=center_idx,
                            rhs=Lit(T.u32, value=off))
            else:
                idx = BinOp(T.u32, op=Arith.SUB, lhs=center_idx,
                            rhs=Lit(T.u32, value=-off))
            neighbor = Subscript(T.u32, array=tile, index=idx)
            weighted = BinOp(T.u32, op=Arith.MUL, lhs=neighbor,
                             rhs=Lit(T.u32, value=abs(off) + 1))
            sum_expr = BinOp(T.u32, op=Arith.ADD, lhs=sum_expr, rhs=weighted)

        extra = expr_gen.kx_generate(T.u32, depth=3)
        result_expr = BinOp(T.u32, op=Arith.BXOR, lhs=sum_expr, rhs=extra)

        saved_visible = ctx._visible_names
        ctx._visible_names = pool_snapshot
        try:
            sink, sink_decl = ctx.kx_sample_or_create(T.u32, for_write=True)
        finally:
            ctx._visible_names = saved_visible
        if sink_decl is not None:
            stmts.append(sink_decl)
        stmts.append(Assign(sink, result_expr))
        # Make the tile safe to reuse on the next repeat-loop iteration.
        stmts.append(Barrier())
        return stmts


# ---------------------------------------------------------------------------
# Matmul (2D tiled)
# ---------------------------------------------------------------------------



