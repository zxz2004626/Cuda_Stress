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
    Atomic,
)
from generator_ir.ir.statements import (
    Stmt, Assign, Declare, Block, If, For, Barrier, ExprStmt,
)
from generator_ir.templates.skeleton import KernelTemplate
from generator_ir.gen.context import GenContext
from generator_ir.gen.expr_builder import ExprGen


def _u(v: Expr) -> Cast:
    return Cast(T.u32, to=T.u32, arg=v)


# ---------------------------------------------------------------------------
# Elementwise
# ---------------------------------------------------------------------------


class MatmulTemplate(KernelTemplate):
    """Tiled GEMM: shared tiles + double barrier per tile step."""

    def __init__(self) -> None:
        self._tile_size = 16

    @property
    def name(self) -> str:
        return "tiled_gemm"

    @property
    def features(self) -> Set[str]:
        return {"shared_mem", "sync", "tiling", "matmul"}

    def fetch_params(self) -> Dict[str, Any]:
        return {"tile_size": self._tile_size}

    def kx_required_shared_mem(self) -> int:
        ts = self._tile_size
        return 2 * ts * ts * 4

    def preferred_block(self) -> Tuple[int, int, int]:
        return (self._tile_size, self._tile_size, 1)

    def preferred_grid(self) -> Tuple[int, int, int]:
        return (4, 4, 1)

    def kx_required_n_elements(self) -> int:
        return self.preferred_grid()[0] * self._tile_size

    def kx_is_2d(self) -> bool:
        return True

    def build_body(self, ctx: GenContext) -> List[Stmt]:
        ts = self._tile_size
        n_tiles = 4
        n_var = ctx.kx_scope.lookup("n")
        output_var = ctx.kx_scope.lookup("output")
        input_var = ctx.kx_scope.lookup("input")
        n_u = _u(n_var)
        n_sq = BinOp(T.u32, op=Arith.MUL, lhs=n_u, rhs=n_u)
        tile_c = Lit(T.u32, value=ts)

        a_name = ctx.kx_fresh_name("As")
        b_name = ctx.kx_fresh_name("Bs")
        a_sh = Var(TArray(T.u32, ts * ts), name=a_name, space=MemSpace.SHARED)
        b_sh = Var(TArray(T.u32, ts * ts), name=b_name, space=MemSpace.SHARED)
        ctx.kx_declare(a_sh)
        ctx.kx_declare(b_sh)

        tx_name, ty_name = ctx.kx_fresh_name("tx"), ctx.kx_fresh_name("ty")
        row_name, col_name = ctx.kx_fresh_name("row"), ctx.kx_fresh_name("col")
        acc_name = ctx.kx_fresh_name("acc")
        tx, ty = Var(T.u32, name=tx_name), Var(T.u32, name=ty_name)
        row, col = Var(T.u32, name=row_name), Var(T.u32, name=col_name)
        acc = Var(T.u32, name=acc_name)
        for v in (tx, ty, row, col, acc):
            ctx.kx_declare(v)

        stmts: List[Stmt] = [
            Declare(name=a_name, gtype=TArray(T.u32, ts * ts),
                    space=MemSpace.SHARED, quals=[Qual.SHARED_MEM]),
            Declare(name=b_name, gtype=TArray(T.u32, ts * ts),
                    space=MemSpace.SHARED, quals=[Qual.SHARED_MEM]),
            Declare(name=tx_name, gtype=T.u32,
                    init=ThreadBuiltin(T.u32, axis="x", source="threadIdx")),
            Declare(name=ty_name, gtype=T.u32,
                    init=ThreadBuiltin(T.u32, axis="y", source="threadIdx")),
            Declare(name=row_name, gtype=T.u32,
                    init=BinOp(T.u32, op=Arith.ADD,
                               lhs=BinOp(T.u32, op=Arith.MUL,
                                         lhs=ThreadBuiltin(T.u32, axis="y", source="blockIdx"),
                                         rhs=tile_c),
                               rhs=ty)),
            Declare(name=col_name, gtype=T.u32,
                    init=BinOp(T.u32, op=Arith.ADD,
                               lhs=BinOp(T.u32, op=Arith.MUL,
                                         lhs=ThreadBuiltin(T.u32, axis="x", source="blockIdx"),
                                         rhs=tile_c),
                               rhs=tx)),
            Declare(name=acc_name, gtype=T.u32, init=Lit(T.u32, value=0)),
        ]

        def lin(r: Expr, c: Expr) -> Expr:
            return BinOp(T.u32, op=Arith.ADD,
                         lhs=BinOp(T.u32, op=Arith.MUL, lhs=r, rhs=tile_c), rhs=c)

        t_name = ctx.kx_fresh_name("t")
        t_var = Var(T.u32, name=t_name)
        k_name = ctx.kx_fresh_name("k")
        k_var = Var(T.u32, name=k_name)

        a_idx = BinOp(T.u32, op=Arith.ADD,
                      lhs=BinOp(T.u32, op=Arith.MUL, lhs=row, rhs=n_u),
                      rhs=BinOp(T.u32, op=Arith.ADD, lhs=t_var, rhs=tx))
        # B aliases the A matrix region (host allocates exactly n*n; a separate
        # B region would read out of bounds).
        b_idx = BinOp(T.u32, op=Arith.ADD,
                      lhs=BinOp(T.u32, op=Arith.MUL,
                                lhs=BinOp(T.u32, op=Arith.ADD, lhs=t_var, rhs=ty),
                                rhs=n_u),
                      rhs=col)

        a_slot = Subscript(T.u32, array=a_sh, index=lin(ty, tx))
        b_slot = Subscript(T.u32, array=b_sh, index=lin(ty, tx))

        inner_acc = Assign(acc, BinOp(T.u32, op=Arith.ADD, lhs=acc,
            rhs=BinOp(T.u32, op=Arith.MUL,
                      lhs=Subscript(T.u32, array=a_sh, index=lin(ty, k_var)),
                      rhs=Subscript(T.u32, array=b_sh, index=lin(k_var, tx)))))
        inner_loop = For(var=k_name, start=Lit(T.i32, value=0),
                         end=Lit(T.i32, value=ts), step=Lit(T.i32, value=1),
                         body=Block([inner_acc]))

        tile_body = Block([
            Assign(a_slot, Lit(T.u32, value=0)),
            If(cond=RelOp(T.boolean, op=Cmp.LT, lhs=a_idx, rhs=n_sq),
               then=Block([Assign(a_slot, Subscript(T.u32, array=input_var, index=a_idx))])),
            Assign(b_slot, Lit(T.u32, value=0)),
            If(cond=RelOp(T.boolean, op=Cmp.LT, lhs=b_idx, rhs=n_sq),
               then=Block([Assign(b_slot, Subscript(T.u32, array=input_var, index=b_idx))])),
            Barrier(),
            inner_loop,
            Barrier(),
        ])
        stmts.append(For(var=t_name, start=Lit(T.i32, value=0),
                         end=Lit(T.i32, value=n_tiles * ts),
                         step=Lit(T.i32, value=ts), body=tile_body))

        out_idx = BinOp(T.u32, op=Arith.ADD,
                        lhs=BinOp(T.u32, op=Arith.MUL, lhs=row, rhs=n_u), rhs=col)
        # In observable mode the assembler emits the sole global result write
        # (output[tid] = _fv).  Writing this template's row/column mapping to
        # the same buffer would race that epilogue across blocks.  Fold both
        # the GEMM value and its logical matrix index into the owning thread's
        # checksum instead.
        result_stmt = (self.observable_result(acc, out_idx)
                       if ctx.config.observable_fold
                       else Assign(Subscript(T.u32, array=output_var,
                                             index=out_idx), acc))
        stmts.append(If(cond=RelOp(T.boolean, op=Cmp.LT, lhs=out_idx, rhs=n_sq),
                        then=Block([result_stmt])))
        return stmts


# ---------------------------------------------------------------------------
# Scan (inclusive prefix, non-tid output)
# ---------------------------------------------------------------------------


_SCAN_OPS = {
    "add": Arith.ADD,
    "xor": Arith.BXOR,
    "or": Arith.BOR,
}


class ScanTemplate(KernelTemplate):
    """Block-level Blelloch scan in shared memory: two elements per thread,
    unrolled up-scan (reduce) + down-scan (distribute)."""

    def __init__(self) -> None:
        self._inclusive = True
        self._scan_op = "add"
        self._block_size = 128

    @property
    def name(self) -> str:
        return "prefix_scan"

    @property
    def features(self) -> Set[str]:
        return {"shared_mem", "sync", "scan"}

    def fetch_params(self) -> Dict[str, Any]:
        return {
            "inclusive": self._inclusive,
            "scan_op": self._scan_op,
            "block_size": self._block_size,
        }

    def kx_required_shared_mem(self) -> int:
        return 2 * self._block_size * 4

    def preferred_block(self) -> Tuple[int, int, int]:
        return (self._block_size, 1, 1)

    def preferred_grid(self) -> Tuple[int, int, int]:
        return (64, 1, 1)

    def uses_non_tid_output(self) -> bool:
        # The structured scan layout is idx = 2*blockIdx.x*blockDim.x +
        # threadIdx.x rather than tid. Observable mode folds that logical
        # mapping into _fv; non-observable mode still writes it directly.
        return True

    def build_body(self, ctx: GenContext) -> List[Stmt]:
        bs = self._block_size
        op = _SCAN_OPS.get(self._scan_op, Arith.ADD)
        tid = ctx.kx_scope.lookup("tid")
        n_var = ctx.kx_scope.lookup("n")
        output_var = ctx.kx_scope.lookup("output")
        input_var = ctx.kx_scope.lookup("input")
        n_u = _u(n_var)
        lane = ThreadBuiltin(T.u32, axis="x", source="threadIdx")
        bdx = ThreadBuiltin(T.u32, axis="x", source="blockDim")
        bx = ThreadBuiltin(T.u32, axis="x", source="blockIdx")

        temp_name = ctx.kx_fresh_name("temp")
        temp = Var(TArray(T.u32, 2 * bs), name=temp_name, space=MemSpace.SHARED)
        ctx.kx_declare(temp)

        # idx = 2 * blockIdx.x * blockDim.x + threadIdx.x
        idx_name = ctx.kx_fresh_name("idx")
        idx = Var(T.u32, name=idx_name)
        ctx.kx_declare(idx)
        idx_expr = BinOp(T.u32, op=Arith.ADD,
            lhs=BinOp(T.u32, op=Arith.MUL, lhs=Lit(T.u32, value=2),
                      rhs=BinOp(T.u32, op=Arith.MUL, lhs=bx, rhs=bdx)),
            rhs=lane)
        idx_bdx = BinOp(T.u32, op=Arith.ADD, lhs=idx, rhs=bdx)
        lane_bdx = BinOp(T.u32, op=Arith.ADD, lhs=lane, rhs=bdx)

        stmts: List[Stmt] = [
            Declare(name=temp_name, gtype=TArray(T.u32, 2 * bs),
                    space=MemSpace.SHARED, quals=[Qual.SHARED_MEM]),
            Declare(name=idx_name, gtype=T.u32, init=idx_expr),
        ]

        # Load two elements per thread (zero-fill, then a guarded load so
        # every shared slot is initialised).
        stmts.append(Assign(Subscript(T.u32, array=temp, index=lane),
                            Lit(T.u32, value=0)))
        stmts.append(If(cond=RelOp(T.boolean, op=Cmp.LT, lhs=idx, rhs=n_u),
                        then=Block([Assign(
                            Subscript(T.u32, array=temp, index=lane),
                            Subscript(T.u32, array=input_var, index=idx))])))
        stmts.append(Assign(Subscript(T.u32, array=temp, index=lane_bdx),
                            Lit(T.u32, value=0)))
        stmts.append(If(cond=RelOp(T.boolean, op=Cmp.LT, lhs=idx_bdx, rhs=n_u),
                        then=Block([Assign(
                            Subscript(T.u32, array=temp, index=lane_bdx),
                            Subscript(T.u32, array=input_var, index=idx_bdx))])))

        def sweep_idx(off: int, add: int) -> Expr:
            # off * (2*lane + add) - 1
            return BinOp(T.u32, op=Arith.SUB,
                lhs=BinOp(T.u32, op=Arith.MUL, lhs=Lit(T.u32, value=off),
                          rhs=BinOp(T.u32, op=Arith.ADD,
                                    lhs=BinOp(T.u32, op=Arith.MUL,
                                              lhs=Lit(T.u32, value=2), rhs=lane),
                                    rhs=Lit(T.u32, value=add))),
                rhs=Lit(T.u32, value=1))

        # Up-scan (reduce) phase, unrolled.
        offset = 1
        d = bs
        while d > 0:
            stmts.append(Barrier())
            ai = sweep_idx(offset, 2)
            bi = sweep_idx(offset, 1)
            stmts.append(If(cond=RelOp(T.boolean, op=Cmp.LT, lhs=lane,
                                       rhs=Lit(T.u32, value=d)),
                            then=Block([Assign(
                                Subscript(T.u32, array=temp, index=ai),
                                BinOp(T.u32, op=op,
                                      lhs=Subscript(T.u32, array=temp, index=ai),
                                      rhs=Subscript(T.u32, array=temp,
                                                    index=bi)))])))
            d //= 2
            offset *= 2
            if offset > 2 * bs:
                break
        stmts.append(Barrier())

        # Clear the last element for an exclusive scan.
        if not self._inclusive:
            stmts.append(If(cond=RelOp(T.boolean, op=Cmp.EQ, lhs=lane,
                                       rhs=Lit(T.u32, value=0)),
                            then=Block([Assign(
                                Subscript(T.u32, array=temp,
                                          index=Lit(T.u32, value=2 * bs - 1)),
                                Lit(T.u32, value=0))])))

        # Down-scan (distribute) phase, unrolled.
        d2 = 1
        offset2 = offset // 2
        while offset2 >= 1:
            stmts.append(Barrier())
            ai = sweep_idx(offset2, 2)
            bi = sweep_idx(offset2, 1)
            sw_name = ctx.kx_fresh_name("sw")
            sw = Var(T.u32, name=sw_name)
            ctx.kx_declare(sw)
            # Declare the swap variable at function kx_scope (not inside the if
            # block) to avoid a C++ kx_scope mismatch with the kx_scope tracker.
            stmts.append(Declare(name=sw_name, gtype=T.u32,
                                 init=Lit(T.u32, value=0)))
            stmts.append(If(cond=RelOp(T.boolean, op=Cmp.LT, lhs=lane,
                                       rhs=Lit(T.u32, value=d2)),
                            then=Block([
                                Assign(sw, Subscript(T.u32, array=temp, index=bi)),
                                Assign(Subscript(T.u32, array=temp, index=bi),
                                       Subscript(T.u32, array=temp, index=ai)),
                                Assign(Subscript(T.u32, array=temp, index=ai),
                                       BinOp(T.u32, op=op,
                                             lhs=Subscript(T.u32, array=temp,
                                                           index=ai),
                                             rhs=sw)),
                            ])))
            d2 *= 2
            offset2 //= 2
        stmts.append(Barrier())

        # Route both scan results. In observable mode these are thread-local
        # folds because the assembler owns output[tid]; in legacy/performance
        # mode the template retains its natural two-elements-per-thread output.
        first_result = Subscript(T.u32, array=temp, index=lane)
        second_result = Subscript(T.u32, array=temp, index=lane_bdx)
        first_stmt = (self.observable_result(first_result, idx)
                      if ctx.config.observable_fold
                      else Assign(Subscript(T.u32, array=output_var, index=idx),
                                  first_result))
        second_stmt = (self.observable_result(second_result, idx_bdx)
                       if ctx.config.observable_fold
                       else Assign(Subscript(T.u32, array=output_var,
                                             index=idx_bdx), second_result))
        stmts.append(If(cond=RelOp(T.boolean, op=Cmp.LT, lhs=idx, rhs=n_u),
                        then=Block([first_stmt])))
        stmts.append(If(cond=RelOp(T.boolean, op=Cmp.LT, lhs=idx_bdx, rhs=n_u),
                        then=Block([second_stmt])))
        return stmts


# ---------------------------------------------------------------------------
# Histogram (shared-memory privatized atomics)
# ---------------------------------------------------------------------------



class HistogramTemplate(KernelTemplate):
    """Bin inputs with shared-memory atomic counters, flush per block."""

    def __init__(self) -> None:
        self._num_bins = 64
        self._block_size = 256

    @property
    def name(self) -> str:
        return "bin_count"

    @property
    def features(self) -> Set[str]:
        return {"shared_mem", "atomics", "sync"}

    def fetch_params(self) -> Dict[str, Any]:
        return {"num_bins": self._num_bins, "block_size": self._block_size}

    def kx_required_shared_mem(self) -> int:
        return self._num_bins * 4

    def preferred_block(self) -> Tuple[int, int, int]:
        return (self._block_size, 1, 1)

    def build_body(self, ctx: GenContext) -> List[Stmt]:
        nb = self._num_bins
        bs = self._block_size
        tid = ctx.kx_scope.lookup("tid")
        n_var = ctx.kx_scope.lookup("n")
        output_var = ctx.kx_scope.lookup("output")
        input_var = ctx.kx_scope.lookup("input")
        n_u = _u(n_var)
        lane = ThreadBuiltin(T.u32, axis="x", source="threadIdx")
        nb_c = Lit(T.u32, value=nb)

        bins_name = ctx.kx_fresh_name("bins")
        bins = Var(TArray(T.u32, nb), name=bins_name, space=MemSpace.SHARED)
        ctx.kx_declare(bins)

        stmts: List[Stmt] = [
            Declare(name=bins_name, gtype=TArray(T.u32, nb),
                    space=MemSpace.SHARED, quals=[Qual.SHARED_MEM]),
        ]
        # Zero the shared bins, strided so num_bins > block_size is covered.
        strides = (nb + bs - 1) // bs

        def stride_idx(s: int) -> Expr:
            if s == 0:
                return lane
            return BinOp(T.u32, op=Arith.ADD, lhs=lane,
                         rhs=Lit(T.u32, value=s * bs))

        for s in range(strides):
            zidx = stride_idx(s)
            stmts.append(If(
                cond=RelOp(T.boolean, op=Cmp.LT, lhs=zidx, rhs=nb_c),
                then=Block([Assign(Subscript(T.u32, array=bins, index=zidx),
                                   Lit(T.u32, value=0))])))
        stmts.append(Barrier())

        if self.kx_flat_mode(ctx):
            # Deterministic binning: each thread bumps bins[(input ^ tid) % nb]
            # and folds the *bin value it hit* into the pool.  Deterministic
            # because reads happen after a barrier.
            val = Subscript(T.u32, array=input_var,
                            index=SafeBin(T.u32, op=SafeOp.SMOD, lhs=tid, rhs=n_u))
            bin_idx = SafeBin(T.u32, op=SafeOp.SMOD,
                              lhs=BinOp(T.u32, op=Arith.BXOR, lhs=val, rhs=tid),
                              rhs=nb_c)
            stmts.append(ExprStmt(Atomic(T.u32, what="Add",
                                         addr=Subscript(T.u32, array=bins, index=bin_idx),
                                         operand=Lit(T.u32, value=1))))
            stmts.append(Barrier())
            hit = Subscript(T.u32, array=bins, index=bin_idx)
            stmts.extend(self.flat_result(
                ctx, BinOp(T.u32, op=Arith.BXOR, lhs=hit, rhs=val), T.u32))
            stmts.append(Barrier())
            return stmts

        # Structured: guarded binning (hval/bin declared inside the guard,
        # with expression noise folded into the loaded value), then a
        # BLOCK-0-ONLY flush into output[0..nb).  Only block 0 writes, and
        # the per-iteration reset of those slots is done by block 0's own
        # threads (nb <= block size), so the block barrier orders
        # reset < flush and the result is deterministic.  A wider flush
        # would race the reset across blocks.
        expr_gen = ExprGen(ctx)
        load = BinOp(T.u32, op=Arith.BXOR,
                     lhs=Subscript(T.u32, array=input_var, index=tid),
                     rhs=expr_gen.kx_generate(T.u32, depth=3))
        val_name = ctx.kx_fresh_name("hval")
        bin_name = ctx.kx_fresh_name("bin")
        val = Var(T.u32, name=val_name)
        bin_var = Var(T.u32, name=bin_name)
        stmts.append(If(cond=RelOp(T.boolean, op=Cmp.LT, lhs=tid, rhs=n_u),
                        then=Block([
                            Declare(name=val_name, gtype=T.u32, init=load),
                            Declare(name=bin_name, gtype=T.u32,
                                    init=SafeBin(T.u32, op=SafeOp.SMOD,
                                                 lhs=val, rhs=nb_c)),
                            ExprStmt(Atomic(T.u32, what="Add",
                                            addr=Subscript(T.u32, array=bins,
                                                           index=bin_var),
                                            operand=Lit(T.u32, value=1))),
                        ])))
        stmts.append(Barrier())
        # Flush, strided the same way as the zero-init.
        write_stmts: List[Stmt] = []
        for s in range(strides):
            widx = stride_idx(s)
            write_stmts.append(If(
                cond=RelOp(T.boolean, op=Cmp.LT, lhs=widx, rhs=nb_c),
                then=Block([Assign(Subscript(T.u32, array=output_var, index=widx),
                                   Subscript(T.u32, array=bins, index=widx))])))
        stmts.append(If(cond=RelOp(T.boolean, op=Cmp.EQ,
                                   lhs=ThreadBuiltin(T.u32, axis="x", source="blockIdx"),
                                   rhs=Lit(T.u32, value=0)),
                        then=Block(write_stmts)))
        stmts.append(Barrier())
        return stmts


# ---------------------------------------------------------------------------
# Transpose (2D, padded shared tile)
# ---------------------------------------------------------------------------



class TransposeTemplate(KernelTemplate):
    """Tiled transpose via a padded shared tile (+1 anti bank conflict)."""

    def __init__(self) -> None:
        self._tile_dim = 32
        self._padding = 1   # +1 avoids bank conflicts

    @property
    def name(self) -> str:
        return "tile_swap"

    @property
    def features(self) -> Set[str]:
        return {"shared_mem", "sync", "coalesced_access"}

    def fetch_params(self) -> Dict[str, Any]:
        return {
            "tile_dim": self._tile_dim,
            "padding": self._padding,
        }

    def kx_required_shared_mem(self) -> int:
        td = self._tile_dim
        return td * (td + self._padding) * 4

    def preferred_block(self) -> Tuple[int, int, int]:
        td = self._tile_dim
        return (td, td, 1)

    def preferred_grid(self) -> Tuple[int, int, int]:
        return (4, 4, 1)

    def kx_required_n_elements(self) -> int:
        return self.preferred_grid()[0] * self._tile_dim

    def kx_is_2d(self) -> bool:
        return True

    def build_body(self, ctx: GenContext) -> List[Stmt]:
        td = self._tile_dim
        row_size = td + self._padding
        n_var = ctx.kx_scope.lookup("n")
        output_var = ctx.kx_scope.lookup("output")
        input_var = ctx.kx_scope.lookup("input")
        n_u = _u(n_var)

        tile_name = ctx.kx_fresh_name("tile")
        tile = Var(TArray(T.u32, td * row_size), name=tile_name, space=MemSpace.SHARED)
        ctx.kx_declare(tile)

        tx_name, ty_name = ctx.kx_fresh_name("tx"), ctx.kx_fresh_name("ty")
        x_name, y_name = ctx.kx_fresh_name("x"), ctx.kx_fresh_name("y")
        tx = Var(T.u32, name=tx_name)
        ty = Var(T.u32, name=ty_name)
        x = Var(T.u32, name=x_name)
        y = Var(T.u32, name=y_name)
        for v in (tx, ty, x, y):
            ctx.kx_declare(v)

        bx = ThreadBuiltin(T.u32, axis="x", source="blockIdx")
        by = ThreadBuiltin(T.u32, axis="y", source="blockIdx")
        td_c = Lit(T.u32, value=td)
        row_c = Lit(T.u32, value=row_size)

        def lin(r: Expr, c: Expr) -> Expr:
            return BinOp(T.u32, op=Arith.ADD,
                         lhs=BinOp(T.u32, op=Arith.MUL, lhs=r, rhs=row_c), rhs=c)

        # x = blockIdx.x * TD + tx;  y = blockIdx.y * TD + ty
        stmts: List[Stmt] = [
            Declare(name=tile_name, gtype=TArray(T.u32, td * row_size),
                    space=MemSpace.SHARED, quals=[Qual.SHARED_MEM]),
            Declare(name=tx_name, gtype=T.u32,
                    init=ThreadBuiltin(T.u32, axis="x", source="threadIdx")),
            Declare(name=ty_name, gtype=T.u32,
                    init=ThreadBuiltin(T.u32, axis="y", source="threadIdx")),
            Declare(name=x_name, gtype=T.u32,
                    init=BinOp(T.u32, op=Arith.ADD,
                               lhs=BinOp(T.u32, op=Arith.MUL, lhs=bx, rhs=td_c), rhs=tx)),
            Declare(name=y_name, gtype=T.u32,
                    init=BinOp(T.u32, op=Arith.ADD,
                               lhs=BinOp(T.u32, op=Arith.MUL, lhs=by, rhs=td_c), rhs=ty)),
        ]

        # Zero the tile, then a nested-guard load so every slot is
        # initialised even when the matrix edge cuts through the tile.
        tile_slot = Subscript(T.u32, array=tile, index=lin(ty, tx))
        stmts.append(Assign(tile_slot, Lit(T.u32, value=0)))
        in_idx = BinOp(T.u32, op=Arith.ADD,
                       lhs=BinOp(T.u32, op=Arith.MUL, lhs=y, rhs=n_u), rhs=x)
        stmts.append(If(cond=RelOp(T.boolean, op=Cmp.LT, lhs=y, rhs=n_u),
                        then=Block([If(
                            cond=RelOp(T.boolean, op=Cmp.LT, lhs=x, rhs=n_u),
                            then=Block([Assign(
                                tile_slot,
                                Subscript(T.u32, array=input_var,
                                          index=in_idx))]))])))
        stmts.append(Barrier())

        # Transposed coordinates: swap block axes and in-tile axes.
        x2_name, y2_name = ctx.kx_fresh_name("x2"), ctx.kx_fresh_name("y2")
        x2 = Var(T.u32, name=x2_name)
        y2 = Var(T.u32, name=y2_name)
        ctx.kx_declare(x2)
        ctx.kx_declare(y2)
        stmts.append(Declare(name=x2_name, gtype=T.u32,
                             init=BinOp(T.u32, op=Arith.ADD,
                                        lhs=BinOp(T.u32, op=Arith.MUL,
                                                  lhs=by, rhs=td_c), rhs=tx)))
        stmts.append(Declare(name=y2_name, gtype=T.u32,
                             init=BinOp(T.u32, op=Arith.ADD,
                                        lhs=BinOp(T.u32, op=Arith.MUL,
                                                  lhs=bx, rhs=td_c), rhs=ty)))
        out_idx = BinOp(T.u32, op=Arith.ADD,
                        lhs=BinOp(T.u32, op=Arith.MUL, lhs=y2, rhs=n_u), rhs=x2)
        transposed = Subscript(T.u32, array=tile, index=lin(tx, ty))
        result_stmt = (self.observable_result(transposed, out_idx)
                       if ctx.config.observable_fold
                       else Assign(Subscript(T.u32, array=output_var,
                                             index=out_idx), transposed))
        stmts.append(If(cond=RelOp(T.boolean, op=Cmp.LT, lhs=y2, rhs=n_u),
                        then=Block([If(
                            cond=RelOp(T.boolean, op=Cmp.LT, lhs=x2, rhs=n_u),
                            then=Block([result_stmt]))])))
        return stmts

