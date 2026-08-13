"""Template base class and shared helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from generator_ir.ir.typedecl import (
    GType, T, TUInt, TUShort, TULong,
)
from generator_ir.ir.operators import Arith, SafeOp
from generator_ir.ir.nodes import Expr, Lit, Var, BinOp, Subscript, SafeBin, Cast
from generator_ir.ir.statements import Stmt, Assign, Declare

if TYPE_CHECKING:
    from generator_ir.gen.context import GenContext

# Integer types safe to fold into the deterministic checksum.
FLAT_SINK_TYPES = (TUInt, TUShort, TULong)
# Names a template result must never overwrite.
FLAT_RESERVED = {"tid", "gid", "n", "output", "input",
                 "_fv", "_rep", "_ci", "_fv_sink", "_ov_sink"}


class KernelTemplate(ABC):
    """A structural kernel pattern with generator-filled holes."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Template registry name."""

    @property
    @abstractmethod
    def features(self) -> Set[str]:
        """GPU feature tags this template exercises."""

    @abstractmethod
    def build_body(self, ctx: "GenContext") -> List[Stmt]:
        """One iteration of the kernel body (the generator wraps it in the
        body-repeat loop)."""

    def fetch_params(self) -> Dict[str, Any]:
        return {}

    def kx_parameterize(self, params: Dict[str, Any]) -> None:
        for key, val in params.items():
            attr = f"_{key}"
            if hasattr(self, attr):
                setattr(self, attr, val)

    def kx_required_shared_mem(self) -> int:
        return 0

    def preferred_block(self) -> Tuple[int, int, int]:
        return (256, 1, 1)

    def preferred_grid(self) -> Tuple[int, int, int]:
        return (128, 1, 1)

    def kx_required_n_elements(self) -> int:
        total = 1
        for d in self.preferred_block():
            total *= d
        for d in self.preferred_grid():
            total *= d
        return max(total, 1024)

    def kx_is_2d(self) -> bool:
        return False

    def uses_non_tid_output(self) -> bool:
        """True when output indexing is not output[tid] (e.g. scan)."""
        return False

    # ------------------------------------------------------------------
    # observable-fold support
    # ------------------------------------------------------------------

    def _uses_flat_emission(self, ctx: "GenContext") -> bool:
        return bool(getattr(ctx.config, "observable_fold", False)) and not ctx.kx_is_2d

    # Backward-compatible aliases for subclasses.
    def kx_flat_mode(self, ctx: "GenContext") -> bool:
        return self._uses_flat_emission(ctx)

    def kx_flat_body(self, ctx: "GenContext") -> List[Stmt]:
        return self._synthetic_flat_sequence(ctx)

    def flat_sinks(self, ctx: "GenContext", k: int = 1) -> List[Var]:
        """Pick up to k in-kx_scope function-kx_scope integer vars to overwrite."""
        cands = [v for v in ctx.kx_function_scope_variables()
                 if isinstance(v.ty, FLAT_SINK_TYPES)
                 and v.name not in FLAT_RESERVED
                 and not v.name.startswith("i_")]
        picks: List[Var] = []
        pool = list(cands)
        for _ in range(min(k, len(pool))):
            v = ctx.rnd_choice(pool)
            pool.remove(v)
            picks.append(v)
        return picks

    def _emit_flat_store(self, sink: Var, value: Expr) -> Stmt:
        return Assign(Var(sink.ty, name=sink.name),
                      Cast(sink.ty, to=sink.ty, arg=value))

    def flat_result(self, ctx: "GenContext", value: Expr,
                    result_type: Optional[GType] = None) -> List[Stmt]:
        sinks = self.flat_sinks(ctx, k=1)
        if sinks:
            return [self._emit_flat_store(sinks[0], value)]
        rt = result_type or T.u32
        name = ctx.kx_fresh_name("v")
        ctx.kx_declare(Var(rt, name=name))
        return [Declare(name=name, gtype=rt, init=value)]

    def observable_result(self, value: Expr,
                          logical_index: Optional[Expr] = None) -> Stmt:
        """Fold a structured template result into the thread-local checksum.

        In observable mode the assembler owns ``output`` and gives every slot
        exactly one writer via its final ``output[tid] = _fv`` epilogue.  A
        structured template whose natural result layout is not ``tid`` must
        therefore route its result through ``_fv`` rather than write the same
        output buffer using a second index mapping.  Including the logical
        result index keeps that mapping observable without a global-memory
        write or any need for grid-wide synchronization.
        """
        value_u = Cast(T.u32, to=T.u32, arg=value)
        if logical_index is not None:
            index_u = Cast(T.u32, to=T.u32, arg=logical_index)
            value_u = BinOp(
                T.u32, op=Arith.BXOR, lhs=value_u,
                rhs=BinOp(T.u32, op=Arith.MUL, lhs=index_u,
                          rhs=Lit(T.u32, value=0x85EBCA6B)))
        fv = Var(T.u32, name="_fv")
        return Assign(
            fv,
            BinOp(T.u32, op=Arith.BXOR,
                  lhs=BinOp(T.u32, op=Arith.MUL, lhs=fv,
                            rhs=Lit(T.u32, value=0x9E3779B1)),
                  rhs=value_u))

    def _synthetic_flat_sequence(self, ctx: "GenContext") -> List[Stmt]:
        """Default flat body: load input[tid % n], a few generated statements,
        fold into the shared variable pool."""
        from generator_ir.gen.expr_builder import ExprGen
        from generator_ir.gen.stmt_builder import StmtGen
        expr_gen = ExprGen(ctx)
        stmt_gen = StmtGen(ctx, expr_gen)
        tid = ctx.kx_scope.lookup("tid")
        n_var = ctx.kx_scope.lookup("n")
        input_var = ctx.kx_scope.lookup("input")
        n_u = Cast(T.u32, to=T.u32, arg=n_var)
        val_name = ctx.kx_fresh_name("val")
        val = Var(T.u32, name=val_name)
        ctx.kx_declare(val)
        stmts: List[Stmt] = [
            Declare(name=val_name, gtype=T.u32,
                    init=Subscript(T.u32, array=input_var,
                                   index=SafeBin(T.u32, op=SafeOp.SMOD,
                                                 lhs=tid, rhs=n_u))),
        ]
        for _ in range(4):
            stmts.extend(stmt_gen.build_stmts(1))
        result = BinOp(T.u32, op=Arith.BXOR, lhs=val,
                       rhs=expr_gen.kx_generate(T.u32))
        stmts.extend(self.flat_result(ctx, result, T.u32))
        return stmts
