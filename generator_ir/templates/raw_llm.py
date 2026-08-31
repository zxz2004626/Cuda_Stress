"""Raw-LLM baseline template families for RQ1 Arm B.

These templates intentionally live outside the default template catalog.  They
model a "prompted LLM, no PassSpec" baseline: reusable CUDA source-level
families for broad optimization-testing areas, without artifact signals,
reverse-engineered trigger constraints, inline PTX, or inline assembly.
"""

from __future__ import annotations

from typing import List, Set, Tuple

from generator_ir.gen.context import GenContext
from generator_ir.ir.nodes import (
    BinOp,
    Call,
    Cast,
    Expr,
    Lit,
    RelOp,
    SafeBin,
    Subscript,
    ThreadBuiltin,
    Var,
)
from generator_ir.ir.operators import Arith, Cmp, SafeOp
from generator_ir.ir.statements import (
    Assign,
    Barrier,
    Block,
    Case,
    Declare,
    Stmt,
    Switch,
)
from generator_ir.ir.typedecl import MemSpace, Qual, T, TArray
from generator_ir.templates.catalog import TemplateRegistry
from generator_ir.templates.skeleton import KernelTemplate


def _u32(v: int) -> Lit:
    return Lit(T.u32, value=int(v) & 0xFFFFFFFF)


def _i32(v: int) -> Lit:
    return Lit(T.i32, value=int(v))


def _cast_u32(e: Expr) -> Cast:
    return Cast(T.u32, to=T.u32, arg=e)


def _add(a: Expr, b: Expr) -> BinOp:
    return BinOp(T.u32, op=Arith.ADD, lhs=a, rhs=b)


def _xor(a: Expr, b: Expr) -> BinOp:
    return BinOp(T.u32, op=Arith.BXOR, lhs=a, rhs=b)


def _mul(a: Expr, b: Expr) -> BinOp:
    return BinOp(T.u32, op=Arith.MUL, lhs=a, rhs=b)


def _shl(a: Expr, b: Expr) -> BinOp:
    return BinOp(T.u32, op=Arith.SHL, lhs=a, rhs=b)


def _shr(a: Expr, b: Expr) -> BinOp:
    return BinOp(T.u32, op=Arith.SHR, lhs=a, rhs=b)


def _band(a: Expr, b: Expr) -> BinOp:
    return BinOp(T.u32, op=Arith.BAND, lhs=a, rhs=b)


def _mod(a: Expr, b: Expr) -> SafeBin:
    return SafeBin(T.u32, op=SafeOp.SMOD, lhs=a, rhs=b)


def _thread_x() -> ThreadBuiltin:
    return ThreadBuiltin(T.u32, axis="x", source="threadIdx")


class _RawLLMTemplate(KernelTemplate):
    """Shared helpers for simple, source-level Raw-LLM families."""

    _block_size = 256

    def preferred_block(self) -> Tuple[int, int, int]:
        return (self._block_size, 1, 1)

    def preferred_grid(self) -> Tuple[int, int, int]:
        return (128, 1, 1)

    def _scope(self, ctx: GenContext):
        tid = ctx.kx_scope.lookup("tid")
        n_var = ctx.kx_scope.lookup("n")
        output_var = ctx.kx_scope.lookup("output")
        input_var = ctx.kx_scope.lookup("input")
        n_u = _cast_u32(n_var)
        return tid, n_var, output_var, input_var, n_u

    def _input_at(self, ctx: GenContext, offset: int = 0) -> Subscript:
        tid, _n_var, _output_var, input_var, n_u = self._scope(ctx)
        idx = _mod(_add(tid, _u32(offset)), n_u)
        return Subscript(T.u32, array=input_var, index=idx)

    def _declare_u32(self, ctx: GenContext, prefix: str, init: Expr) -> tuple[Var, Declare]:
        name = ctx.kx_fresh_name(prefix)
        var = Var(T.u32, name=name)
        ctx.kx_declare(var)
        return var, Declare(name=name, gtype=T.u32, init=init)

    def _declare_shared(self, ctx: GenContext, prefix: str, size: int) -> tuple[Var, Declare]:
        name = ctx.kx_fresh_name(prefix)
        var = Var(TArray(T.u32, size), name=name, space=MemSpace.SHARED)
        ctx.kx_declare(var)
        decl = Declare(
            name=name,
            gtype=TArray(T.u32, size),
            space=MemSpace.SHARED,
            quals=[Qual.SHARED_MEM],
        )
        return var, decl

    def _sink(self, ctx: GenContext, value: Expr) -> List[Stmt]:
        if self.kx_flat_mode(ctx):
            sink, decl = ctx.kx_sample_or_create(T.u32, for_write=True, init=value)
            if decl is not None:
                return [decl]
            return [Assign(sink, value)]

        tid, _n_var, output_var, _input_var, _n_u = self._scope(ctx)
        return [
            Assign(
                Subscript(T.u32, array=output_var, index=tid),
                _cast_u32(value),
            )
        ]


class RawLLMSyncTemplate(_RawLLMTemplate):
    """Naive synchronization family: compute locals, synchronize, combine."""

    @property
    def name(self) -> str:
        return "rawllm_sync_span_live"

    @property
    def features(self) -> Set[str]:
        return {"raw_llm", "sync", "shared_mem"}

    def kx_required_shared_mem(self) -> int:
        return self._block_size * 4

    def build_body(self, ctx: GenContext) -> List[Stmt]:
        smem, smem_decl = self._declare_shared(ctx, "rl_sync_smem", self._block_size)
        local = _mod(_thread_x(), _u32(self._block_size))
        live_count = ctx.rnd_choice([1, 2, 2, 3, 3, 4, 5])
        stmts: List[Stmt] = [smem_decl]
        values: List[Var] = []
        for i in range(live_count):
            mixed = _xor(
                self._input_at(ctx, i),
                _add(_mul(_thread_x(), _u32(17 + i * 5)), _u32(0x9E3779B9 + i)),
            )
            var, decl = self._declare_u32(ctx, "rl_sync_v", mixed)
            values.append(var)
            stmts.append(decl)

        if ctx.rnd_bool(0.70):
            stmts.append(Assign(Subscript(T.u32, array=smem, index=local), values[0]))

        stmts.append(Barrier())

        acc: Expr = values[0]
        for v in values[1:]:
            acc = _xor(acc, v)
        if ctx.rnd_bool(0.75):
            peer = Subscript(
                T.u32,
                array=smem,
                index=_mod(_add(_thread_x(), _u32(1)), _u32(self._block_size)),
            )
            acc = _xor(acc, peer)

        stmts.extend(self._sink(ctx, acc))
        stmts.append(Barrier())
        return stmts


class RawLLMControlFlowTemplate(_RawLLMTemplate):
    """Naive control-flow family: thread-dependent switch over mixed input."""

    @property
    def name(self) -> str:
        return "rawllm_cf_switch_tree"

    @property
    def features(self) -> Set[str]:
        return {"raw_llm", "control_flow", "switch"}

    def build_body(self, ctx: GenContext) -> List[Stmt]:
        key_seed = _xor(self._input_at(ctx, 0), _mul(_thread_x(), _u32(3)))
        key, key_decl = self._declare_u32(ctx, "rl_key", key_seed)
        acc, acc_decl = self._declare_u32(ctx, "rl_cf_acc", self._input_at(ctx, 1))
        case_count = ctx.rnd_choice([2, 2, 3, 3, 4, 5])
        cases: List[Case] = []
        for i in range(case_count):
            rhs = _xor(
                acc,
                _add(self._input_at(ctx, i + 2), _u32(0x45D9F3B + i * 97)),
            )
            cases.append(Case(value=_u32(i), body=Block([Assign(acc, rhs)])))

        sw = Switch(expr=_mod(key, _u32(case_count)), cases=cases)
        return [key_decl, acc_decl, sw] + self._sink(ctx, acc)


class RawLLMSharedMemoryTemplate(_RawLLMTemplate):
    """Naive shared-memory family: tile, synchronize, optionally read back."""

    @property
    def name(self) -> str:
        return "rawllm_smem_bank_stress"

    @property
    def features(self) -> Set[str]:
        return {"raw_llm", "shared_mem"}

    def kx_required_shared_mem(self) -> int:
        return self._block_size * 4

    def build_body(self, ctx: GenContext) -> List[Stmt]:
        smem, smem_decl = self._declare_shared(ctx, "rl_smem", self._block_size)
        slot = _mod(_thread_x(), _u32(self._block_size))
        val, val_decl = self._declare_u32(
            ctx,
            "rl_smem_val",
            _xor(self._input_at(ctx, 0), _u32(0xA5A5A5A5)),
        )
        variant = ctx.rnd_choice(["tile_readback", "tile_readback", "write_only", "local_read"])
        stmts: List[Stmt] = [smem_decl, val_decl]
        stmts.append(Assign(Subscript(T.u32, array=smem, index=slot), val))

        if variant == "local_read":
            before, before_decl = self._declare_u32(
                ctx, "rl_smem_before", Subscript(T.u32, array=smem, index=slot)
            )
            stmts.append(before_decl)
            stmts.append(Barrier())
            stmts.extend(self._sink(ctx, _xor(before, val)))
            return stmts

        stmts.append(Barrier())
        if variant == "write_only":
            stmts.extend(self._sink(ctx, val))
            return stmts

        peer = Subscript(
            T.u32,
            array=smem,
            index=_mod(_add(_thread_x(), _u32(1)), _u32(self._block_size)),
        )
        out, out_decl = self._declare_u32(ctx, "rl_smem_out", _xor(peer, val))
        stmts.append(out_decl)
        stmts.extend(self._sink(ctx, out))
        stmts.append(Barrier())
        return stmts


class RawLLMIntegerTemplate(_RawLLMTemplate):
    """Naive integer-codegen family: bit mixing plus occasional CUDA intrinsics."""

    @property
    def name(self) -> str:
        return "rawllm_isa_stress"

    @property
    def features(self) -> Set[str]:
        return {"raw_llm", "integer", "bit_ops"}

    def build_body(self, ctx: GenContext) -> List[Stmt]:
        a, a_decl = self._declare_u32(ctx, "rl_isa_a", self._input_at(ctx, 0))
        b, b_decl = self._declare_u32(
            ctx,
            "rl_isa_b",
            _xor(self._input_at(ctx, 1), _mul(_thread_x(), _u32(1103515245))),
        )
        mode = ctx.rnd_choice([
            "generic_mix",
            "generic_mix",
            "byte_perm",
            "mulhi",
            "both",
        ])
        stmts: List[Stmt] = [a_decl, b_decl]
        generic = _xor(
            _xor(_shl(a, _u32(5)), _shr(b, _u32(7))),
            _mul(_xor(a, b), _u32(1664525)),
        )
        result: Expr = generic
        if mode in {"byte_perm", "both"}:
            perm = Call(
                T.u32,
                fn="__byte_perm",
                args=[a, b, _u32(ctx.rnd_choice([0x0123, 0x1032, 0x5410, 0x7654]))],
            )
            pvar, pdecl = self._declare_u32(ctx, "rl_perm", perm)
            stmts.append(pdecl)
            result = _xor(result, pvar)
        if mode in {"mulhi", "both"}:
            hi = Call(T.u32, fn="__umulhi", args=[a, b])
            hvar, hdecl = self._declare_u32(ctx, "rl_hi", hi)
            stmts.append(hdecl)
            result = _xor(result, hvar)

        stmts.extend(self._sink(ctx, result))
        return stmts


def kx_build_raw_llm_registry() -> TemplateRegistry:
    reg = TemplateRegistry()
    reg.register(RawLLMSyncTemplate())
    reg.register(RawLLMControlFlowTemplate())
    reg.register(RawLLMSharedMemoryTemplate())
    reg.register(RawLLMIntegerTemplate())
    return reg
