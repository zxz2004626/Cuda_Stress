"""Weaving blocks: small composable code-generation units.

Each block consumes values from the shared pool, emits a few statements via
``ctx.emit``/``ctx.fresh``, and folds results into the running checksum via
``ctx.observe``.  Selection is per-seed swarm-weighted with no static bias.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from generator_ir.ir.typedecl import T, GType, TArray, MemSpace, Qual
from generator_ir.ir.operators import Arith, Cmp, SafeOp
from generator_ir.ir.nodes import (
    Expr, Lit, Var, BinOp, RelOp, Cast, SafeBin, Subscript, Call,
    ThreadBuiltin,
)
from generator_ir.ir.statements import (
    Stmt, Assign, Declare, Block, If, For, Barrier, ExprStmt,
)


@dataclass
class WeaveBlock:
    """A composable generation block (Fuzzilli-style CodeGenerator)."""
    name: str
    fn: Callable[[object], None]
    produces: tuple = ()
    inputs: tuple = ()
    requires_uniform: bool = False
    weight: float = 1.0
    cost: int = 1
    backends: tuple = ("cuda", "hip")
    once: bool = False

    def can_execute(self, ctx) -> bool:
        if self.weight <= 0.0:
            # Disabled blocks (feature-coverage gens at weight=0) must never
            # enter the selection pool -- pick_block's max(1e-6, ...) clamp
            # would otherwise resurrect them with a tiny nonzero probability.
            return False
        backend = getattr(ctx.config, "backend", "cuda")
        if backend not in self.backends:
            return False
        if self.name.startswith("fp_") and not getattr(ctx, "use_float_types", False):
            return False
        if self.once and getattr(ctx, "fired_blocks", {}).get(self.name, 0) >= 1:
            return False
        if self.requires_uniform:
            if not ctx.snapshot.uniform:
                return False
            if not getattr(ctx, "_block_is_1d", True):
                return False
        # Classic-template fragments only have a pool-sharing flat form in 1D
        # kernels; in 2D they can only no-op, so don't let them burn budget.
        if self.name.startswith("tmpl_") and not getattr(ctx, "_block_is_1d", True):
            return False
        if self.inputs:
            if not ctx._live_pool_vars(for_write=False):
                return False
        return True

    def run(self, ctx) -> None:
        self.fn(ctx)


REGISTRY: Dict[str, WeaveBlock] = {}


def weave_block(name: str, *, inputs: tuple = (), requires_uniform: bool = False,
                weight: float = 1.0, cost: int = 1,
                backends: tuple = ("cuda", "hip"), once: bool = False):
    def deco(fn):
        REGISTRY[name] = WeaveBlock(
            name=name, fn=fn, inputs=inputs,
            requires_uniform=requires_uniform, weight=weight, cost=cost,
            backends=backends, once=once)
        return REGISTRY[name]
    return deco


def runnable(ctx) -> List[WeaveBlock]:
    return [b for b in REGISTRY.values() if b.can_execute(ctx)]


def pick_block(ctx) -> Optional[WeaveBlock]:
    """Swarm-weighted selection among runnable blocks."""
    pool = runnable(ctx)
    if not pool:
        return None
    names = [b.name for b in pool]
    rand_w = ctx.weave_schedule.fresh_weights(names)
    weights = [max(1e-6, b.weight * rand_w.get(b.name, 1.0)) for b in pool]
    return ctx.rng.choices(pool, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Built-in blocks
# ---------------------------------------------------------------------------

def _u32() -> GType:
    return T.u32


@weave_block("int_binop", inputs=(T.u32,))
def _blk_int_binop(ctx) -> None:
    """a = b <op> c; folded into the checksum."""
    t = T.u32
    lhs = ctx.kx_random_var(t, loose=True)
    rhs = ctx.kx_random_var(t, loose=True)
    op = ctx.rnd_choice([Arith.ADD, Arith.SUB, Arith.MUL,
                          Arith.BAND, Arith.BOR, Arith.BXOR])
    out = ctx.fresh(t, init=BinOp(t, op=op, lhs=lhs, rhs=rhs))
    ctx.observe(out)


@weave_block("safe_divmod", inputs=(T.u32,))
def _blk_safe_divmod(ctx) -> None:
    t = T.u32
    op = ctx.rnd_choice([SafeOp.SDIV, SafeOp.SMOD])
    lhs = ctx.kx_random_var(t, loose=True)
    rhs = ctx.kx_random_var(t, loose=True)
    out = ctx.fresh(t, init=SafeBin(t, op=op, lhs=lhs, rhs=rhs))
    ctx.observe(out)


@weave_block("shift_mix", inputs=(T.u32,))
def _blk_shift_mix(ctx) -> None:
    t = T.u32
    x = ctx.kx_random_var(t, loose=True)
    s = ctx.rnd_int(1, 31)
    rot = BinOp(t, op=Arith.BOR,
                lhs=SafeBin(t, op=SafeOp.SSHL, lhs=x, rhs=Lit(t, value=s)),
                rhs=SafeBin(t, op=SafeOp.SSHR, lhs=x, rhs=Lit(t, value=32 - s)))
    out = ctx.fresh(t, init=rot)
    ctx.observe(out)


@weave_block("u64_muldiv", inputs=(T.u32,))
def _blk_u64(ctx) -> None:
    """Widening u64 multiply + software divide (IMAD.HI / division surface)."""
    a = Cast(T.u64, to=T.u64, arg=ctx.kx_random_var(T.u32, loose=True))
    b = Cast(T.u64, to=T.u64, arg=ctx.kx_random_var(T.u32, loose=True))
    wide = BinOp(T.u64, op=Arith.MUL, lhs=a, rhs=b)
    div = SafeBin(T.u64, op=SafeOp.SMOD, lhs=wide,
                  rhs=BinOp(T.u64, op=Arith.BOR,
                            lhs=Cast(T.u64, to=T.u64,
                                     arg=ctx.kx_random_var(T.u32, loose=True)),
                            rhs=Lit(T.u64, value=1)))
    out = ctx.fresh(T.u64, init=div)
    ctx.observe(out)


@weave_block("intrinsic_mix", inputs=(T.u32,))
def _blk_intrinsic(ctx) -> None:
    t = T.u32
    fn = ctx.rnd_choice(["__popc", "__clz", "__brev", "__ffs"])
    arg = ctx.kx_random_var(t, loose=True)
    call_ty = T.i32 if fn in ("__ffs", "__clz", "__popc") else T.u32
    call = Call(call_ty, fn=fn, args=[arg], result_ty=call_ty)
    mixed = BinOp(T.u32, op=Arith.BXOR,
                  lhs=ctx.kx_random_var(t, loose=True),
                  rhs=Cast(T.u32, to=T.u32, arg=call))
    out = ctx.fresh(t, init=mixed)
    ctx.observe(out)


@weave_block("bounded_loop", inputs=(T.u32,), cost=2)
def _blk_bounded_loop(ctx) -> None:
    """Small constant-trip loop with a runtime-looking bound."""
    t = T.u32
    acc = ctx.kx_random_var(t, for_write=True)
    trips = ctx.rnd_int(2, 8)
    jv = ctx.kx_fresh_name("i")
    body = Assign(Var(acc.ty, name=acc.name),
                  BinOp(acc.ty, op=Arith.BXOR, lhs=Var(acc.ty, name=acc.name),
                        rhs=Cast(acc.ty, to=acc.ty,
                                 arg=BinOp(T.u32, op=Arith.MUL,
                                           lhs=Var(T.i32, name=jv),
                                           rhs=ctx.kx_random_var(T.u32, loose=True)))))
    ctx.emit(For(var=jv, start=Lit(T.i32, value=0), end=Lit(T.i32, value=trips),
                 step=Lit(T.i32, value=1), body=Block([body])))
    ctx.observe(acc)


@weave_block("guarded_assign", inputs=(T.u32,))
def _blk_guarded(ctx) -> None:
    """if (uniform-looking guard) x ^= y  -- predication surface."""
    t = T.u32
    cond = RelOp(T.boolean, op=ctx.rnd_choice(list(Cmp)),
                 lhs=ctx.kx_random_var(t, loose=True),
                 rhs=ctx.kx_random_var(t, loose=True))
    target = ctx.kx_random_var(t, for_write=True)
    assign = Assign(Var(target.ty, name=target.name),
                    BinOp(target.ty, op=Arith.BXOR,
                          lhs=Var(target.ty, name=target.name),
                          rhs=Cast(target.ty, to=target.ty,
                                   arg=ctx.kx_random_var(t, loose=True))))
    ctx.emit(If(cond=cond, then=Block([assign])))
    ctx.observe(target)


@weave_block("shared_exchange", requires_uniform=True, cost=2)
def _blk_shared_exchange(ctx) -> None:
    """Shared-memory lane exchange: smem[lane] = v; sync; v ^= smem[lane^1]."""
    t = T.u32
    # Sized for the largest block the launch picker can draw (vary_launch
    # 1D caps at 512); indexing lane directly would read OOB past the array
    # otherwise.  Lanes beyond blockDim.x are simply never touched.
    size = 512
    name = ctx.kx_fresh_name("wsmem")
    arr_ty = TArray(t, size)
    arr = Var(arr_ty, name=name, space=MemSpace.SHARED)
    lane = ThreadBuiltin(T.u32, axis="x", source="threadIdx")
    ctx.emit(Declare(name=name, gtype=arr_ty, space=MemSpace.SHARED,
                     quals=[Qual.SHARED_MEM]))
    src = ctx.kx_random_var(t, loose=True)
    size_c = Lit(t, value=size)
    ctx.emit(Assign(Subscript(t, array=arr, index=lane), src))
    ctx.emit(Barrier())
    nbr = SafeBin(t, op=SafeOp.SMOD,
                  lhs=BinOp(t, op=Arith.BXOR, lhs=lane, rhs=Lit(t, value=1)),
                  rhs=size_c)
    out = ctx.fresh(t, init=BinOp(t, op=Arith.BXOR, lhs=src,
                                  rhs=Subscript(t, array=arr, index=nbr)))
    ctx.observe(out)
    ctx.emit(Barrier())


@weave_block("fp_island", cost=2)
def _blk_fp_island(ctx) -> None:
    """Self-contained FP chain, folded into the off-checksum accumulator."""
    f = T.f32
    a = ctx.kx_random_var(f, loose=True)
    b = ctx.kx_random_var(f, loose=True)
    fn = ctx.rnd_choice(["sinf", "cosf", "expf", "sqrtf"])
    chain = BinOp(f, op=Arith.MUL, lhs=a, rhs=b)
    chain = BinOp(f, op=Arith.ADD, lhs=chain,
                  rhs=Call(f, fn=fn, args=[chain], result_ty=f))
    out = ctx.fresh(f, init=chain)
    ctx.kx_observe_fp(out)


# ---------------------------------------------------------------------------
# Feature-coverage generators (disabled by default).
#
# These blocks exercise CUDA codegen surface (constant memory, WMMA, dp4a,
# inline PTX, vectorized loads, shared atomics, cg reduce/scan, structs,
# sync/warp) but fold deterministic, heavy compute into the timed body, so
# they dilute the perf-divergence signal.  Disabled (weight=0) unless
# --enable-feature-gens is passed; env GPUPERF_FEATURE_GEN_WEIGHT overrides.
# Mirrors perffuzzer generator/codegen/__init__.py.
# ---------------------------------------------------------------------------
import os as _os

_FEATURE_COVERAGE_GENS = (
    "block_predicate_barrier", "warp_vote_any_all", "warp_shuffle_scan",
    "intrinsic_dp4a", "inline_ptx_int", "mem_vectorized_load",
    "shared_atomic_reduce", "mem_constant_load", "tensor_wmma_int8",
    "coop_groups_reduce_scan", "struct_aggregate", "block_device_fence",
)


def apply_feature_gen_weight(enabled: bool = False) -> None:
    """Set feature-coverage block weights.  Called once at import time
    (weight=0) and again from the CLI when --enable-feature-gens is parsed."""
    env = _os.environ.get("GPUPERF_FEATURE_GEN_WEIGHT")
    if env is None:
        env = _os.environ.get("PERFFUZZER_FEATURE_GEN_WEIGHT")
    if env is not None:
        try:
            w = float(env)
        except ValueError:
            w = 0.0
    else:
        w = 1.0 if enabled else 0.0
    for name in _FEATURE_COVERAGE_GENS:
        if name in REGISTRY:
            REGISTRY[name].weight = w


# Feature-coverage blocks are registered with weight 0 unless enabled.
