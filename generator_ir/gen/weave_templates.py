"""Whole-kernel weave templates (ProgramTemplate) + registry.

A ``WeaveTemplate`` owns the structural compute phase of a weaving-mode
kernel: the weaver drives a 3-phase construction (pre-template flat stmt
generation -> ``template.render_body(ctx)`` -> post-template generation), so a
template only emits its compute into the current open block via
``ctx.emit``/``ctx.weave``.  Mirrors perffuzzer's
``generator/codegen/program_template.py``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from generator_ir.ir.typedecl import T, TArray, TPointer, Qual, MemSpace
from generator_ir.ir.operators import Arith, Cmp, SafeOp
from generator_ir.ir.nodes import (
    Expr, Lit, Var, ThreadBuiltin, BinOp, RelOp, Cast, SafeBin, Subscript,
    WarpShuffle,
)
from generator_ir.ir.statements import Assign, Declare, Block, If, Barrier
from generator_ir.gen.context import GenContext


class WeaveTemplate:
    """Base for whole-kernel weave templates."""

    name: str = "weave_template"

    def render_body(self, ctx: GenContext) -> None:
        """Emit template-specific compute into the current open block.

        Called by generate_weave_kernel after the prologue (params, tid,
        bootstrap) and the phase-1 statement batch.  Variables created here
        are visible to the post-template phase.
        """
        raise NotImplementedError


TEMPLATE_REGISTRY: Dict[str, WeaveTemplate] = {}


def weave_template(cls):
    """Class decorator: register a WeaveTemplate subclass instance."""
    inst = cls()
    TEMPLATE_REGISTRY[inst.name] = inst
    return cls


def run_required_template_fragment(ctx: GenContext) -> int:
    """Ensure the flat weave body contains a same-kx_scope classic-template
    fragment.

    The weaving loop may select ``tmpl_*`` fragments by chance.  If none has
    fired, force one now (honouring ``ctx._preferred_template_fragment`` when
    the caller asked for a specific classic template).  Returns the fragment
    cost so the caller can subtract it from the remaining budget.
    """
    from generator_ir.gen.weave_handlers import REGISTRY

    if any(name.startswith("tmpl_") and count > 0
           for name, count in getattr(ctx, "fired_blocks", {}).items()):
        return 0

    candidates = [
        blk for name, blk in REGISTRY.items()
        if name.startswith("tmpl_") and blk.can_execute(ctx)
    ]
    if not candidates:
        return 0

    preferred = getattr(ctx, "_preferred_template_fragment", None)
    chosen = None
    if preferred:
        preferred_names = [
            preferred if preferred.startswith("tmpl_") else f"tmpl_{preferred}",
        ]
        for blk in candidates:
            if blk.name in preferred_names:
                chosen = blk
                break

    if chosen is None:
        names = [b.name for b in candidates]
        rand_w = ctx.weave_schedule.fresh_weights(names)
        weights = [max(1e-6, rand_w.get(b.name, 1.0)) for b in candidates]
        chosen = ctx.rng.choices(candidates, weights=weights, k=1)[0]

    ctx.fired_blocks[chosen.name] += 1
    chosen.run(ctx)
    ctx.features_used.add("forced_template_fragment:" + chosen.name)
    return max(1, int(getattr(chosen, "cost", 1)))


@weave_template
class PureComposableTemplate(WeaveTemplate):
    """Flat fusion: ordinary blocks and template fragments share one kx_scope."""

    name = "pure_composable"

    def render_body(self, ctx: GenContext) -> None:
        """Budget-split weaving around one forced template fragment."""
        budget = ctx.kx_budget_for_complexity()
        pre_budget = max(4, budget // 3)
        ctx.weave(pre_budget)
        forced_cost = run_required_template_fragment(ctx)
        ctx.weave(max(0, budget - pre_budget - forced_cost))


# ===========================================================================
# Whole-kernel ProgramTemplates ported from perffuzzer/generator/codegen/pt/.
#
# Only the weave-mode ``render_body`` path is ported: the weaver drives the
# prologue (params, tid, bootstrap), the 3-phase construction, and the
# epilogue (guarded observe-write + launch kx_pick), so each template emits just
# its structural compute into the current open block.
# ===========================================================================

_U32_MASK = 0xFFFFFFFF


def _kx_mix_const(i: int) -> int:
    """A distinct, odd, non-trivial per-slot constant (blocks constant-folding)."""
    base = (0x9E3779B1 + i * 0x85EBCA6B) & _U32_MASK
    return (base ^ (i * 0xC2B2AE35 + 0x6B9E5A3)) & _U32_MASK | 1


def _kx_rotl7(c: Expr) -> BinOp:
    """(c << 7) | (c >> 25) -- a pure-unsigned rotate (defined for any 32-bit c)."""
    return BinOp(T.u32, op=Arith.BOR,
                 lhs=BinOp(T.u32, op=Arith.SHL, lhs=c, rhs=Lit(T.u32, value=7)),
                 rhs=BinOp(T.u32, op=Arith.SHR, lhs=c, rhs=Lit(T.u32, value=25)))


def _kx_seed_expr(ctx: GenContext, tid_var: Var, input_var: Var,
               n_uint: Expr, i: int) -> BinOp:
    """Per-thread, per-slot deterministic seed.

    Combines a data-dependent bounded load ``input[SAFE_MOD(tid+i, n)]`` (blocks
    DCE / constant folding -- the value depends on runtime input) with a tid- and
    index-mixed constant.  SAFE_MOD guarantees the index is in [0, n)."""
    idx = SafeBin(T.u32, op=SafeOp.SMOD,
                  lhs=BinOp(T.u32, op=Arith.ADD,
                            lhs=tid_var, rhs=Lit(T.u32, value=i)),
                  rhs=n_uint)
    load = Subscript(T.u32, array=input_var, index=idx)
    mixed = BinOp(T.u32, op=Arith.MUL,
                  lhs=tid_var, rhs=Lit(T.u32, value=_kx_mix_const(i)))
    return BinOp(T.u32, op=Arith.BXOR,
                 lhs=load,
                 rhs=BinOp(T.u32, op=Arith.ADD, lhs=mixed,
                           rhs=Lit(T.u32, value=_kx_mix_const(i + 17))))


# Complexity -> (number of live vars, cross-mix rounds, arithmetic chain depth).
# Bounded so even "stress" stays fast under the acceptance run-timeout.
_STRESS_PROFILE = {
    "low":    (8,  2, 4),
    "medium": (16, 2, 6),
    "high":   (28, 3, 8),
    "stress": (40, 3, 10),
}


def _stress_profile(ctx: GenContext):
    return _STRESS_PROFILE.get(getattr(ctx.config, "complexity", "medium"),
                               _STRESS_PROFILE["medium"])


# ---------------------------------------------------------------------------
# compute_stress -- ALU throughput emphasis (pt/compute_stress.py)
# ---------------------------------------------------------------------------

@weave_template
class ComputeStressTemplate(WeaveTemplate):
    """Whole kernel: a wide set of live vars driven through deep unsigned
    arithmetic chains, plus a weaving pass that piles more ALU work on top.

    Every produced value is folded into ``_fv``.  No control flow around the
    compute, so the body stays uniform for any sync/warp block the weaving
    pass picks.
    """

    name = "compute_stress"

    def render_body(self, ctx: GenContext) -> None:
        """Emit compute-stress body: wide live set + deep arithmetic chains + weaving."""
        n_vars, _rounds, chain = _stress_profile(ctx)
        n_uint = Cast(T.u32, to=T.u32, arg=Var(T.i32, name="n"))
        tid_var = Var(T.u32, name="tid")
        input_var = Var(TPointer(T.u32), name="input")

        # --- Wide live set: N data-dependent seeds (block DCE / fold). --------
        regs: List[Var] = []
        for i in range(n_vars):
            var = ctx.fresh(
                T.u32, init=_kx_seed_expr(ctx, tid_var, input_var, n_uint, i),
                divergent=True)
            regs.append(var)

        # --- Deep arithmetic chains: each var absorbs a long dependent chain of
        #     mixed unsigned ops (add / mul / xor / rotate) over its neighbours,
        #     keeping the whole set live and ALU-bound.  Fully unrolled. -------
        for i in range(n_vars):
            acc: Expr = regs[i]
            for d in range(chain):
                b = regs[(i + d + 1) % n_vars]
                c = regs[(i + 2 * d + 3) % n_vars]
                if d % 3 == 0:
                    step = BinOp(T.u32, op=Arith.MUL,
                                 lhs=BinOp(T.u32, op=Arith.ADD,
                                           lhs=acc, rhs=b),
                                 rhs=Lit(T.u32, value=_kx_mix_const(d) | 1))
                elif d % 3 == 1:
                    step = BinOp(T.u32, op=Arith.BXOR,
                                 lhs=BinOp(T.u32, op=Arith.ADD,
                                           lhs=acc, rhs=_kx_rotl7(c)),
                                 rhs=b)
                else:
                    step = BinOp(T.u32, op=Arith.ADD,
                                 lhs=BinOp(T.u32, op=Arith.BXOR,
                                           lhs=acc, rhs=c),
                                 rhs=_kx_rotl7(b))
                acc = step
            ctx.emit(Assign(Var(T.u32, name=regs[i].name), acc))

        # --- Fold the whole live set so nothing is dead-code-eliminated. ------
        for var in regs:
            ctx.observe(var)

        # --- Weaving pass: pile additional generated ALU work onto the live
        #     pool.  Budget scaled UP for stress. ------------------------------
        ctx.weave(ctx.kx_budget_for_complexity() + n_vars)


# ---------------------------------------------------------------------------
# register_pressure_stress -- live-register / spill emphasis
# ---------------------------------------------------------------------------

@weave_template
class RegisterPressureStressTemplate(WeaveTemplate):
    """Whole kernel reproducing the register-pressure / spill finder.

    N simultaneously-live unsigned scalars are cross-mixed in a dependency web
    for several rounds so the allocator must keep them all resident (or spill
    to local memory).  Honors ``allow_volatile``: the locals are ``volatile``
    ONLY when the flag is set; never ungated volatile.

    All N vars are folded into ``_fv``.  Body stays uniform.
    """

    name = "register_pressure_stress"

    def render_body(self, ctx: GenContext) -> None:
        """Emit register-pressure body: N live scalars cross-mixed in a dependency web."""
        n_vars, rounds, _chain = _stress_profile(ctx)
        n_uint = Cast(T.u32, to=T.u32, arg=Var(T.i32, name="n"))
        tid_var = Var(T.u32, name="tid")
        input_var = Var(TPointer(T.u32), name="input")
        allow_vol = bool(getattr(ctx.config, "allow_volatile", False))

        # --- Declare N live scalars (data-dependent seeds).  We manage these
        #     decls ourselves (not ctx.fresh) so we can attach the volatile
        #     qualifier when allow_volatile is True. ---------------------------
        regs: List[Var] = []
        for i in range(n_vars):
            name = ctx.kx_fresh_name("rps")
            var = Var(T.u32, name=name)
            ctx.kx_declare(var, divergent=True)
            ctx.emit(Declare(
                name=name, gtype=T.u32,
                init=_kx_seed_expr(ctx, tid_var, input_var, n_uint, i),
                quals=[Qual.VOLATILE] if allow_vol else []))
            regs.append(var)

        # --- Dependency web: each var updated from two neighbours, repeated for
        #     several rounds so every var stays live across the whole region. --
        for _ in range(rounds):
            for i in range(n_vars):
                a = regs[i]
                b = regs[(i + 1) % n_vars]
                c = regs[(i + 2) % n_vars]
                mix = BinOp(T.u32, op=Arith.BXOR,
                            lhs=BinOp(T.u32, op=Arith.ADD,
                                      lhs=a, rhs=b),
                            rhs=_kx_rotl7(c))
                ctx.emit(Assign(Var(T.u32, name=a.name), mix))

        # --- Fold ALL of them (none can be DCE'd -> all stay live). ----------
        for var in regs:
            ctx.observe(var)


# ---------------------------------------------------------------------------
# fp_stress -- FP-dominant whole kernel (pt/fp_stress.py)
# ---------------------------------------------------------------------------

# The FP-island blocks to concentrate on.  We fire whichever of these are
# runnable in the current context/backend (``can_execute`` filters the CUDA-only
# blocks out on the hip backend automatically).  The branchy/library families
# are listed first so they dominate the forced weave.
_FP_GEN_NAMES = [
    # branchy / library transcendentals + categorize (the missed surface) --
    # listed first so they dominate the forced weave.
    "fp_lib_transcendental_f", "fp_lib_transcendental_d",
    "fp_lib_cuda_special", "fp_double_classify", "fp_double_div_loop",
    "fp_branch_twoarm", "fp_branch_multiregion", "fp_branch_libcall_guard",
    # the original straight-line island generators, for breadth --
    "fp_transcendental", "fp_binary", "fp_fma", "fp_divmix",
    "fp_to_int_roundtrip", "fp_rounding_mode", "fp_bridge", "fp_half2",
]

# Complexity -> (integer seeds, target FP-cost budget to force-weave).  Bounded
# so even "stress" stays within the acceptance run-timeout (the _rep loop adds
# the runtime kx_multiplier, so the body itself stays modest).
_FP_PROFILE = {
    "low":    (3, 20),
    "medium": (4, 36),
    "high":   (6, 52),
    "stress": (8, 72),
}


def _fp_profile(ctx: GenContext):
    return _FP_PROFILE.get(getattr(ctx.config, "complexity", "medium"),
                           _FP_PROFILE["medium"])


def _load_fp_weave_blocks() -> None:
    """Best-effort import of the FP weave-block modules so the FP-island
    blocks are registered before fp_stress force-fires them (mirrors
    perffuzzer's package-level gens_fp / gens_fp_lib / gens_fp_branch
    registration).  Modules that do not exist yet are skipped."""
    import importlib
    import importlib.util
    for mod in ("fp", "fp_lib", "fp_branch"):
        full = f"generator_ir.gen.weave_blocks.{mod}"
        try:
            if importlib.util.find_spec(full) is not None:
                importlib.import_module(full)
        except ImportError:
            pass


@weave_template
class FpStressTemplate(WeaveTemplate):
    """Whole kernel dominated by floating-point library / branch / categorize work
    folded off-checksum, with a small integer checksum spine.  Pairs with the
    production ``_rep`` wrapper to make FP perf regressions timing-visible."""

    name = "fp_stress"

    def render_body(self, ctx: GenContext) -> None:
        """Emit FP-stress body: integer spine + force-fired FP island blocks."""
        n_seeds, fp_budget = _fp_profile(ctx)

        tid_var = Var(T.u32, name="tid")
        input_var = Var(TPointer(T.u32), name="input")
        n_uint = Cast(T.u32, to=T.u32, arg=Var(T.i32, name="n"))

        # --- Integer spine: a few data-dependent seeds folded into _fv.  These
        #     keep the integer checksum non-trivial (so old==new is a real test)
        #     AND populate the pool with live unsigned values the fp_branch_*
        #     blocks read for their (bit-identical) branch conditions. ---------
        for i in range(n_seeds):
            idx = SafeBin(T.u32, op=SafeOp.SMOD,
                          lhs=BinOp(T.u32, op=Arith.ADD,
                                    lhs=tid_var, rhs=Lit(T.u32, value=i)),
                          rhs=n_uint)
            load = Subscript(T.u32, array=input_var, index=idx)
            mixed = BinOp(T.u32, op=Arith.BXOR, lhs=load,
                          rhs=BinOp(T.u32, op=Arith.MUL,
                                    lhs=tid_var,
                                    rhs=Lit(T.u32, value=0x9E3779B1)))
            ctx.observe(ctx.fresh(T.u32, init=mixed, divergent=True))

        # --- FP-dominant body: force-fire the FP-island blocks until the
        #     target FP cost budget is spent.  Each runnable block folds its
        #     result via kx_observe_fp, so used_fp() becomes True and the weaver's
        #     epilogue adds the output_fp param/write.  We respect can_execute so
        #     the CUDA-only family is skipped on the hip backend. --------------
        _load_fp_weave_blocks()
        from generator_ir.gen.weave_handlers import REGISTRY
        runnable = [REGISTRY[name] for name in _FP_GEN_NAMES
                    if name in REGISTRY and REGISTRY[name].can_execute(ctx)]
        spent = 0
        guard = 0
        # Cap the number of forced firings so a pathological cost can't loop
        # forever; the budget is the real terminator.
        while spent < fp_budget and runnable and guard < 256:
            guard += 1
            blk = ctx.rnd_choice(runnable)
            blk.run(ctx)
            ctx.fired_blocks[blk.name] += 1
            spent += max(1, int(getattr(blk, "cost", 1)))

        # --- A short ordinary weave for breadth (more FP + a little integer ALU
        #     to interleave), folded into _fv / output_fp by the blocks.  Kept
        #     to a third of the normal budget so the FORCED FP work above stays
        #     the dominant fraction of the body (this template's purpose). -----
        ctx.weave(max(2, ctx.kx_budget_for_complexity() // 3))


# ---------------------------------------------------------------------------
# memory_2d -- 2D memory-access whole kernel (pt/memory_2d.py)
# ---------------------------------------------------------------------------

def _safe_load_2d(input_var: Var, raw_idx: Expr, total_uint: Expr) -> Subscript:
    """input[ SAFE_MOD(raw_idx, n*n) ] -- always in-bounds (in_size == n*n)."""
    bounded = SafeBin(T.u32, op=SafeOp.SMOD, lhs=raw_idx, rhs=total_uint)
    return Subscript(T.u32, array=input_var, index=bounded)


def _kx_weave_2d_body(ctx: GenContext, input_var: Var, n_uint: Expr,
                   x_var: Var, y_var: Var, idx_var: Var) -> Var:
    """Emit a handful of 2D gather/transpose/stencil-style reads of *input*,
    folding each loaded value into the running checksum via ctx.observe().

    Coverage is data-dependent on (x, y) so every thread's contribution differs,
    yet stays deterministic + bounded.  ``total`` = n*n bounds every read.
    """
    total = BinOp(T.u32, op=Arith.MUL, lhs=n_uint, rhs=n_uint)
    total_var = ctx.fresh(T.u32, init=total)   # cache n*n once

    # 1) Row-major gather at the thread's own cell: input[idx].
    ctx.observe(ctx.fresh(T.u32,
                          init=_safe_load_2d(input_var, idx_var, total_var)))

    # 2) Transpose-style read: input[x*n + y]  (the mirrored coordinate).
    transp = BinOp(T.u32, op=Arith.ADD,
                   lhs=BinOp(T.u32, op=Arith.MUL, lhs=x_var, rhs=n_uint),
                   rhs=y_var)
    ctx.observe(ctx.fresh(T.u32,
                          init=_safe_load_2d(input_var, transp, total_var)))

    # 3) A small stencil: sum a randomized set of neighbour offsets around idx.
    #    Each offset read is SAFE_MOD-bounded, so a wrap at a matrix edge is
    #    defined and deterministic (it folds a different-but-fixed neighbour).
    offsets = ctx.rnd_choice([
        [1, -1, 0],                 # horizontal 3-point
        [0],                        # 1-point (degenerate)
        [1, -1],                    # x neighbours
    ])
    # Vertical neighbours expressed as +-n; horizontal as +-1.  We model them as
    # additive offsets on idx (unsigned wrap is fine -- SAFE_MOD re-bounds it).
    deltas: List[Expr] = []
    for o in offsets:
        deltas.append(Lit(T.u32, value=(o & 0xFFFFFFFF)))   # +-1, 0
    # add one vertical neighbour (row +/- 1 == idx +/- n) when stencil > 1 pt.
    if len(offsets) > 1 and ctx.rnd_bool(0.6):
        deltas.append(n_uint)   # idx + n  (next row, same column)

    acc: Optional[Expr] = None
    for d in deltas:
        raw = BinOp(T.u32, op=Arith.ADD, lhs=idx_var, rhs=d)
        val = _safe_load_2d(input_var, raw, total_var)
        acc = val if acc is None else BinOp(T.u32, op=Arith.ADD,
                                            lhs=acc, rhs=val)
    if acc is not None:
        ctx.observe(ctx.fresh(T.u32, init=acc))

    return total_var


@weave_template
class Memory2DTemplate(WeaveTemplate):
    """Whole-kernel 2D template: 2D coords -> bounded input gathers + weaving
    coupling.  Sets ``ctx.kx_is_2d`` so the weaver's epilogue + launch picker use
    the 2D contract (n*n buffers, 2D grid)."""

    name = "memory_2d"

    def render_body(self, ctx: GenContext) -> None:
        """Emit 2D memory body: kx_declare x/y/idx coords, then weave."""
        ctx.kx_is_2d = True
        # 2D block geometry: threadIdx.x is NOT the unique in-block lane, so
        # block-collective (requires_uniform: sync/shared/warp) blocks would
        # race on shared memory.  Clearing this keeps the weaving pass to
        # geometry-agnostic blocks only (no FALSE_MISMATCH from races).
        ctx._block_is_1d = False

        # The 3-phase caller only sets up 1D tid.  Emit x/y/idx declarations
        # so the 2D coordinate system is available to the weaving body.
        x_expr = BinOp(T.u32, op=Arith.ADD,
            lhs=BinOp(T.u32, op=Arith.MUL,
                      lhs=ThreadBuiltin(T.u32, axis="x", source="blockIdx"),
                      rhs=ThreadBuiltin(T.u32, axis="x", source="blockDim")),
            rhs=ThreadBuiltin(T.u32, axis="x", source="threadIdx"))
        y_expr = BinOp(T.u32, op=Arith.ADD,
            lhs=BinOp(T.u32, op=Arith.MUL,
                      lhs=ThreadBuiltin(T.u32, axis="y", source="blockIdx"),
                      rhs=ThreadBuiltin(T.u32, axis="y", source="blockDim")),
            rhs=ThreadBuiltin(T.u32, axis="y", source="threadIdx"))

        x_var = Var(T.u32, name="x")
        y_var = Var(T.u32, name="y")
        ctx.kx_declare(x_var, divergent=True)
        ctx.kx_declare(y_var, divergent=True)
        ctx.emit(Declare(name="x", gtype=T.u32, init=x_expr))
        ctx.emit(Declare(name="y", gtype=T.u32, init=y_expr))

        n_uint = Cast(T.u32, to=T.u32, arg=Var(T.i32, name="n"))
        idx_expr = BinOp(T.u32, op=Arith.ADD,
                         lhs=BinOp(T.u32, op=Arith.MUL, lhs=y_var, rhs=n_uint),
                         rhs=x_var)
        idx_var = Var(T.u32, name="idx")
        ctx.kx_declare(idx_var, divergent=True)
        ctx.emit(Declare(name="idx", gtype=T.u32, init=idx_expr))

        # Reassign tid to the 2D linear index so Phase 1/3 stmts that
        # reference tid get unique per-thread values across both x and y.
        ctx.emit(Assign(Var(T.u32, name="tid"), idx_var))

        input_var = Var(TPointer(T.u32), name="input")
        self._kx_weave_memory_body(ctx, input_var, n_uint, x_var, y_var, idx_var)

        # A SMALL coupling budget so the weaving blocks (arith/intrinsic) mix
        # the loaded values further -- still integer-exact + UB-free.
        budget = max(4, ctx.kx_budget_for_complexity() // 2)
        ctx.weave(budget)

    # Hook so subclasses (e.g. a stencil-heavy variant) can swap the body.
    def _kx_weave_memory_body(self, ctx: GenContext, input_var: Var,
                           n_uint: Expr, x_var: Var, y_var: Var,
                           idx_var: Var) -> Var:
        return _kx_weave_2d_body(ctx, input_var, n_uint, x_var, y_var, idx_var)


# ---------------------------------------------------------------------------
# reduction family -- shared-memory block reduction + inclusive scan
# (pt/reduction_family.py)
# ---------------------------------------------------------------------------

# Associative, commutative, overflow-safe combine ops (unsigned -> all defined).
_COMBINE_OPS = [Arith.ADD, Arith.BXOR, Arith.BOR, Arith.BAND]

_BLOCK_SIZE = 256   # the weaver pins the launch block to 256 for these


@weave_template
class ReductionBlockTemplate(WeaveTemplate):
    """Whole-kernel block reduction.

    Structure (uniform / race-free):
        sdata[threadIdx.x] = input[tid % n] ^ <coupling expr>;   // unconditional
        __syncthreads();
        for s = bs/2 .. 32 (unrolled):
            if (threadIdx.x < s) sdata[threadIdx.x] OP= sdata[threadIdx.x + s];
            __syncthreads();
        warp_val = sdata[threadIdx.x];
        for delta in 16,8,4,2,1: warp_val OP= __shfl_down_sync(-1, warp_val, delta);
    then observe(warp_val).  A small ``ctx.weave`` budget runs first so pool
    blocks couple into the coupling expr / output.
    """

    name = "reduction_block"

    def render_body(self, ctx: GenContext) -> None:
        """Emit reduction body: shared-memory tree + warp shuffle, folded via observe."""
        bs = _BLOCK_SIZE
        op = ctx.rnd_choice(_COMBINE_OPS)

        # Light pool weaving for coupling/diversity (kept small so the
        # reduction structure dominates).  Stays uniform.
        ctx.weave(max(4, ctx.kx_budget_for_complexity() // 3))

        local_tid = ThreadBuiltin(T.u32, axis="x", source="threadIdx")
        tid_var = Var(T.u32, name="tid")
        n_uint = Cast(T.u32, to=T.u32, arg=Var(T.i32, name="n"))
        input_var = Var(TPointer(T.u32), name="input")

        # __shared__ unsigned int sdata[bs];
        sdata_name = ctx.kx_fresh_name("sdata")
        sdata = Var(TArray(T.u32, bs), name=sdata_name, space=MemSpace.SHARED)
        ctx.kx_declare(sdata)
        ctx.emit(Declare(name=sdata_name, gtype=TArray(T.u32, bs),
                         space=MemSpace.SHARED, quals=[Qual.SHARED_MEM]))

        # Unconditional, in-bounds load:  sdata[threadIdx.x] = input[tid%n] ^ expr
        load_idx = SafeBin(T.u32, op=SafeOp.SMOD, lhs=tid_var, rhs=n_uint)
        coupling = ctx.build_observed_int(T.u32, depth=2)   # integer-exact
        load_expr = BinOp(T.u32, op=Arith.BXOR,
                          lhs=Subscript(T.u32, array=input_var, index=load_idx),
                          rhs=coupling)
        ctx.emit(Assign(Subscript(T.u32, array=sdata, index=local_tid),
                        load_expr))
        ctx.emit(Barrier())

        # Block-level tree reduction (unrolled) down to one warp (32 lanes).
        s_val = bs // 2
        while s_val >= 32:
            s_const = Lit(T.u32, value=s_val)
            ctx.emit(If(
                cond=RelOp(T.boolean, op=Cmp.LT,
                           lhs=local_tid, rhs=s_const),
                then=Block([Assign(
                    Subscript(T.u32, array=sdata, index=local_tid),
                    BinOp(T.u32, op=op,
                          lhs=Subscript(T.u32, array=sdata, index=local_tid),
                          rhs=Subscript(T.u32, array=sdata,
                                        index=BinOp(T.u32, op=Arith.ADD,
                                                    lhs=local_tid,
                                                    rhs=s_const))))])))
            ctx.emit(Barrier())
            s_val //= 2

        # Warp-level reduction with __shfl_down_sync (full mask -> all 32 lanes).
        warp_val = ctx.fresh(T.u32,
                             init=Subscript(T.u32, array=sdata,
                                            index=local_tid),
                             divergent=True)
        full_mask = Lit(T.u32, value=0xFFFFFFFF)
        for delta in (16, 8, 4, 2, 1):
            sh = ctx.fresh(T.u32, init=WarpShuffle(
                T.u32, kind="shfl_down_sync", mask=full_mask,
                value=warp_val, delta=Lit(T.i32, value=delta)),
                divergent=True)
            ctx.emit(Assign(warp_val,
                            BinOp(T.u32, op=op, lhs=warp_val, rhs=sh)))

        # Fold the reduction accumulator into the running checksum.
        ctx.observe(warp_val)


@weave_template
class ScanInclusiveTemplate(WeaveTemplate):
    """Whole-kernel inclusive prefix scan (Hillis-Steele, double-buffered).

    Two shared buffers ping/pong so each step reads the previous buffer and
    writes the next -- no read-after-write race, and every barrier is uniform:

        buf0[threadIdx.x] = input[tid % n] ^ <coupling expr>;
        __syncthreads();
        for off = 1,2,4 .. < bs (unrolled), src/dst alternate:
            if (threadIdx.x >= off) dst[tid] = src[tid] OP src[tid-off];
            else                    dst[tid] = src[tid];
            __syncthreads();
        scanned = <final buffer>[threadIdx.x];

    ``dst[threadIdx.x]`` is written by EVERY thread on every step (the else-arm
    copies forward), so the next step's read of ``dst`` is always defined.
    The ``threadIdx.x - off`` read is in-bounds because it is taken only when
    ``threadIdx.x >= off``.  All indices are < bs (the static buffer length).
    """

    name = "scan_inclusive"

    def render_body(self, ctx: GenContext) -> None:
        """Emit scan body: Hillis-Steele inclusive prefix scan, folded via observe."""
        bs = _BLOCK_SIZE
        op = ctx.rnd_choice(_COMBINE_OPS)

        ctx.weave(max(4, ctx.kx_budget_for_complexity() // 3))

        local_tid = ThreadBuiltin(T.u32, axis="x", source="threadIdx")
        tid_var = Var(T.u32, name="tid")
        n_uint = Cast(T.u32, to=T.u32, arg=Var(T.i32, name="n"))
        input_var = Var(TPointer(T.u32), name="input")

        # Two ping-pong __shared__ buffers.
        def _kx_shared_buf() -> Var:
            name = ctx.kx_fresh_name("scan_buf")
            var = Var(TArray(T.u32, bs), name=name, space=MemSpace.SHARED)
            ctx.kx_declare(var)
            ctx.emit(Declare(name=name, gtype=TArray(T.u32, bs),
                             space=MemSpace.SHARED, quals=[Qual.SHARED_MEM]))
            return var

        buf_a = _kx_shared_buf()
        buf_b = _kx_shared_buf()

        # Initial load into buf_a (unconditional, bounded).
        load_idx = SafeBin(T.u32, op=SafeOp.SMOD, lhs=tid_var, rhs=n_uint)
        coupling = ctx.build_observed_int(T.u32, depth=2)
        load_expr = BinOp(T.u32, op=Arith.BXOR,
                          lhs=Subscript(T.u32, array=input_var, index=load_idx),
                          rhs=coupling)
        ctx.emit(Assign(Subscript(T.u32, array=buf_a, index=local_tid),
                        load_expr))
        ctx.emit(Barrier())

        # Hillis-Steele steps, ping-ponging src/dst.
        src, dst = buf_a, buf_b
        off = 1
        while off < bs:
            off_const = Lit(T.u32, value=off)
            self_v = Subscript(T.u32, array=src, index=local_tid)
            prev_v = Subscript(T.u32, array=src,
                               index=BinOp(T.u32, op=Arith.SUB,
                                           lhs=local_tid, rhs=off_const))
            combined = BinOp(T.u32, op=op, lhs=self_v, rhs=prev_v)
            ctx.emit(If(
                cond=RelOp(T.boolean, op=Cmp.GE,
                           lhs=local_tid, rhs=off_const),
                then=Block([Assign(
                    Subscript(T.u32, array=dst, index=local_tid), combined)]),
                otherwise=Block([Assign(
                    Subscript(T.u32, array=dst, index=local_tid),
                    Subscript(T.u32, array=src, index=local_tid))])))
            ctx.emit(Barrier())
            src, dst = dst, src
            off *= 2

        # `src` now holds the final scanned buffer (after the last swap).
        scanned = ctx.fresh(T.u32,
                            init=Subscript(T.u32, array=src, index=local_tid),
                            divergent=True)
        ctx.observe(scanned)
