"""Yield-family weave blocks: the proven differential-divergence drivers plus
a deterministic register-pressure generator (port of
perffuzzer/generator/codegen/gens_yield.py).

These wrap the patterns historically responsible for the most nvcc 11.5-vs-13.1
findings.  They are *not* statically elevated -- they register at base weight
and the per-seed swarm decides emphasis.  Every block here upholds the weaving
invariants: deterministic + UB-free by construction, folds its produced
value(s) into the running checksum via ``ctx.observe`` so old==new always holds
(the divergence lives in CODEGEN / instruction selection / scheduling, NOT in
results), and NEVER emits ungated ``volatile`` (honors ``allow_volatile``).

Thin-layer recipe: we WRAP the existing builders --
``StmtGen._divergent_switch`` / ``_strided_unroll`` (stmt_builder) and
``ExprGen._u64_divmod`` / the GPU_FAKE_DIVERGE / GPU_GROUP_DIVERGE expr macros
(expr_builder / emit.base_emitter.write_diverge_macros) -- rather than
reimplementing them, so the verified pattern shape reaches the emitted
PTX/source unchanged.  Weave mode keeps the wrapped ExprGen path
integer-exact, so no FP leaks into ``_fv``.

File name note: ``yield`` is a Python keyword, hence ``yield_.py``; only the
registered block names matter.
"""

from __future__ import annotations

from typing import List

from generator_ir.ir.typedecl import T, Qual
from generator_ir.ir.operators import Arith, Cmp
from generator_ir.ir.nodes import (
    Expr, Lit, Var, BinOp, RelOp, Cast, Call, ThreadBuiltin,
)
from generator_ir.ir.statements import (
    Stmt, Assign, Block, Declare, For, If, Switch,
)
from generator_ir.gen.weave_handlers import weave_block
from generator_ir.gen.expr_builder import ExprGen
from generator_ir.gen.stmt_builder import StmtGen


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stmt_build(ctx) -> StmtGen:
    """A StmtGen bound to *ctx* (so we can call its verified pattern builders)."""
    return StmtGen(ctx, ExprGen(ctx))


def _as_stmt_list(stmt) -> List[Stmt]:
    """Normalise a builder result: gpuperf's divergence builders return a FLAT
    list (decl + pattern [+ fold]); accept a lone Stmt too."""
    if stmt is None:
        return []
    if isinstance(stmt, list):
        return [s for s in stmt if s is not None]
    return [stmt]


# Running-checksum accumulators that must NEVER be re-observed as a sink.  The
# wrapped divergence builders append their own ``_fv ^= acc`` fold
# (``Assign(_fv, ...)``) when ``_fv`` is in kx_scope (the mutate/splice path);
# folding ``_fv`` into ITSELF via observe() would corrupt the determinism
# oracle, so these names are excluded from the recovered sinks.
_KX_CHECKSUM_ACCUM = frozenset({"_fv", "output_fp_acc"})


def _kx_assigned_targets(stmts: List[Stmt]) -> List[Var]:
    """Walk a Stmt forest and collect the lvalue Vars of every Assign.

    The wrapped stmt_builder divergence builders kx_pick their OWN sink and return
    IR that mutates it without telling us which var it chose.  To fold that
    mutation into ``_fv`` we recover the sink by scanning the returned IR for
    assignment targets that are plain Vars.  The checksum accumulators
    themselves are skipped (folding them into the checksum is never intended)."""
    found: List[Var] = []
    seen: set = set()

    def inspect(node) -> None:
        if node is None:
            return
        if isinstance(node, Assign):
            tgt = node.target
            if (isinstance(tgt, Var) and tgt.name not in seen
                    and tgt.name not in _KX_CHECKSUM_ACCUM):
                seen.add(tgt.name)
                found.append(tgt)
        if isinstance(node, Block):
            for s in node.stmts:
                inspect(s)
        elif isinstance(node, For):
            inspect(node.body)
        elif isinstance(node, If):
            inspect(node.then)
            inspect(node.otherwise)
        elif isinstance(node, Switch):
            for case in node.cases:
                inspect(case.body)

    for s in stmts:
        inspect(s)
    return found


def _kx_opaque_true(ctx) -> RelOp:
    """A runtime-but-always-true predicate the optimizer cannot fold away.

    ``n`` is the (runtime) element count, always >= 1 for any real launch, so
    ``(n | 1) > 0`` is always true but compiler-opaque (no ``volatile`` needed --
    opacity from the runtime ``n`` parameter, not from a volatile local)."""
    n_or1 = BinOp(T.i32, op=Arith.BOR,
                  lhs=Var(T.i32, name="n"), rhs=Lit(T.i32, value=1))
    return RelOp(T.boolean, op=Cmp.GT, lhs=n_or1, rhs=Lit(T.i32, value=0))


# ---------------------------------------------------------------------------
# 1. switch-divergence  (switch(v & 3) with divergent-lowering arms)
# ---------------------------------------------------------------------------

@weave_block("yield_switch_divergence", inputs=(T.u32,), cost=3)
def _blk_switch_divergence(ctx) -> None:
    """Wrap stmt_builder's verified divergent-switch pattern: arms apply
    DIFFERENT integer work (rotate / *C / shift-mix / +expr) to an unsigned
    accumulator, driving 13.1's if-conversion + reconvergence-barrier removal.
    We emit the switch, then fold the mutated sink into ``_fv`` so the value is
    observed (old==new holds; the divergence is purely in lowering)."""
    sg = _stmt_build(ctx)
    stmts = _as_stmt_list(sg._divergent_switch(depth=0))
    if not any(isinstance(s, Switch) for s in stmts):
        # builder fell back (no assignable sink) -> nothing verified to wrap.
        return
    for s in stmts:
        ctx.emit(s)
    for sink in _kx_assigned_targets(stmts):
        ctx.observe(sink)


# ---------------------------------------------------------------------------
# 2. strided-unroll reduction  (acc ^= input[(tid%n) + j*n], bare affine index)
# ---------------------------------------------------------------------------

@weave_block("yield_strided_unroll", inputs=(T.u32,), cost=3)
def _blk_strided_unroll(ctx) -> None:
    """Wrap stmt_builder's verified strided-unroll reduction over ``input``.
    The bare affine index (stride = n) lets 13.1's IMAD.WIDE address
    strength-reduction fire against 11.5's serial carry chain.  In-bounds is
    guaranteed by enlarging the INPUT buffer (``input_size_mult``) to cover the
    maximum unroll reach; the read is data-dependent (blocks DCE) and
    per-thread deterministic."""
    # The pattern reads input[(tid%n) + j*n] for j in [0, U); max index < U*n.
    # Ensure the host-side input buffer is at least U_max*n so there is no OOB
    # read.  ``input_size_mult`` is consumed by the weaver / harness (default 1).
    u_max = int(getattr(ctx.config, "strided_unroll_max", 32) or 32)
    cur = int(getattr(ctx.config, "input_size_mult", 1) or 1)
    needed = u_max + 1  # base in [0,n) plus (U-1) full strides -> < U*n; +1 slack
    if cur < needed:
        ctx.config.input_size_mult = needed

    sg = _stmt_build(ctx)
    stmts = _as_stmt_list(sg._strided_unroll(depth=0))
    if not any(isinstance(s, For) for s in stmts):
        return  # builder fell back (input/n/tid or sink missing)
    for s in stmts:
        ctx.emit(s)
    for sink in _kx_assigned_targets(stmts):
        ctx.observe(sink)


# ---------------------------------------------------------------------------
# 3. u64 div/mod driver  (software 64-bit divide; sm_75 has no HW path)
# ---------------------------------------------------------------------------

@weave_block("yield_u64_divmod", inputs=(T.u32,), cost=2)
def _blk_u64_divmod_driver(ctx) -> None:
    """Wrap ExprGen's verified ``_u64_divmod``: a bare 64-bit DIV/MOD by a
    runtime, forced-nonzero ``(b | 1)`` divisor.  sm_75 has no hardware 64-bit
    divide, so both nvccs expand a software-divide helper -- 13.1 reworks its
    instruction selection.  Unsigned operands + the (b|1) divisor make it
    UB-free and deterministic.  Folded into ``_fv``."""
    val = ExprGen(ctx)._u64_divmod(T.u64, depth=0)
    ctx.observe(ctx.fresh(T.u64, init=val))


# ---------------------------------------------------------------------------
# 4. EMI-style dead code  (net-zero contribution, obscured from the optimizer)
# ---------------------------------------------------------------------------

@weave_block("yield_emi_dead_code", inputs=(T.u32,), cost=2)
def _blk_emi_dead_code(ctx) -> None:
    """Inject computation that is provably an identity on the observed checksum
    but obscured so the optimizer must reason about it (the historical #1
    finder).

    Two obscured net-zero idioms (using the always-emitted GPU_FAKE_DIVERGE /
    GPU_GROUP_DIVERGE macros from emit.base_emitter.write_diverge_macros):
      * GPU_FAKE_DIVERGE(threadIdx.x, threadIdx.x) == 0  -> XOR-ing it into a
        value is a no-op, but looks data-/divergence-dependent to the compiler.
      * GPU_GROUP_DIVERGE(blockIdx.x, x) == x            -> ``x ^
        GPU_GROUP_DIVERGE(bid,x)`` == 0 exactly, a cancelling pair the
        optimizer must prove away.

    The folded value is EXACTLY equal to a plain pool read, so old==new is
    preserved -- the purpose is to perturb optimization, not change results."""
    t = T.u32
    a = ctx.kx_random_var(t, loose=True, recency=0.6)
    tid = ThreadBuiltin(t, axis="x", source="threadIdx")
    bid = ThreadBuiltin(t, axis="x", source="blockIdx")

    if ctx.rnd_bool(0.5):
        # a ^ GPU_FAKE_DIVERGE(tid, tid)  ==  a ^ 0  ==  a
        zero = Call(t, fn="GPU_FAKE_DIVERGE", args=[tid, tid], result_ty=t)
        val = BinOp(t, op=Arith.BXOR, lhs=a, rhs=zero)
    else:
        # a ^ (a ^ GPU_GROUP_DIVERGE(bid, a))  ==  a ^ 0  ==  a
        gd = Call(t, fn="GPU_GROUP_DIVERGE", args=[bid, a], result_ty=t)
        cancel = BinOp(t, op=Arith.BXOR, lhs=a, rhs=gd)
        val = BinOp(t, op=Arith.BXOR, lhs=a, rhs=cancel)

    ctx.observe(ctx.fresh(t, init=val))


# ---------------------------------------------------------------------------
# 5. true-guard  (runtime-but-always-true predicate wrapping real folded work)
# ---------------------------------------------------------------------------

@weave_block("yield_true_guard", inputs=(T.u32,), cost=2)
def _blk_true_guard(ctx) -> None:
    """Wrap real folded work in a branch guarded by a runtime-but-always-true,
    optimizer-opaque predicate ``(n | 1) > 0``.  The body ALWAYS runs, so its
    observe is unconditional in effect (old==new holds), but the compiler must
    carry the branch / reason about its (un)reachability -- perturbing branch
    layout and predication.  Opacity comes from the runtime ``n`` parameter,
    NOT from volatile."""
    t = T.u32
    val = ctx.build_observed_int(t, depth=0)
    sink = ctx.fresh(t, init=Lit(t, value=0))

    ctx.kx_open_block()
    ctx.emit(Assign(Var(t, name=sink.name), val))
    body = ctx.kx_close_block()
    ctx.emit(If(cond=_kx_opaque_true(ctx), then=Block(body)))

    # sink is 0 if (impossibly) the guard were false; folds the real work
    # otherwise.  Always-true -> deterministic; old==new.
    ctx.observe(sink)


# ---------------------------------------------------------------------------
# 6. deterministic register-pressure  (N cross-mixed live unsigned vars)
# ---------------------------------------------------------------------------

# Complexity -> number of simultaneously-live registers to force.
_REG_N = {"low": 6, "medium": 10, "high": 16, "stress": 24}


@weave_block("yield_register_pressure", inputs=(T.u32,), cost=4)
def _blk_register_pressure(ctx) -> None:
    """Declare N live UNSIGNED vars and cross-mix them in a dependency web so
    they ALL stay live to the end, then fold every one into ``_fv``.  This
    reproduces the D1 register-pressure finder (spill / register->local_depot)
    UB-free.

    Honors ``ctx.config.allow_volatile``: when True the locals are declared
    ``volatile`` (the dominant D1 trigger); when False they are PLAIN -- NO
    ungated volatile.  Either way the values are deterministic and folded, so
    old==new holds."""
    t = T.u32
    n = _REG_N.get(getattr(ctx.config, "complexity", "medium"), 10)
    n = max(4, n + ctx.rnd_int(-1, 2))

    allow_vol = bool(getattr(ctx.config, "allow_volatile", False))
    quals = [Qual.VOLATILE] if allow_vol else []

    # Declare N seeded vars (distinct per-thread seeds via tid mixing) and keep
    # references so we can cross-mix and fold them.
    regs: List[Var] = []
    tid = ThreadBuiltin(t, axis="x", source="threadIdx")
    for i in range(n):
        seed = BinOp(
            t, op=Arith.BXOR,
            lhs=BinOp(t, op=Arith.MUL,
                      lhs=Cast(t, to=t, arg=tid),
                      rhs=Lit(t, value=(2654435761 + i) & 0xFFFFFFFF)),
            rhs=Lit(t, value=(0x9E3779B1 ^ (i * 0x85EBCA6B)) & 0xFFFFFFFF))
        # Mix in a live pool value so the web depends on prior blocks too.
        seed = BinOp(t, op=Arith.ADD, lhs=seed,
                     rhs=ctx.kx_random_var(t, loose=True, recency=0.5))
        name = ctx.kx_fresh_name("rp")
        var = Var(t, name=name)
        ctx.kx_declare(var, divergent=True)
        ctx.emit(Declare(name=name, gtype=t, init=seed, quals=list(quals)))
        regs.append(var)

    # Dependency web: each var is updated using its two neighbours, repeated
    # for a few rounds so every var stays live across the whole region (forcing
    # the allocator to keep all N resident / spill).  Unsigned arith -> UB-free.
    rounds = 2 if n <= 10 else 3
    for _ in range(rounds):
        for i in range(n):
            a = regs[i]
            b = regs[(i + 1) % n]
            c = regs[(i + 2) % n]
            # a = (a + b) ^ rotl-ish(c)  -- pure unsigned, wraps (defined).
            rot = BinOp(t, op=Arith.BOR,
                        lhs=BinOp(t, op=Arith.SHL, lhs=c, rhs=Lit(t, value=7)),
                        rhs=BinOp(t, op=Arith.SHR, lhs=c, rhs=Lit(t, value=25)))
            mix = BinOp(t, op=Arith.BXOR,
                        lhs=BinOp(t, op=Arith.ADD, lhs=a, rhs=b),
                        rhs=rot)
            ctx.emit(Assign(Var(t, name=a.name), mix))

    # Fold ALL of them so none can be dead-code-eliminated -> all stay live.
    for var in regs:
        ctx.observe(var)
