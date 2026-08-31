"""Control-flow weaving blocks (port of perffuzzer gens_control).

Each block introduces a piece of *control flow* (if / if-else, a bounded
for-loop, a bounded while-loop, and a branchless select chain) and recursively
weaves a small codegen body INSIDE it, then folds the produced value(s) into
the running checksum ``_fv``.  They are the blocks that turn the
otherwise-flat weaving body into a tree with real divergence.

Invariants upheld:
  * **Deterministic, UB-free by construction.**  Conditions and loop bounds
    are derived from in-kx_scope per-thread integer values (or small constants)
    and are masked so a loop ALWAYS terminates with a bounded trip count --
    no infinite loops, no OOB, no div/mod-by-zero.  Every produced value is
    ``observe()``-d.
  * **Bounded nesting / trip product.**  Loop trip counts are clamped against
    ``ctx.max_loop_trip_product / ctx.loop_trip_product`` so nested loops can
    never multiply into a runaway, non-terminating kernel.
  * **Context discipline.**  ``kx_open_block``/``kx_close_block`` push/pop SCOPE +
    emission only; the *depth* counters (``if_depth`` / ``loop_depth``) and
    the safety flags (``in_nested_control_flow`` / ``inside_breakable_loop``)
    are managed HERE, inside ``try/finally`` so they are always restored --
    even on an early return.  While these are bumped, ``ctx.snapshot.uniform``
    is False, so statements woven into the body record ``uniform == False``
    and sync/warp blocks auto-gate off.
  * ``requires_uniform=False``: these CREATE non-uniformity; they don't need
    it.

All paths are integer-only and use plain pointer-free arithmetic, so the
default ("cuda", "hip") backend tuple is correct.
"""

from __future__ import annotations

from contextlib import contextmanager

from generator_ir.ir.typedecl import T
from generator_ir.ir.operators import Arith, Cmp, Compound, IncDec
from generator_ir.ir.nodes import BinOp, RelOp, Cast, Select, Var, Lit
from generator_ir.ir.statements import (
    If, For, While, Block, Assign, CompoundAssign, IncDecStmt, Declare,
)
from generator_ir.gen.weave_handlers import weave_block


# ---------------------------------------------------------------------------
# Context / depth management helpers (kx_open_block/kx_close_block do NOT touch the
# depth counters -- we must, balanced and exception-safe).
# ---------------------------------------------------------------------------

@contextmanager
def _kx_branch_scope(ctx):
    """Open a nested emission block as the body of an IF branch.

    Increments ``if_depth`` and sets ``in_nested_control_flow`` so that, while
    the body is woven, ``ctx.snapshot.uniform`` is False (statements emitted
    here record uniform=False, and sync/warp blocks stay disabled).  Restores
    all of it -- emission stack, kx_scope, depth, flag -- on exit, even on
    exception.  Yields the list of statements collected for the branch body.
    """
    ctx.if_depth += 1
    prev_nested = ctx.in_nested_control_flow
    ctx.in_nested_control_flow = True
    ctx.kx_open_block()
    stmts_box = []
    try:
        yield stmts_box
    finally:
        stmts_box.extend(ctx.kx_close_block())
        ctx.in_nested_control_flow = prev_nested
        ctx.if_depth -= 1


@contextmanager
def _kx_loop_scope(ctx, trip: int):
    """Open a nested emission block as a loop body for *trip* iterations.

    Increments ``loop_depth`` and ``loop_trip_product`` (so deeper loops
    shrink their own trip budget), sets ``inside_breakable_loop`` (sync inside
    a loop is unsafe).  ``inside_breakable_loop`` also makes
    ``ctx.snapshot.uniform`` False, so loop-body statements record
    uniform=False.  Fully restored on exit.  Yields the body statement list.
    """
    ctx.loop_depth += 1
    ctx.loop_trip_product *= max(1, trip)
    prev_breakable = ctx.inside_breakable_loop
    ctx.inside_breakable_loop = True
    ctx.kx_open_block()
    stmts_box = []
    try:
        yield stmts_box
    finally:
        stmts_box.extend(ctx.kx_close_block())
        ctx.inside_breakable_loop = prev_breakable
        ctx.loop_trip_product //= max(1, trip)
        ctx.loop_depth -= 1


def _kx_bounded_trip(ctx, choices) -> int:
    """Pick a small trip count and clamp it to the remaining per-thread trip
    budget so nested loops can never multiply into a runaway kernel."""
    trip = ctx.rnd_choice(choices)
    headroom = ctx.max_loop_trip_product // max(1, ctx.loop_trip_product)
    return max(1, min(trip, headroom))


def _kx_det_condition(ctx):
    """Build a deterministic, per-thread, UB-free boolean condition.

    Operands are existing in-kx_scope integer values (recency-biased) compared
    with a random comparison op.  Pure read of integer vars -> always defined;
    the branch it guards is therefore per-thread deterministic."""
    uint_t = T.u32
    a = ctx.kx_random_var(uint_t, recency=0.6)
    # Compare against either another live value or a small masked constant so
    # the branch is actually taken sometimes (not statically always-true/false
    # in an obvious way the compiler folds identically -- divergence is the
    # point).
    if ctx.rnd_bool(0.5):
        b = ctx.kx_random_var(uint_t, recency=0.4)
    else:
        b = BinOp(uint_t, op=Arith.BAND,
                  lhs=ctx.kx_random_var(uint_t, recency=0.3),
                  rhs=Lit(uint_t, value=ctx.rnd_choice([1, 3, 7, 15])))
    op = ctx.rnd_choice(list(Cmp))
    return RelOp(T.boolean, op=op, lhs=a, rhs=b)


# ---------------------------------------------------------------------------
# if / if-else
# ---------------------------------------------------------------------------

@weave_block("ctrl_if", inputs=(T.u32,), cost=3)
def _blk_ctrl_if(ctx) -> None:
    """if (cond) { <woven body, observed> } [ else { <woven body, observed> } ]

    A divergent branch.  The condition is a deterministic per-thread
    comparison.  Each branch weaves a small nested codegen body (so the
    produced values are observed inside the branch and fold into _fv on the
    taken path) plus a direct folded value to guarantee the branch is never
    empty / DCE'd.  if_depth is bumped around each branch so woven statements
    record uniform=False."""
    cond = _kx_det_condition(ctx)
    sub_budget = max(2, ctx.kx_budget_for_complexity() // 4)

    with _kx_branch_scope(ctx) as then_stmts:
        ctx.weave_nested(sub_budget)
        ctx.observe(ctx.build_observed_int(T.u32))
    then_block = Block(then_stmts)

    else_block = None
    if ctx.rnd_bool(0.5):
        with _kx_branch_scope(ctx) as else_stmts:
            ctx.weave_nested(sub_budget)
            ctx.observe(ctx.build_observed_int(T.u32))
        else_block = Block(else_stmts)

    ctx.emit(If(cond=cond, then=then_block, otherwise=else_block))


# ---------------------------------------------------------------------------
# bounded for-loop with a loop-carried accumulator
# ---------------------------------------------------------------------------

@weave_block("ctrl_for", inputs=(T.u32,), cost=4)
def _blk_ctrl_for(ctx) -> None:
    """for (int i=0; i<TRIP; ++i) { acc = acc*K ^ f(i, ...); } then observe(acc).

    Fixed, small, assemble-time trip count, clamped against the global
    trip-count budget so nested loops cannot run away.  The accumulator
    (``acc``) is declared in the OUTER block (so it stays live after the loop
    and can be folded), updated each iteration, and observed once the loop
    closes.  All arithmetic is unsigned (wraps, no UB).  loop_depth is bumped
    around the body so body statements record uniform=False."""
    uint_t = T.u32
    int_t = T.i32

    trip = _kx_bounded_trip(ctx, [2, 3, 4, 6, 8])

    # Loop-carried accumulator, declared OUTSIDE the loop body so it is still
    # in kx_scope for the post-loop observe.  Seeded from a live value (or 1).
    seed = ctx.kx_random_var(uint_t, recency=0.5)
    acc = ctx.fresh(uint_t, init=seed)

    loop_var_name = ctx.kx_fresh_name("i")
    loop_var = Var(int_t, name=loop_var_name)

    with _kx_loop_scope(ctx, trip) as body_stmts:
        # Make the loop var visible inside the body kx_scope for any woven reads.
        ctx.kx_declare(loop_var)
        # i mixed in as an unsigned value.
        i_u = Cast(uint_t, to=uint_t, arg=loop_var)
        # Optionally weave a tiny extra body for diversity (keeps it bounded).
        if ctx.rnd_bool(0.5):
            ctx.weave_nested(max(2, ctx.kx_budget_for_complexity() // 6))
        rhs = BinOp(uint_t, op=Arith.MUL,
                    lhs=Var(uint_t, name=acc.name),
                    rhs=Lit(uint_t, value=0x9E3779B1))
        mixed = BinOp(uint_t, op=Arith.BXOR, lhs=rhs, rhs=i_u)
        ctx.emit(Assign(Var(uint_t, name=acc.name), mixed))

    ctx.emit(For(
        var=loop_var_name,
        start=Lit(int_t, value=0),
        end=Lit(int_t, value=trip),
        step=Lit(int_t, value=1),
        body=Block(body_stmts),
    ))
    # acc is still in kx_scope (declared in the outer block): fold it in.
    ctx.observe(Var(uint_t, name=acc.name))


# ---------------------------------------------------------------------------
# bounded while-loop (data-dependent body, counter-terminated)
# ---------------------------------------------------------------------------

@weave_block("ctrl_while", inputs=(T.u32,), cost=4)
def _blk_ctrl_while(ctx) -> None:
    """int w=0; while (w < TRIP) { acc ^= g(w, ...); ++w; } then observe(acc).

    Bounded BY CONSTRUCTION: the counter ``w`` starts at 0, is incremented
    every iteration, and the only loop condition is ``w < TRIP`` with a fixed
    assemble-time TRIP -- so it always terminates in exactly TRIP iterations
    regardless of data.  (The body MAY also mix in data-dependent values, but
    the termination is purely the counter, so there is no risk of an infinite
    loop.)  TRIP is clamped against the global trip budget.  loop_depth is
    bumped around the body so body statements record uniform=False."""
    uint_t = T.u32
    int_t = T.i32

    trip = _kx_bounded_trip(ctx, [2, 3, 4, 6])

    seed = ctx.kx_random_var(uint_t, recency=0.5)
    acc = ctx.fresh(uint_t, init=seed)

    counter_name = ctx.kx_fresh_name("w")
    counter = Var(int_t, name=counter_name)
    # Counter declared in the OUTER block (decl emitted now) so it terminates
    # the loop and stays balanced; loop condition reads it.
    ctx.emit(Declare(name=counter_name, gtype=int_t,
                     init=Lit(int_t, value=0)))
    ctx.kx_declare(counter)

    with _kx_loop_scope(ctx, trip) as body_stmts:
        w_u = Cast(uint_t, to=uint_t, arg=Var(int_t, name=counter_name))
        mixed = BinOp(uint_t, op=Arith.BXOR,
                      lhs=Var(uint_t, name=acc.name),
                      rhs=BinOp(uint_t, op=Arith.MUL,
                                lhs=w_u,
                                rhs=Lit(uint_t, value=0x85EBCA6B)))
        ctx.emit(Assign(Var(uint_t, name=acc.name), mixed))
        # Always-present terminating increment (LAST in the body).
        ctx.emit(IncDecStmt(Var(int_t, name=counter_name),
                            op=IncDec.INC, prefix=False))

    cond = RelOp(T.boolean, op=Cmp.LT,
                 lhs=Var(int_t, name=counter_name),
                 rhs=Lit(int_t, value=trip))
    ctx.emit(While(cond=cond, body=Block(body_stmts)))
    ctx.observe(Var(uint_t, name=acc.name))


# ---------------------------------------------------------------------------
# data-dependent for-loop (runtime, but bounded) -- the observable_fold
# divergence driver: a runtime trip bound prevents full unrolling so compilers
# differ on unroll/trip heuristics.
# ---------------------------------------------------------------------------

@weave_block("ctrl_for_datadep", inputs=(T.u32,), cost=4)
def _blk_ctrl_for_datadep(ctx) -> None:
    """for (int i=0; i < ((v & MASK)+1); ++i) { acc = ...; } then observe(acc).

    The trip bound is data-dependent ( (v & MASK) + 1 ) so it is in
    [1, MASK+1] -- ALWAYS >= 1 and <= a small assemble-time bound, hence
    terminating and OOB-free -- but NOT a assemble-time constant, which blocks
    full unrolling and drives the trip/unroll divergence.  The bound is
    computed BEFORE the loop var exists, so it is loop-invariant (cannot
    depend on the counter).
    """
    uint_t = T.u32
    int_t = T.i32

    # Upper cap (assemble-time) for the trip budget accounting.
    cap = _kx_bounded_trip(ctx, [4, 6, 8])
    mask = cap - 1
    if mask < 1:
        # No headroom for a data-dependent loop -> fall back to a single
        # folded value (still observed; keeps the block a no-op-free
        # producer).
        ctx.observe(ctx.build_observed_int(uint_t))
        return

    # Runtime bound: (v & mask) + 1 in [1, mask+1==cap].
    bound = BinOp(int_t, op=Arith.ADD,
                  lhs=BinOp(int_t, op=Arith.BAND,
                            lhs=Cast(int_t, to=int_t,
                                     arg=ctx.kx_random_var(uint_t, recency=0.6)),
                            rhs=Lit(int_t, value=mask)),
                  rhs=Lit(int_t, value=1))

    seed = ctx.kx_random_var(uint_t, recency=0.5)
    acc = ctx.fresh(uint_t, init=seed)

    loop_var_name = ctx.kx_fresh_name("i")
    loop_var = Var(int_t, name=loop_var_name)

    with _kx_loop_scope(ctx, cap) as body_stmts:
        ctx.kx_declare(loop_var)
        i_u = Cast(uint_t, to=uint_t, arg=loop_var)
        rhs = BinOp(uint_t, op=Arith.MUL,
                    lhs=Var(uint_t, name=acc.name),
                    rhs=Lit(uint_t, value=0xC2B2AE35))
        mixed = BinOp(uint_t, op=Arith.BXOR, lhs=rhs, rhs=i_u)
        ctx.emit(CompoundAssign(Var(uint_t, name=acc.name),
                                op=Compound.ADD, value=i_u))
        ctx.emit(Assign(Var(uint_t, name=acc.name), mixed))

    ctx.emit(For(
        var=loop_var_name,
        start=Lit(int_t, value=0),
        end=bound,
        step=Lit(int_t, value=1),
        body=Block(body_stmts),
    ))
    ctx.observe(Var(uint_t, name=acc.name))


# ---------------------------------------------------------------------------
# branchless select chain (ternary) -- non-uniform value mix without a branch
# ---------------------------------------------------------------------------

@weave_block("ctrl_select_chain", inputs=(T.u32,), cost=2)
def _blk_ctrl_select_chain(ctx) -> None:
    """v = (c0 ? x0 : (c1 ? x1 : x2))  -- a nested branchless select chain.

    Predicated, deterministic, UB-free (all operands are live unsigned
    values), folded into _fv.  Stays uniform (no real control flow),
    complementing the branching blocks above by stressing if-conversion /
    predication paths."""
    uint_t = T.u32
    cur = ctx.kx_random_var(uint_t, recency=0.5)
    for _ in range(ctx.rnd_int(1, 3)):
        cond = _kx_det_condition(ctx)
        alt = ctx.kx_random_var(uint_t, recency=0.4)
        cur = Select(uint_t, cond=cond, when_true=cur, when_false=alt)
    ctx.observe(ctx.fresh(uint_t, init=cur))
