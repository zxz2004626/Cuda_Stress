"""Synchronization / shared-memory / warp-collective weave blocks.

Port of perffuzzer/generator/codegen/gens_sync.py (the S5 family) to gpuperf
weave blocks.  Every block here is decorated ``requires_uniform=True`` so it
only fires at the uniform top level of the woven body: when these fire the
ENTIRE block reaches the barrier / the ENTIRE warp is active.  That is what
makes the patterns race-free AND deadlock-free by construction.

Determinism (the load-bearing invariant):
  * A barrier / warp-collective result that depends on *which lanes/threads are
    active* is nondeterministic across schedulings.  Because these run only in a
    fully-converged context (all block threads / all 32 warp lanes active), the
    active set is the FULL set -- a constant -- so the observed values are stable
    across runs AND across nvcc 11.5 vs 13.1.
  * We never observe ``__activemask()`` (its value depends on scheduling); we use
    the full mask ``0xffffffff`` on a provably-converged warp instead.
  * Inputs come from the shared pool (``ctx.kx_random_var``), which is seeded
    deterministically from ``tid`` + literals, so every written/reduced value is a
    deterministic function of thread id and fixed device input.

UB / safety:
  * Shared array is fixed-size (>= max supported block dim); writes use
    ``threadIdx.x`` (< blockDim.x <= size) and every READ index is reduced
    ``% blockDim.x`` so it only ever touches an INITIALIZED, in-bounds slot (no
    OOB, no uninitialized read).
  * Warp shuffle lane/delta operands are small constants in [1,16] (< warpSize),
    so no out-of-range shuffle.  All arithmetic folded is unsigned (wraps, no UB).

Backends:
  * Block-barrier reduction + warp shuffle reduction are CUDA+HIP: the shuffle
    uses the IR ``WarpShuffle`` node, which the HIP emitter rewrites to the
    maskless ``__shfl_*`` form automatically (CUDA gets ``__shfl_*_sync`` with
    the full mask).  ``warpSize`` is used (32 on NVIDIA, 32/64 on AMD) so the
    loop is wavefront-correct.
  * Ballot/popc population count is CUDA-only (``backends=("cuda",)``): HIP's
    ``__ballot`` is 64-bit and lane-width dependent, so we don't fold it on HIP.
  * We deliberately do NOT emit ``__reduce_*_sync`` (sm_80+) -- acceptance
    compiles at sm_75, and the shuffle reduction covers the same warp-reduce
    ground portably.

Feature-coverage blocks (perffuzzer disables them by default with weight=0.0;
preserved here): ``warp_shuffle_scan``, ``warp_vote_any_all``,
``shared_atomic_reduce``, ``block_predicate_barrier``,
``coop_groups_reduce_scan``, ``block_device_fence``.

Raw-code notes: perffuzzer's verbatim fragments use ``SAFE_MOD``; gpuperf's
safe-op macro prelude provides the same guarded modulo as ``GPU_SMOD``, so the
raw fragments below use ``GPU_SMOD`` (identical semantics).
"""

from __future__ import annotations

from generator_ir.ir.typedecl import T, GType, TArray, MemSpace, Qual
from generator_ir.ir.operators import Arith, Cmp, SafeOp
from generator_ir.ir.nodes import (
    Expr, Lit, Var, BinOp, RelOp, Cast, SafeBin, Subscript, Call,
    ThreadBuiltin, WarpShuffle, WarpVote,
)
from generator_ir.ir.statements import Assign, Declare, Barrier, RawCode
from generator_ir.emit.base_emitter import BaseEmitter
from generator_ir.gen.weave_handlers import weave_block


# Fixed shared-memory capacity: >= the largest block dim we launch (<=1024).
# Writing smem[threadIdx.x] is therefore always in-bounds, and reads reduced
# % blockDim.x only ever touch slots a thread initialized -> no uninit read.
_SMEM_SIZE = 1024

# Full-warp mask: SAFE only because requires_uniform guarantees a converged warp.
_FULL_MASK = 0xFFFFFFFF

# Backend-neutral expression -> C text renderer, for interpolating pool values
# into RawCode fragments (perffuzzer relied on Expr.__str__).
_RENDERER = BaseEmitter()


def _c(e: Expr) -> str:
    return _RENDERER.expr(e)


def _kx_threadidx_x() -> ThreadBuiltin:
    return ThreadBuiltin(T.u32, axis="x", source="threadIdx")


def _kx_blockdim_x() -> ThreadBuiltin:
    return ThreadBuiltin(T.u32, axis="x", source="blockDim")


def _kx_warpsize() -> ThreadBuiltin:
    return ThreadBuiltin(T.u32, axis="x", source="warpSize")


def _pool_src(ctx) -> Expr:
    """A deterministic pool value to publish/reduce (perffuzzer recency=0.6)."""
    return ctx.kx_random_var(T.u32, loose=True, recency=0.6)


# ---------------------------------------------------------------------------
# Block-barrier reduction via __shared__ memory (CUDA + HIP)
# ---------------------------------------------------------------------------

@weave_block("block_barrier_reduce", inputs=(T.u32,),
             requires_uniform=True, cost=4)
def _blk_block_barrier_reduce(ctx) -> None:
    """__shared__ scratch[N]; scratch[tIdx]=val; __syncthreads();
    r = scratch[tIdx] ^ scratch[(tIdx ^ k) % blockDim]; observe(r).

    RACE-FREE: every thread writes its OWN slot, then a single barrier
    publishes all writes before any read.  DEADLOCK-FREE: requires_uniform =>
    every thread in the block reaches __syncthreads().  DETERMINISTIC: the
    written value is a deterministic function of tid+input; the read index is
    reduced % blockDim.x so it only reads an initialized, in-bounds slot."""
    t = T.u32

    # Distinct shared array per firing.
    smem_name = ctx.kx_fresh_name("s5_smem")
    arr_ty = TArray(t, _SMEM_SIZE)
    smem_var = Var(arr_ty, name=smem_name, space=MemSpace.SHARED)
    ctx.emit(Declare(name=smem_name, gtype=arr_ty, space=MemSpace.SHARED,
                     quals=[Qual.SHARED_MEM]))

    tidx = _kx_threadidx_x()
    # Value to publish: a real pool value mixed with tid (deterministic).
    src = _pool_src(ctx)
    val = BinOp(t, op=Arith.BXOR,
                lhs=src,
                rhs=BinOp(t, op=Arith.MUL,
                          lhs=tidx, rhs=Lit(t, value=0x9E3779B9)))
    # Write own slot (threadIdx.x < blockDim.x <= _SMEM_SIZE -> in-bounds).
    ctx.emit(Assign(Subscript(t, array=smem_var, index=tidx), val))

    # Barrier: publish all writes (whole block converged).
    ctx.emit(Barrier())

    # Butterfly read of a peer's published value; index reduced % blockDim.x so
    # it is always an initialized in-bounds slot (no OOB / uninitialized read).
    k = ctx.rnd_choice([1, 2, 4, 8, 16])
    peer_idx = SafeBin(t, op=SafeOp.SMOD,
                       lhs=BinOp(t, op=Arith.BXOR,
                                 lhs=tidx, rhs=Lit(t, value=k)),
                       rhs=_kx_blockdim_x())
    own = Subscript(t, array=smem_var, index=tidx)
    peer = Subscript(t, array=smem_var, index=peer_idx)
    reduced = BinOp(t, op=Arith.BXOR, lhs=own, rhs=peer)

    # Materialize + fold.  A trailing barrier keeps the smem live across the
    # read before any (future) reuse -- safe at uniform top level.
    res = ctx.fresh(t, init=reduced)
    ctx.emit(Barrier())
    ctx.observe(res)


# ---------------------------------------------------------------------------
# Warp shuffle reduction (CUDA + HIP via the IR WarpShuffle node)
# ---------------------------------------------------------------------------

def _kx_shfl_xor(value: Expr, lane_const: int) -> WarpShuffle:
    """Build a warp shuffle-XOR of *value* by *lane_const*.

    The IR node carries the full mask; the CUDA emitter emits
    __shfl_xor_sync(0xffffffff, value, lane), the HIP emitter rewrites it to
    the maskless __shfl_xor(value, lane).  SAFE: full converged warp."""
    return WarpShuffle(T.u32, kind="shfl_xor_sync",
                       mask=Lit(T.u32, value=_FULL_MASK),
                       value=value,
                       delta=Lit(T.u32, value=lane_const),
                       width=32)


@weave_block("warp_shuffle_reduce", inputs=(T.u32,),
             requires_uniform=True, cost=4)
def _blk_warp_shuffle_reduce(ctx) -> None:
    """Butterfly XOR-reduction across a warp using shuffle:
        acc = val; for k in {16,8,4,2,1}: acc ^= __shfl_xor(acc, k); observe(acc).

    All 32 lanes participate (requires_uniform => converged warp), so after the
    butterfly every lane holds the SAME warp-wide XOR -- a value that is constant
    regardless of scheduling, hence stable across runs and across compilers.
    Portable: WarpShuffle lowers to __shfl_xor_sync(full mask) on CUDA and
    maskless __shfl_xor on HIP."""
    t = T.u32
    src = _pool_src(ctx)
    # Seed accumulator with a deterministic per-lane value.
    acc = ctx.fresh(t, init=src)
    # Fixed butterfly schedule (lanes 16..1) -- a full-warp reduction.
    for k in (16, 8, 4, 2, 1):
        nxt = BinOp(t, op=Arith.BXOR, lhs=acc, rhs=_kx_shfl_xor(acc, k))
        ctx.emit(Assign(acc, nxt))
    ctx.observe(acc)


@weave_block("warp_shuffle_scan", inputs=(T.u32,),
             requires_uniform=True, weight=0.0, cost=4)
def _blk_warp_shuffle_scan(ctx) -> None:
    """Hillis-Steele up-scan via __shfl_up_sync, plus one __shfl_down_sync mix.

        acc = val; for d in {1,2,4,8,16}: acc += __shfl_up_sync(full, acc, d);
        acc ^= __shfl_down_sync(full, acc, 1); observe(acc).

    All 32 lanes participate (requires_uniform => converged warp).  For lanes
    whose source lane is out of range, __shfl_up/down_sync returns the caller's
    OWN value (defined behaviour), so every step is a deterministic function of
    the per-lane seed -- stable across runs and across nvcc 11.5/13.1.  Unsigned
    add wraps (no UB).  Portable: WarpShuffle lowers to __shfl_*_sync(full
    mask) on CUDA and maskless __shfl_* on HIP."""
    t = T.u32
    full = Lit(t, value=_FULL_MASK)
    src = _pool_src(ctx)
    acc = ctx.fresh(t, init=src)
    for d in (1, 2, 4, 8, 16):
        up = WarpShuffle(t, kind="shfl_up_sync", mask=full,
                         value=acc, delta=Lit(t, value=d), width=32)
        ctx.emit(Assign(acc, BinOp(t, op=Arith.ADD, lhs=acc, rhs=up)))
    dn = WarpShuffle(t, kind="shfl_down_sync", mask=full,
                     value=acc, delta=Lit(t, value=1), width=32)
    ctx.emit(Assign(acc, BinOp(t, op=Arith.BXOR, lhs=acc, rhs=dn)))
    ctx.observe(acc)


@weave_block("warp_shuffle_broadcast", inputs=(T.u32,),
             requires_uniform=True, cost=3)
def _blk_warp_shuffle_broadcast(ctx) -> None:
    """Broadcast lane 0's value to the whole warp via __shfl(value, 0), then mix
    in the lane id and observe.  shfl(.,0) on a full warp is deterministic; the
    broadcast value is whatever lane 0 holds (a deterministic function of its
    tid), identical on every lane and every compiler."""
    t = T.u32
    src = _pool_src(ctx)
    val = ctx.fresh(t, init=src)
    bcast = WarpShuffle(t, kind="shfl_sync", mask=Lit(t, value=_FULL_MASK),
                        value=val, delta=Lit(t, value=0), width=32)
    lane = BinOp(t, op=Arith.BAND,
                 lhs=_kx_threadidx_x(),
                 rhs=BinOp(t, op=Arith.SUB,
                           lhs=_kx_warpsize(), rhs=Lit(t, value=1)))
    mixed = BinOp(t, op=Arith.ADD, lhs=bcast, rhs=lane)
    ctx.observe(ctx.fresh(t, init=mixed))


# ---------------------------------------------------------------------------
# Ballot + popc uniform-predicate count (CUDA-only)
# ---------------------------------------------------------------------------

@weave_block("warp_ballot_popc", inputs=(T.u32,),
             requires_uniform=True, cost=3, backends=("cuda",))
def _blk_warp_ballot_popc(ctx) -> None:
    """c = __popc(__ballot_sync(full, pred)); observe(c).

    __ballot_sync on a FULL converged warp returns a bitmask whose bits are set
    exactly for the lanes satisfying *pred*.  Because all 32 lanes are active
    (requires_uniform), that mask -- and therefore its population count -- is a
    deterministic function of the per-lane predicate, identical across runs and
    across 11.5/13.1.  CUDA-only: HIP's __ballot is 64-bit / lane-width-dependent.

    The predicate is a deterministic per-lane test (a pool value's low bit), so
    no scheduling-dependent value leaks into the checksum."""
    t = T.u32
    src = _pool_src(ctx)
    # pred: (src & 1) == 0  -- deterministic per lane.
    pred = RelOp(T.i32, op=Cmp.EQ,
                 lhs=BinOp(t, op=Arith.BAND, lhs=src, rhs=Lit(t, value=1)),
                 rhs=Lit(t, value=0))
    pred_int = Cast(T.i32, to=T.i32, arg=pred)
    ballot = WarpVote(t, kind="ballot_sync",
                      mask=Lit(t, value=_FULL_MASK),
                      predicate=pred_int)
    # __popc takes an int and returns an int.
    ballot_int = Cast(T.i32, to=T.i32, arg=ballot)
    popc = Call(T.i32, fn="__popc", args=[ballot_int], result_ty=T.i32)
    ctx.observe(ctx.fresh(t, init=popc))


@weave_block("warp_vote_any_all", inputs=(T.u32,),
             requires_uniform=True, weight=0.0, cost=3, backends=("cuda",))
def _blk_warp_vote_any_all(ctx) -> None:
    """a = (__any_sync(full, pred)!=0); b = (__all_sync(full, pred)!=0); fold.

    On a FULL converged warp (requires_uniform), the any/all vote over a
    deterministic per-lane predicate is a constant independent of scheduling,
    so the folded value is stable across runs and across nvcc 11.5/13.1.  We
    normalise each vote to {0,1} (``!= 0``) so we never depend on the exact
    non-zero value the intrinsic returns.  CUDA-only (HIP vote width differs)."""
    t = T.u32
    src = _pool_src(ctx)
    k = ctx.rnd_choice([1, 3, 7])
    pred = RelOp(T.i32, op=Cmp.EQ,
                 lhs=BinOp(t, op=Arith.BAND, lhs=src, rhs=Lit(t, value=k)),
                 rhs=Lit(t, value=0))
    pred_int = Cast(T.i32, to=T.i32, arg=pred)
    full = Lit(t, value=_FULL_MASK)
    any_v = WarpVote(T.i32, kind="any_sync", mask=full, predicate=pred_int)
    all_v = WarpVote(T.i32, kind="all_sync", mask=full, predicate=pred_int)
    a = Cast(t, to=t,
             arg=RelOp(T.i32, op=Cmp.NE, lhs=any_v, rhs=Lit(T.i32, value=0)))
    b = Cast(t, to=t,
             arg=RelOp(T.i32, op=Cmp.NE, lhs=all_v, rhs=Lit(T.i32, value=0)))
    mixed = BinOp(t, op=Arith.BXOR,
                  lhs=a,
                  rhs=BinOp(t, op=Arith.SHL, lhs=b, rhs=Lit(t, value=1)))
    ctx.observe(ctx.fresh(t, init=mixed))


@weave_block("warp_sync_handoff", inputs=(T.u32,),
             requires_uniform=True, cost=2, backends=("cuda",))
def _blk_warp_sync_handoff(ctx) -> None:
    """Emit a converged CUDA ``__syncwarp`` block and fold a deterministic
    lane-local value.  The synchronization point itself has no data result, so
    the observed value is ordinary unsigned arithmetic around the barrier.

    SAFE: requires_uniform means every lane in every full warp executes the same
    full-mask sync; all supported 1D launch block sizes are multiples of 32."""
    t = T.u32
    src = _pool_src(ctx)
    lane = BinOp(t, op=Arith.BAND,
                 lhs=_kx_threadidx_x(),
                 rhs=Lit(t, value=31))
    seed = BinOp(t, op=Arith.BXOR,
                 lhs=src,
                 rhs=BinOp(t, op=Arith.MUL,
                           lhs=lane,
                           rhs=Lit(t, value=0x45D9F3B)))
    acc = ctx.fresh(t, init=seed)
    ctx.emit(RawCode(text="__syncwarp(0xffffffffu);"))
    mixed = BinOp(t, op=Arith.BXOR,
                  lhs=acc,
                  rhs=BinOp(t, op=Arith.MUL,
                            lhs=_kx_threadidx_x(),
                            rhs=Lit(t, value=0x9E3779B9)))
    ctx.emit(Assign(acc, mixed))
    ctx.emit(RawCode(text="__syncwarp(0xffffffffu);"))
    ctx.observe(acc)


@weave_block("coop_groups_tiled_reduce", inputs=(T.u32,),
             requires_uniform=True, cost=4, backends=("cuda",))
def _blk_coop_groups_tiled_reduce(ctx) -> None:
    """CUDA cooperative-groups tiled partition with a deterministic warp-local
    reduction.  This intentionally avoids ``coalesced_threads`` and active-mask
    observation; the tile is formed from a converged full thread block."""
    ctx._needs_coop_groups = True
    src_s = _c(_pool_src(ctx))
    block = ctx.kx_fresh_name("cg_block")
    tile = ctx.kx_fresh_name("cg_tile")
    lane = ctx.kx_fresh_name("cg_lane")
    val = ctx.kx_fresh_name("cg_val")
    off = ctx.kx_fresh_name("cg_off")
    code = (
        "{ // Cooperative groups tiled reduce (deterministic full tile)\n"
        "    namespace cg = cooperative_groups;\n"
        f"    cg::thread_block {block} = cg::this_thread_block();\n"
        f"    cg::thread_block_tile<32> {tile} = cg::tiled_partition<32>({block});\n"
        f"    unsigned int {lane} = {tile}.thread_rank();\n"
        f"    unsigned int {val} = (({src_s}) ^ (((unsigned int)(tid)) * 747796405u) ^ {lane});\n"
        f"    for (int {off} = 16; {off} > 0; {off} /= 2) {{\n"
        f"        {val} ^= {tile}.shfl_xor({val}, {off});\n"
        "    }\n"
        f"    _fv = (((_fv) * 2654435761u) ^ {val} ^ ({lane} * 2246822519u));\n"
        "}"
    )
    ctx.emit(RawCode(text=code))


@weave_block("block_device_fence", inputs=(T.u32,),
             requires_uniform=True, weight=0.0, cost=3)
def _blk_block_device_fence(ctx) -> None:
    """Shared producer/consumer guarded by an explicit DEVICE-kx_scope
    ``__threadfence()`` (membar.gl), then a barrier.

    Exercises the device-kx_scope memory fence in codegen -- only block-kx_scope
    ``__threadfence_block`` was emitted before.  The fence orders memory but does
    not change the value; the folded peer read is a deterministic function of a
    published, in-bounds shared slot.  requires_uniform => converged block, so
    it is race/deadlock-free and stable across runs and nvcc 11.5/13.1.  CUDA +
    HIP both provide ``__threadfence``."""
    smem = ctx.kx_fresh_name("df_smem")
    val = ctx.kx_fresh_name("df_val")
    peer = ctx.kx_fresh_name("df_peer")
    src = _c(_pool_src(ctx))
    off = ctx.rnd_choice([1, 3, 7, 15, 31])
    code = (
        "{ // device-kx_scope memory fence (__threadfence) around a shared handoff\n"
        f"    __shared__ unsigned int {smem}[{_SMEM_SIZE}];\n"
        f"    unsigned int {val} = (({src})"
        f" ^ (((unsigned int)(tid)) * 2654435761u));\n"
        f"    {smem}[threadIdx.x] = {val};\n"
        "    __threadfence();\n"
        "    __syncthreads();\n"
        f"    unsigned int {peer} = {smem}[GPU_SMOD((threadIdx.x + {off}u),"
        " blockDim.x)];\n"
        f"    _fv = (((_fv) * 2654435761u) ^ {peer});\n"
        "    __syncthreads();\n"
        "}"
    )
    ctx.emit(RawCode(text=code))


@weave_block("shared_atomic_reduce", inputs=(T.u32,),
             requires_uniform=True, weight=0.0, cost=4)
def _blk_shared_atomic_reduce(ctx) -> None:
    """Deterministic block reduction through a shared atomic.

    Every thread applies a COMMUTATIVE + ASSOCIATIVE atomic (Min/Max/And/Or/Xor/
    Add) to a single ``__shared__`` slot seeded with that op's identity; after a
    barrier the FINAL slot value is the reduction over all block threads, which
    is independent of the (scheduling-dependent) atomic ORDER -- hence stable
    across runs AND across nvcc 11.5/13.1.  We fold ONLY the post-barrier slot
    value, never the order-dependent atomic return value.  requires_uniform =>
    every thread reaches the init/barriers/atomic (race-free, deadlock-free).
    Extends atomics beyond shared atomicAdd (Min/Max were absent before).  Both
    CUDA and HIP provide these integer atomics."""
    op, init = ctx.rnd_choice([
        ("Min", "0xFFFFFFFFu"),
        ("Max", "0u"),
        ("And", "0xFFFFFFFFu"),
        ("Or", "0u"),
        ("Xor", "0u"),
        ("Add", "0u"),
    ])
    slot = ctx.kx_fresh_name("sar_slot")
    val = _c(_pool_src(ctx))
    code = (
        "{ // deterministic shared-atomic block reduction (order-independent)\n"
        f"    __shared__ unsigned int {slot}[1];\n"
        f"    if (threadIdx.x == 0u) {{ {slot}[0] = {init}; }}\n"
        "    __syncthreads();\n"
        f"    atomic{op}(&{slot}[0], (unsigned int)({val}));\n"
        "    __syncthreads();\n"
        f"    _fv = (((_fv) * 2654435761u) ^ {slot}[0]);\n"
        "    __syncthreads();\n"
        "}"
    )
    ctx.emit(RawCode(text=code))


@weave_block("block_predicate_barrier", inputs=(T.u32,),
             requires_uniform=True, weight=0.0, cost=3)
def _blk_block_predicate_barrier(ctx) -> None:
    """__syncthreads_count / _and / _or over a deterministic per-thread predicate.

    Each is a block barrier that ALSO returns a block-wide reduction of the
    predicate: ``_count`` = number of threads whose predicate is nonzero, ``_and``
    = 1 iff all are nonzero, ``_or`` = 1 iff any is nonzero.  Because
    requires_uniform guarantees every thread of the block reaches each barrier
    (converged) and the predicate is a deterministic function of tid+input, all
    three results are constants independent of scheduling -- stable across runs
    AND across nvcc 11.5 vs 13.1.  CUDA + HIP both provide these intrinsics."""
    src_s = _c(_pool_src(ctx))
    pred = ctx.kx_fresh_name("sc_pred")
    cnt = ctx.kx_fresh_name("sc_cnt")
    av = ctx.kx_fresh_name("sc_and")
    ov = ctx.kx_fresh_name("sc_or")
    k = ctx.rnd_choice([1, 2, 3, 7, 15])
    code = (
        "{ // block predicate barriers (__syncthreads_count/_and/_or)\n"
        f"    int {pred} = (int)((((unsigned int)({src_s})) & {k}u) != 0u);\n"
        f"    int {cnt} = __syncthreads_count({pred});\n"
        f"    int {av} = __syncthreads_and({pred});\n"
        f"    int {ov} = __syncthreads_or({pred});\n"
        f"    _fv = (((_fv) * 2654435761u) ^ ((unsigned int){cnt})"
        f" ^ (((unsigned int){av}) << 1) ^ (((unsigned int){ov}) << 2));\n"
        "}"
    )
    ctx.emit(RawCode(text=code))


@weave_block("coop_groups_reduce_scan", inputs=(T.u32,),
             requires_uniform=True, weight=0.0, cost=4, backends=("cuda",))
def _blk_coop_groups_reduce_scan(ctx) -> None:
    """cg::reduce + cg::inclusive_scan over a converged 32-thread tile.

    Exercises the actual cooperative-groups collective-algorithm API (vs the
    hand-rolled shuffle reduction already present).  Over a FULL tile
    (requires_uniform => converged), an integer plus/xor/min/max reduction and a
    plus inclusive-scan are order-fixed and thus deterministic -- stable across
    runs and across nvcc 11.5/13.1.  On sm_75 these use the software (shuffle)
    fallback (HW __reduce_*_sync is sm_80+), which is still deterministic.
    CUDA-only; pulls the reduce/scan headers via _needs_coop_groups_reduce."""
    ctx._needs_coop_groups = True
    ctx._needs_coop_groups_reduce = True
    block = ctx.kx_fresh_name("cgr_block")
    tile = ctx.kx_fresh_name("cgr_tile")
    val = ctx.kx_fresh_name("cgr_val")
    red = ctx.kx_fresh_name("cgr_red")
    scn = ctx.kx_fresh_name("cgr_scan")
    src = _c(_pool_src(ctx))
    op = ctx.rnd_choice([
        "cg::plus<unsigned int>()",
        "cg::bit_xor<unsigned int>()",
        "cg::minimum<unsigned int>()",
        "cg::maximum<unsigned int>()",
    ])
    code = (
        "{ // cooperative-groups reduce + inclusive_scan (deterministic tile)\n"
        "    namespace cg = cooperative_groups;\n"
        f"    cg::thread_block {block} = cg::this_thread_block();\n"
        f"    cg::thread_block_tile<32> {tile} = cg::tiled_partition<32>({block});\n"
        f"    unsigned int {val} = (({src})"
        f" ^ (((unsigned int)(tid)) * 2654435761u) ^ {tile}.thread_rank());\n"
        f"    unsigned int {red} = cg::reduce({tile}, {val}, {op});\n"
        f"    unsigned int {scn} = cg::inclusive_scan({tile}, {val},"
        " cg::plus<unsigned int>());\n"
        f"    _fv = (((_fv) * 2654435761u) ^ {red} ^ ({scn} * 2246822519u));\n"
        "}"
    )
    ctx.emit(RawCode(text=code))


@weave_block("dynamic_shared_handoff", inputs=(T.u32,),
             requires_uniform=True, cost=4, backends=("cuda",), once=True)
def _blk_dynamic_shared_handoff(ctx) -> None:
    """Extern dynamic shared memory handoff.

    The launch-size plumbing is driven by ``ctx._needs_dynamic_shared``.  The
    code writes each thread's slot, synchronizes, reads another initialized slot
    via a bounded index, and folds the result into ``_fv``."""
    ctx._needs_dynamic_shared = True
    src_s = _c(_pool_src(ctx))
    smem = ctx.kx_fresh_name("dyn_smem")
    lane = ctx.kx_fresh_name("dyn_lane")
    val = ctx.kx_fresh_name("dyn_val")
    peer = ctx.kx_fresh_name("dyn_peer")
    offset = ctx.rnd_choice([1, 3, 5, 7, 17, 31])
    code = (
        "{ // Dynamic shared memory handoff\n"
        f"    extern __shared__ unsigned int {smem}[];\n"
        f"    unsigned int {lane} = threadIdx.x;\n"
        f"    unsigned int {val} = (({src_s}) ^ (((unsigned int)(tid)) * 2891336453u) ^ {offset}u);\n"
        f"    {smem}[{lane}] = {val};\n"
        "    __syncthreads();\n"
        f"    unsigned int {peer} = {smem}[GPU_SMOD(({lane} + {offset}u), blockDim.x)];\n"
        f"    _fv = (((_fv) * 2654435761u) ^ {peer} ^ ({val} >> 11));\n"
        "    __syncthreads();\n"
        "}"
    )
    ctx.emit(RawCode(text=code))
