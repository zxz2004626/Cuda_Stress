"""Weave-mode memory-access-pattern blocks.

Port of perffuzzer/generator/codegen/gens_memory.py.  These blocks exercise
diverse READ patterns from the read-only ``input`` buffer (size ``n``) and
fold every loaded value into the running checksum ``_fv`` via ``ctx.observe``:

  * Every index is wrapped to ``[0, n)`` with ``GPU_SMOD(idx, n)`` (the
    gpuperf spelling of perffuzzer's ``SAFE_MOD``) -- there is NO
    out-of-bounds access.  ``n`` is the kernel's element count (cast to
    unsigned for the mod).
  * Index arithmetic is done at ``unsigned`` width, where overflow is
    defined, so the pre-mod address computation is UB-free even for huge
    strides.
  * We only READ ``input`` (it is ``const __restrict``); we NEVER write
    ``input`` and NEVER write ``output`` directly -- the single output write
    is owned by the template's running-fold.  We just ``observe``.

The ``__ldg`` / cache-hint loads wrap the ExprGen builders (``_ldg_load`` /
``_cuda_cache_load``), which carry their own arch gates and look up
``input``/``tid``/``n`` from kx_scope -- so those are registered CUDA-only.

``mem_vectorized_load`` and ``mem_constant_load`` are feature-coverage gens:
perffuzzer disables them by default (weight=0 unless --enable-feature-gens),
so they are registered here with weight=0.0.
"""

from __future__ import annotations

from generator_ir.ir.typedecl import T, GType, TArray, TPointer
from generator_ir.ir.operators import Arith, SafeOp
from generator_ir.ir.nodes import Expr, Lit, Var, BinOp, Cast, SafeBin, Subscript
from generator_ir.ir.statements import Assign, Declare, RawCode
from generator_ir.gen.expr_builder import ExprGen
from generator_ir.gen.weave_handlers import weave_block


# Strides chosen to span the coalescing spectrum:
#   1   -> fully coalesced (consecutive lanes hit consecutive addresses)
#   32  -> warp-stride (each lane skips a full warp -> partially scattered)
#   large primes -> effectively random gather (poor locality, many cache lines)
_STRIDES = [1, 2, 32, 33, 127, 1031, 4099, 65537]


# ---------------------------------------------------------------------------
# Index helpers (always GPU_SMOD-bounded -> no OOB)
# ---------------------------------------------------------------------------

def _kx_n_uint() -> Cast:
    """(unsigned)n -- the element count as an unsigned bound for GPU_SMOD."""
    return Cast(T.u32, to=T.u32, arg=Var(T.i32, name="n"))


def _kx_safe_index(base: Expr, stride: int, k: Expr) -> SafeBin:
    """Build GPU_SMOD(base + k*stride, n) at unsigned width.

    ``base`` and ``k`` are unsigned exprs; ``stride`` is a python int constant.
    All arithmetic is unsigned (defined overflow), and the final GPU_SMOD keeps
    the result strictly in ``[0, n)`` so the load is in-bounds."""
    offset = BinOp(T.u32, op=Arith.MUL, lhs=k, rhs=Lit(T.u32, value=stride))
    addr = BinOp(T.u32, op=Arith.ADD, lhs=base, rhs=offset)
    return SafeBin(T.u32, op=SafeOp.SMOD, lhs=addr, rhs=_kx_n_uint())


def _fetch(idx: Expr) -> Subscript:
    """input[idx] -- a uint read from the read-only input buffer."""
    return Subscript(T.u32,
                     array=Var(TPointer(T.u32), name="input"),
                     index=idx)


def _c_text(e: Expr) -> str:
    """Render the limited expr forms ``kx_random_var`` returns as C text, for
    embedding into RawCode (perffuzzer relies on ``str(expr)``)."""
    if isinstance(e, Var):
        return e.name
    if isinstance(e, Lit):
        return str(e.value)
    if isinstance(e, Cast):
        return f"(({e.to.spelling()})({_c_text(e.arg)}))"
    raise TypeError(f"cannot render {type(e).__name__} as C text")


# ---------------------------------------------------------------------------
# Strided / gather load (single stride, varied)
# ---------------------------------------------------------------------------

@weave_block("mem_strided_load", inputs=(T.u32,), cost=2)
def _blk_mem_strided_load(ctx) -> None:
    """v = input[GPU_SMOD(base + k*stride, n)] -- one strided/gather load.

    The stride is picked from the coalesced->scattered spectrum (1/32/prime).
    GPU_SMOD bounds the index, so this is UB-free and data-dependent (no DCE).
    Backend-portable (plain pointer index)."""
    base = ctx.kx_random_var(T.u32, loose=True, recency=0.6)
    k = ctx.kx_random_var(T.u32, loose=True, recency=0.4)
    stride = ctx.rnd_choice(_STRIDES)
    val = _fetch(_kx_safe_index(base, stride, k))
    ctx.observe(ctx.fresh(T.u32, init=val))


# ---------------------------------------------------------------------------
# Multi-load gather mix: several strided loads combined with arith
# ---------------------------------------------------------------------------

@weave_block("mem_gather_chain", inputs=(T.u32,), cost=3)
def _blk_mem_gather_chain(ctx) -> None:
    """v = mix( input[i0], input[i1], input[i2] ) with distinct strides.

    Combines 2-3 independent GPU_SMOD-bounded loads (each at a different stride
    so the access pattern mixes coalesced + scattered) via ADD/XOR/MUL on
    unsigned (defined overflow).  Folds the combined value in."""
    base = ctx.kx_random_var(T.u32, loose=True, recency=0.6)
    nloads = ctx.rnd_int(2, 3)
    strides = [ctx.rnd_choice(_STRIDES) for _ in range(nloads)]
    acc = None
    for i, stride in enumerate(strides):
        k = ctx.kx_random_var(T.u32, loose=True, recency=0.3 + 0.1 * i)
        load = _fetch(_kx_safe_index(base, stride, k))
        if acc is None:
            acc = load
        else:
            op = ctx.rnd_choice([Arith.ADD, Arith.BXOR, Arith.MUL])
            acc = BinOp(T.u32, op=op, lhs=acc, rhs=load)
    ctx.observe(ctx.fresh(T.u32, init=acc))


# ---------------------------------------------------------------------------
# Pointer-arith chain: feed one load's value into the next load's index
# ---------------------------------------------------------------------------

@weave_block("mem_pointer_chain", inputs=(T.u32,), cost=3)
def _blk_mem_pointer_chain(ctx) -> None:
    """Data-dependent gather: idx_{i+1} derived from value loaded at idx_i.

    v0 = input[GPU_SMOD(base, n)]
    v1 = input[GPU_SMOD(v0 + k*stride, n)]   (and optionally one more hop)
    Each hop's index is GPU_SMOD-bounded so the chained (pointer-chasing-style)
    gather can never go out of bounds.  Stresses dependent-load latency."""
    base = ctx.kx_random_var(T.u32, loose=True, recency=0.6)
    cur = ctx.fresh(T.u32,
                    init=_fetch(SafeBin(T.u32, op=SafeOp.SMOD,
                                       lhs=base, rhs=_kx_n_uint())))
    ctx.observe(cur)
    for _ in range(ctx.rnd_int(1, 2)):
        stride = ctx.rnd_choice(_STRIDES)
        k = ctx.kx_random_var(T.u32, loose=True, recency=0.4)
        cur = ctx.fresh(T.u32, init=_fetch(_kx_safe_index(cur, stride, k)))
        ctx.observe(cur)


# ---------------------------------------------------------------------------
# __ldg read-only cache load (CUDA-only, arch-gated by ExprGen)
# ---------------------------------------------------------------------------

@weave_block("mem_ldg_load", inputs=(T.u32,), cost=2, backends=("cuda",))
def _blk_mem_ldg_load(ctx) -> None:
    """v = __ldg(&input[GPU_SMOD(tid, n)]) -- read-only (texture) cache load.

    Wraps ExprGen._ldg_load, which builds the GPU_SMOD-bounded address and
    the __ldg() call itself (returning a uint, cast to target if needed).  It
    looks up input/tid/n from kx_scope -- all provided by the template.  CUDA-only
    intrinsic, so backends=('cuda',)."""
    val = ExprGen(ctx)._ldg_load(T.u32, 0)
    ctx.observe(ctx.fresh(T.u32, init=val))


# ---------------------------------------------------------------------------
# Cache-hint loads (__ldcg/__ldca/__ldcs/__ldlu) -- CUDA-only, arch-gated
# ---------------------------------------------------------------------------

@weave_block("mem_cache_load", inputs=(T.u32,), cost=2, backends=("cuda",))
def _blk_mem_cache_load(ctx) -> None:
    """v = __ldcg/__ldca/__ldcs/__ldlu(&input[GPU_SMOD(tid, n)]) -- cache-hint
    read-only loads.  Wraps ExprGen._cuda_cache_load, which is gated to
    sm_70+ (returns a terminal otherwise) and GPU_SMOD-bounds the address.
    CUDA-only intrinsics."""
    val = ExprGen(ctx)._cuda_cache_load(T.u32, 0)
    ctx.observe(ctx.fresh(T.u32, init=val))


# ---------------------------------------------------------------------------
# Mixed: combine an __ldg load with a plain strided load (pointer-arith mix)
# ---------------------------------------------------------------------------

@weave_block("mem_ldg_mix", inputs=(T.u32,), cost=3, backends=("cuda",))
def _blk_mem_ldg_mix(ctx) -> None:
    """v = __ldg(&input[..]) ^ input[GPU_SMOD(base + k*stride, n)].

    Couples a read-only-cache load with a normal global load at a (possibly
    scattered) stride, mixing the two cache paths.  Both indices
    GPU_SMOD-bounded.  CUDA-only (uses __ldg)."""
    ldg = ExprGen(ctx)._ldg_load(T.u32, 0)
    base = ctx.kx_random_var(T.u32, loose=True, recency=0.6)
    k = ctx.kx_random_var(T.u32, loose=True, recency=0.4)
    stride = ctx.rnd_choice(_STRIDES)
    plain = _fetch(_kx_safe_index(base, stride, k))
    val = BinOp(T.u32, op=Arith.BXOR, lhs=ldg, rhs=plain)
    ctx.observe(ctx.fresh(T.u32, init=val))


# ---------------------------------------------------------------------------
# 64-bit-index strided load: exercise wide (size_t-style) address arithmetic
# ---------------------------------------------------------------------------

@weave_block("mem_wide_index_load", inputs=(T.u32,), cost=2)
def _blk_mem_wide_index_load(ctx) -> None:
    """v = input[(uint)GPU_SMOD((ull)base * stride + k, (ull)n)].

    Computes the address in 64-bit (unsigned long long), GPU_SMOD-bounds it by
    n, then narrows to uint for the index -- after the mod the value is in
    [0, n) and fits a uint, so the narrowing is exact and in-bounds.  Stresses
    64-bit index math / address generation.  Backend-portable."""
    ull = T.u64
    base = ctx.kx_random_var(ull, loose=True, recency=0.6)
    k = ctx.kx_random_var(ull, loose=True, recency=0.4)
    stride = ctx.rnd_choice(_STRIDES)
    scaled = BinOp(ull, op=Arith.MUL, lhs=base, rhs=Lit(ull, value=stride))
    addr = BinOp(ull, op=Arith.ADD, lhs=scaled, rhs=k)
    n_ull = Cast(ull, to=ull, arg=Var(T.i32, name="n"))
    bounded = SafeBin(ull, op=SafeOp.SMOD, lhs=addr, rhs=n_ull)
    idx = Cast(T.u32, to=T.u32, arg=bounded)
    ctx.observe(ctx.fresh(T.u32, init=_fetch(idx)))


# ---------------------------------------------------------------------------
# Scratch local-array accumulation (registers/local mem, NOT shared)
# ---------------------------------------------------------------------------

@weave_block("mem_scratch_accum", inputs=(T.u32,), cost=3)
def _blk_mem_scratch_accum(ctx) -> None:
    """Load a few GPU_SMOD-bounded values into a small local scratch array,
    then fold the accumulated sum.

        unsigned cv_buf[L];
        cv_buf[0] = input[i0]; ... cv_buf[L-1] = input[i_{L-1}];
        acc = cv_buf[0] + ... + cv_buf[L-1];   // unsigned, defined overflow

    All array indices are assemble-time-constant in [0, L) and all loads are
    GPU_SMOD-bounded -> no OOB.  This is local memory, not shared."""
    L = ctx.rnd_int(2, 4)
    arr_t = TArray(T.u32, L)
    buf_name = ctx.kx_fresh_name("cv_buf")
    buf = Var(arr_t, name=buf_name)
    # Declare the scratch array (uninitialized; every slot is written below).
    ctx.emit(Declare(name=buf_name, gtype=arr_t))

    base = ctx.kx_random_var(T.u32, loose=True, recency=0.6)
    for i in range(L):
        stride = ctx.rnd_choice(_STRIDES)
        k = ctx.kx_random_var(T.u32, loose=True, recency=0.4)
        slot = Subscript(T.u32, array=buf, index=Lit(T.u32, value=i))
        ctx.emit(Assign(slot, _fetch(_kx_safe_index(base, stride, k))))

    acc: Expr = Subscript(T.u32, array=buf, index=Lit(T.u32, value=0))
    for i in range(1, L):
        slot = Subscript(T.u32, array=buf, index=Lit(T.u32, value=i))
        acc = BinOp(T.u32, op=Arith.ADD, lhs=acc, rhs=slot)
    ctx.observe(ctx.fresh(T.u32, init=acc))


@weave_block("mem_vectorized_load", inputs=(T.u32,), weight=0.0, cost=3)
def _blk_mem_vectorized_load(ctx) -> None:
    """128-bit vectorized load: read a ``uint4`` (4 consecutive uints) from
    ``input`` via a 16-byte-aligned reinterpret, folding its 4 lanes into _fv.

    UB-free + deterministic:
      * The base index is ``GPU_SMOD(tid, n/4)`` so the 16-byte read covers
        ``input[4*i .. 4*i+3]`` -- always in-bounds (input has n uints = 4n
        bytes).  A uniform ``n>=4`` guard keeps it safe for any n.
      * ``input`` is cudaMalloc-allocated (>=256-byte aligned), so the
        ``(const uint4*)`` access is well-aligned.
      * Every lane is a deterministic function of fixed input + tid, so the
        fold is identical across runs and across nvcc versions.
    Closes the vector-type gap (only scalar 4-wide unroll existed before).
    Portable: ``uint4`` exists on both CUDA and HIP.

    Feature-coverage gen: registered with weight=0.0 (perffuzzer disables it
    unless --enable-feature-gens)."""
    v4 = ctx.kx_fresh_name("v4")
    v4n = ctx.kx_fresh_name("v4n")
    v4i = ctx.kx_fresh_name("v4i")
    a = ctx.rnd_choice([0x9E3779B9, 0x85EBCA6B, 0xC2B2AE35, 0x27D4EB2F])
    b = ctx.rnd_choice([0x165667B1, 0xD3A2646C, 0xFD7046C5, 0xB55A4F09])
    c = ctx.rnd_choice([0x9E3779B9, 0x85EBCA6B, 0xC2B2AE35, 0x27D4EB2F])
    code = (
        "{ // 128-bit vectorized uint4 load (bounded, aligned, deterministic)\n"
        f"    unsigned int {v4n} = (((unsigned int)(n)) >= 4u)"
        f" ? (((unsigned int)(n)) >> 2) : 1u;\n"
        f"    unsigned int {v4i} = GPU_SMOD(tid, {v4n});\n"
        f"    if (((unsigned int)(n)) >= 4u) {{\n"
        f"        uint4 {v4} = ((const uint4*)input)[{v4i}];\n"
        f"        _fv = (((_fv) * 2654435761u) ^ {v4}.x ^ ({v4}.y * {a}u)"
        f" ^ ({v4}.z * {b}u) ^ ({v4}.w * {c}u));\n"
        "    }\n"
        "}"
    )
    ctx.emit(RawCode(text=code))


@weave_block("mem_constant_load", inputs=(T.u32,), weight=0.0, cost=2,
             backends=("cuda",))
def _blk_mem_constant_load(ctx) -> None:
    """Read from a host-initialised ``__constant__`` pool ``_cmem[256]``.

    Exercises the constant-memory space / constant cache -- a codegen path the
    fuzzer never touched before (only __device__ global + shared + registers).
    Setting ``ctx._needs_constant_mem`` makes the kx_translator emit the
    file-kx_scope ``__constant__`` decl plus a deterministic host fill via
    cudaMemcpyToSymbol.  Both indices are ``GPU_SMOD(.., 256)``-bounded
    (in-bounds), and the pool holds a fixed pattern, so reads are identical
    across runs and nvcc versions.  CUDA-only (constant-symbol plumbing is
    emitted only for the CUDA backend).

    Feature-coverage gen: registered with weight=0.0 (perffuzzer disables it
    unless --enable-feature-gens)."""
    ctx._needs_constant_mem = True
    base = _c_text(ctx.kx_random_var(T.u32, loose=True, recency=0.6))
    cr = ctx.kx_fresh_name("cmr")
    code = (
        "{ // __constant__ memory read (bounded, deterministic)\n"
        f"    unsigned int {cr} = _cmem[GPU_SMOD((unsigned int)({base}), 256u)]"
        f" ^ _cmem[GPU_SMOD((((unsigned int)(tid)) + 1u), 256u)];\n"
        f"    _fv = (((_fv) * 2654435761u) ^ {cr});\n"
        "}"
    )
    ctx.emit(RawCode(text=code))
