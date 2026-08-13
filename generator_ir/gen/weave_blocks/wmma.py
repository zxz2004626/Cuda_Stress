"""Tensor-Core (WMMA) weave block.

Port of perffuzzer/generator/codegen/gens_wmma.py to a gpuperf weave block:
a single whole-warp INT8 matrix-multiply-accumulate via the ``nvcuda::wmma``
API (16x16x16, ``signed char`` x ``signed char`` -> ``int`` accumulator).
Turing (sm_75) has INT8 tensor cores, so this is HW-native there.

Determinism (the load-bearing invariant):
  * INT8 WMMA accumulates in a 32-bit INTEGER accumulator -- the result is EXACT
    (no FP rounding), so it folds into the integer checksum ``_fv`` directly and
    is bit-identical across nvcc 11.5 vs 13.1 (both target sm_75, same fragment
    layout) and stable across runs.
  * The A/B tiles are filled from a deterministic function of tid + input and
    masked to [0,127], so every product/sum is a fixed function of the input;
    the s32 accumulation cannot overflow (16 * 127*127 ~ 4.1M << 2^31).
  * requires_uniform => the whole block is converged when this fires; the inner
    ``threadIdx.x < 32`` selects warp 0, which is therefore FULLY converged (all
    32 lanes), satisfying the warp-collective contract of load/mma/store_*_sync.
    Other warps skip the MMA; every thread then reads the shared result tile, so
    the fold is a deterministic per-thread function of a published, fixed tile.

UB / safety:
  * Shared tiles are fixed 16x16 (= 256 elements); the fill loop strides by
    blockDim.x and writes ``wa[i]`` for i < 256, then a barrier publishes them
    before the MMA reads.  The final fold reads ``wc[GPU_SMOD(tid, 256)]`` -- an
    initialized, in-bounds slot.
  * CUDA-only (``backends=("cuda",)``): the ``nvcuda::wmma`` API + ``<mma.h>``.
    ``once`` so a kernel has at most one (expensive) MMA block.

This is one of perffuzzer's feature-coverage generators, registered with
weight=0.0 (disabled by default) -- preserved here.

Raw-code note: perffuzzer's fragment uses ``SAFE_MOD``; gpuperf's safe-op
macro prelude provides the same guarded modulo as ``GPU_SMOD``.
"""

from __future__ import annotations

from generator_ir.ir.typedecl import T
from generator_ir.ir.statements import RawCode
from generator_ir.gen.weave_handlers import weave_block


@weave_block("tensor_wmma_int8", inputs=(T.u32,),
             requires_uniform=True, weight=0.0, cost=5,
             backends=("cuda",), once=True)
def _blk_tensor_wmma_int8(ctx) -> None:
    ctx._needs_wmma = True
    wa = ctx.kx_fresh_name("wmma_a")
    wb = ctx.kx_fresh_name("wmma_b")
    wc = ctx.kx_fresh_name("wmma_c")
    fa = ctx.kx_fresh_name("wf_a")
    fb = ctx.kx_fresh_name("wf_b")
    fc = ctx.kx_fresh_name("wf_c")
    code = (
        "{ // INT8 Tensor-Core MMA 16x16x16 (deterministic integer accumulate)\n"
        "    using namespace nvcuda;\n"
        # 32-byte alignment: WMMA load/store_matrix_sync require an aligned base
        # pointer; without this the tiles land at a shared-memory offset that
        # depends on preceding shared decls, intermittently misaligning the
        # tensor-core access (a deterministic fault on both compilers).
        f"    __shared__ __align__(32) signed char {wa}[256];\n"
        f"    __shared__ __align__(32) signed char {wb}[256];\n"
        f"    __shared__ __align__(32) int {wc}[256];\n"
        f"    for (unsigned int i = threadIdx.x; i < 256u; i += blockDim.x) {{\n"
        f"        {wa}[i] = (signed char)((input[GPU_SMOD(i, ((unsigned int)(n)))] ^ i) & 0x7fu);\n"
        f"        {wb}[i] = (signed char)((input[GPU_SMOD((i + 1u), ((unsigned int)(n)))] + i) & 0x7fu);\n"
        "    }\n"
        "    __syncthreads();\n"
        "    if (threadIdx.x < 32u) {\n"
        f"        wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> {fa};\n"
        f"        wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> {fb};\n"
        f"        wmma::fragment<wmma::accumulator, 16, 16, 16, int> {fc};\n"
        f"        wmma::load_matrix_sync({fa}, {wa}, 16);\n"
        f"        wmma::load_matrix_sync({fb}, {wb}, 16);\n"
        f"        wmma::fill_fragment({fc}, 0);\n"
        f"        wmma::mma_sync({fc}, {fa}, {fb}, {fc});\n"
        f"        wmma::store_matrix_sync({wc}, {fc}, 16, wmma::mem_row_major);\n"
        "    }\n"
        "    __syncthreads();\n"
        f"    _fv = (((_fv) * 2654435761u) ^ ((unsigned int){wc}[GPU_SMOD(tid, 256u)]));\n"
        "    __syncthreads();\n"
        "}"
    )
    ctx.emit(RawCode(text=code))
