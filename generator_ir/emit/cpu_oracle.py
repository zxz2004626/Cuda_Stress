"""IR -> CPU-C++ oracle backend for generator_ir.

Emits a self-contained C++ translation unit from the *same* ``Program`` IR
that produced the GPU testcase, replaying the exact deterministic
Driver-checksum contract shared by ``native/cuda_runner.cpp`` (``--ir-spec``
path) and ``generator_ir/emit/host_harness.py``:

  * module ``__device__`` globals zero-initialized, ``_cmem`` constant pool
    filled with ``i * 2654435761u + 40503u``;
  * input buffer built from the case's ``input_distribution`` (optionally
    XOR-salted through ``CUDA_STRESS_INPUT_SALT``, exactly like the harness);
  * one kernel launch over the recorded grid/block geometry;
  * boost-style u64 hash over the ``buf_size`` u32 output words, printed as
    ``checksum=<u64>``.

Execution model (deterministic by construction — this mirrors the invariants
the generator already guarantees on the GPU side, cf. gen/weave_blocks/sync.py):

  * one persistent pthread per block thread; blocks execute strictly in
    sequence, so shared memory is one buffer set reused block after block;
  * ``__syncthreads()`` -> sense-reversing block barrier whose last arriver
    *commits* all shared-memory buffers (generation guarantees every thread
    reaches it the same number of times);
  * shared memory uses **commit-on-barrier semantics**: writes land in a
    pending buffer stamped with the writing thread; reads return the pending
    value for slots the reading thread itself wrote this phase, otherwise the
    last committed value.  For the generator's warp-synchronous patterns
    (``wc[own] = wc[own] ^ wc[peer] ^ r`` inside a barrier-delimited loop)
    this is exactly the GPU's lockstep result: every lane reads the value
    committed by the previous barrier, then writes its own slot;
  * warp collectives (``__shfl_*_sync`` / ``__all_sync`` / ``__any_sync`` /
    ``__ballot_sync`` / ``__syncwarp``) -> per-warp exchange buffers guarded
    by a sense-reversing warp barrier (generation guarantees converged,
    full-mask, whole warps);
  * atomics (Add/Sub/Xor/Or/And/Min/Max — all commutative and associative)
    -> lock-serialized RMW for shared-memory proxies, GCC ``__atomic``
    builtins for ordinary memory; final state is order-independent either
    way.  Order-sensitive atomics (Exch/CAS/Inc/Dec) are rejected.

The kernel body itself is rendered by the shared ``BaseEmitter`` with only
CUDA qualifiers / API names macro-mapped or declaration-rewritten, so the GPU
testcase body and the oracle body stay structurally identical.  Device code
is C++ under nvcc, so the oracle compiles as C++ (g++) without portability
surprises.

Integer intrinsic semantics are GPU-verified (sm_86 + CUDA 12.6): see the
comments on ``gpu___byte_perm`` / ``gpu_funnel_*`` / ``gpu___sad`` for the
empirically confirmed corners that differ from naive readings of the PTX ISA
or CUDA docs.

Constructs the CPU runtime cannot mirror exactly raise
``CpuOracleUnsupported`` (never a wrong checksum); the strategy then falls
back to the cross-build oracle for that case.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from generator_ir.emit.base_emitter import (
    BaseEmitter, SourceWriter, write_safe_macros, write_diverge_macros,
)
from generator_ir.ir.nodes import (
    Expr, Var, AddrOf, Call, Atomic, WarpShuffle, WarpVote,
)
from generator_ir.ir.statements import (
    Stmt, Assign, CompoundAssign, IncDecStmt, Declare, RawCode, UnrollHint,
)
from generator_ir.ir.program import Func, StructDecl, Program
from generator_ir.ir.typedecl import (
    GType, TArray, TVec, TStruct, TPointer, TQualified, Qual, MemSpace,
    THalf, TBF16, TFloat, TDouble, TUInt, TInt, TChar, TSChar, TShort,
    TUShort,
)


class CpuOracleUnsupported(Exception):
    """The program uses a construct the CPU oracle cannot mirror exactly."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# ---------------------------------------------------------------------------
# Capability scan
# ---------------------------------------------------------------------------

_FP_TYPES = (THalf, TBF16, TFloat, TDouble)

# Shared-memory element types the commit-on-barrier wrapper supports
# (<= 4 bytes, integral).
_SMEM_ELEM_TYPES = (TUInt, TInt, TChar, TSChar, TShort, TUShort)

# Atomic RMW ops that are commutative AND associative: the final memory state
# is independent of the interleaving, so pthread execution on the CPU matches
# any GPU scheduling.  Exch/CAS/Inc/Dec are order-sensitive -> rejected.
_ATOMIC_OPS_OK = {"Add", "Sub", "Xor", "Or", "And", "Min", "Max"}

# Device functions with exact CPU implementations in the runtime prelude.
_CALL_WHITELIST: Set[str] = {
    # integer bit manipulation
    "__clz", "__clzll", "__popc", "__popcll", "__ffs", "__ffsll",
    "__brev", "__brevll", "__byte_perm",
    "__funnelshift_l", "__funnelshift_r", "__funnelshift_lc", "__funnelshift_rc",
    # integer math
    "__mul24", "__umul24", "__mulhi", "__umulhi",
    "__mul64hi", "__umul64hi", "__sad", "__usad",
    "__dp4a", "__dp2a_lo", "__dp2a_hi",
    # cache hints (pure loads/stores on the CPU)
    "__ldg", "__ldcg", "__ldca", "__ldcs", "__ldlu",
    "__stwb", "__stcg", "__stcs", "__stwt",
    # misc portable helpers
    "min", "max", "abs",
    # generator macros (also emitted into the oracle prelude)
    "GPU_FAKE_DIVERGE", "GPU_GROUP_DIVERGE",
}

# RawCode tokens that can never be mirrored.  Checked after comment removal.
_RAW_FORBIDDEN = (
    "wmma", "nvcuda", "cooperative_groups", "cuda::", "__pipeline",
    "cudaTrigger", "cudaGridDependency", "printf", "clock64", "clock(",
    "nanosleep", "__activemask", "__reduce_", "__match_",
    "atomicExch", "atomicCAS", "atomicInc", "atomicDec",
    "__half", "bfloat16", "float", "double", "volatile",
    "malloc", "free(", "new ", "cudaMalloc", "cudaMemcpy", "__managed__",
    "__grid_constant__", "__lowpriority", "membar",
)

_EMPTY_ASM_RE = re.compile(r'asm\s+volatile\s*\(\s*""\s*:::\s*"memory"\s*\)')

# Shared-memory declaration forms emitted by the generator (raw text).  The
# bracket groups are optional so scalar shared variables match too.
_SMEM_RE = re.compile(
    r'__shared__(?:\s+__align__\(\s*\d+\s*\))?'
    r'\s+(unsigned\s+int|signed\s+char|unsigned\s+char|short|unsigned\s+short|int)'
    r'\s+(\w+)'
    r'(?:\s*\[\s*(\d+)\s*\]\s*(?:\[\s*(\d+)\s*\]\s*)?)?;')
_SMEM_DYN_RE = re.compile(
    r'extern\s+__shared__\s+unsigned\s+int\s+(\w+)\s*\[\s*\]\s*;')

# Tokens inside RawCode that imply warp-collective execution.
_RAW_WARP_TOKENS = (
    "__syncwarp", "__shfl_", "__all_sync", "__any_sync", "__ballot_sync",
    "__reduce_", "__match_",
)


def _strip_c_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def _type_problems(t: Optional[GType], problems: List[str]) -> None:
    """Reject FP/vector types anywhere in an IR type tree."""
    if not isinstance(t, GType):
        return
    if isinstance(t, _FP_TYPES):
        problems.append(f"unsupported type {t.spelling()} (integer-only oracle)")
        return
    if isinstance(t, TVec):
        problems.append(f"vector type {t.spelling()} is not mirrored")
        return
    for attr in ("elem", "base"):
        child = getattr(t, attr, None)
        if isinstance(child, GType):
            _type_problems(child, problems)
    if isinstance(t, TStruct):
        for _fname, ft in t.fields:
            _type_problems(ft, problems)
    if isinstance(t, TQualified):
        if Qual.VOLATILE in t.quals:
            problems.append("volatile-qualified type is not mirrored")
        _type_problems(t.base, problems)


def _iter_nodes(program: Program):
    """Yield every expression/statement node reachable from the Program."""
    from generator_ir.ir.traversal import traverse
    yield from traverse(program)


def _iter_raw_text(program: Program):
    for node in _iter_nodes(program):
        if isinstance(node, RawCode):
            yield node.text


def _uses_warp_collectives(program: Program) -> bool:
    for node in _iter_nodes(program):
        if isinstance(node, (WarpShuffle, WarpVote)):
            return True
        if isinstance(node, Call) and node.fn in _RAW_WARP_TOKENS:
            return True
    for text in _iter_raw_text(program):
        if any(tok in text for tok in _RAW_WARP_TOKENS):
            return True
    return False


def _smem_type_ok(t: Optional[GType]) -> bool:
    """Shared memory must be a scalar or 1D/2D array of a supported
    <=4-byte integer type (the commit-on-barrier wrapper's domain)."""
    if isinstance(t, _SMEM_ELEM_TYPES):
        return True
    if not isinstance(t, TArray):
        return False
    if not isinstance(t.elem, _SMEM_ELEM_TYPES):
        if isinstance(t.elem, TArray):
            return isinstance(t.elem.elem, _SMEM_ELEM_TYPES) \
                and not isinstance(t.elem.elem, TArray)
        return False
    return True


def scan_program(program: Program) -> List[str]:
    """Return a list of reasons the program cannot run on the CPU oracle."""
    problems: List[str] = []
    cfg = program.launch

    if not cfg.checksum_only or cfg.time_ci or cfg.time_clock64 \
            or cfg.time_loop or cfg.time_profiler:
        problems.append("oracle requires checksum_only mode (single fresh launch)")
    if getattr(cfg, "codegen", False):
        # Weave-mode memory blocks include device-global / cross-iteration
        # handoffs whose result depends on GPU lockstep scheduling; the
        # pthread model cannot guarantee an exact mirror yet.
        problems.append("codegen/weave mode is not mirrored")

    # -- types everywhere (exprs, casts, calls, decls, params, structs) -----
    for node in _iter_nodes(program):
        for attr in ("ty", "gtype", "to", "result_ty"):
            t = getattr(node, attr, None)
            if isinstance(t, GType):
                _type_problems(t, problems)
        if isinstance(node, Declare):
            if Qual.VOLATILE in node.quals:
                problems.append("volatile declaration is not mirrored")
            if Qual.MANAGED in node.quals or Qual.GRID_CONSTANT in node.quals:
                problems.append("managed/grid_constant declaration is not mirrored")
            if node.space is MemSpace.SHARED and not _smem_type_ok(node.gtype):
                problems.append(
                    f"shared memory of {node.gtype.spelling()} is not mirrored")
    for d in program.decls:
        if isinstance(d, StructDecl):
            _type_problems(d.struct, problems)
        elif isinstance(d, Declare):
            _type_problems(d.gtype, problems)

    if program.fetch_kernel() is None:
        problems.append("program has no kernel")

    # -- calls to functions with exact CPU mirrors ---------------------------
    declared_funcs = {d.name for d in program.decls if isinstance(d, Func)}
    for node in _iter_nodes(program):
        if isinstance(node, Call) and node.fn not in _CALL_WHITELIST \
                and node.fn not in declared_funcs:
            problems.append(f"call to {node.fn!r} has no CPU mirror")

    # -- atomics: commutative ops on u32 storage only -------------------------
    for node in _iter_nodes(program):
        if isinstance(node, Atomic):
            if node.what not in _ATOMIC_OPS_OK:
                problems.append(
                    f"atomic{node.what} is order-sensitive (not commutative)")
            addr_t = getattr(node.addr, "ty", None)
            if isinstance(addr_t, TPointer) and not isinstance(addr_t.base, TUInt):
                problems.append(
                    f"atomic{node.what} on {addr_t.spelling()} is not mirrored")

    # -- raw code ---------------------------------------------------------------
    for text in _iter_raw_text(program):
        clean = _strip_c_comments(text)
        cleaned = _EMPTY_ASM_RE.sub(" ", clean)
        if re.search(r"\basm\b", cleaned):
            problems.append("raw inline asm is not mirrorable")
        for tok in _RAW_FORBIDDEN:
            if tok in cleaned:
                problems.append(f"raw code token {tok!r} is not mirrorable")
                break
        residual = _SMEM_DYN_RE.sub(" ", clean)
        residual = _SMEM_RE.sub(" ", residual)
        if "__shared__" in residual:
            problems.append("unrecognized __shared__ declaration form")

    # -- device-global write/read handoffs ---------------------------------------
    # Plain device arrays are mirrored as ordinary memory; cross-thread
    # write-then-read handoffs through them (weave memory blocks) rely on
    # GPU lockstep scheduling the pthread model cannot reproduce, so any
    # program that writes a device global it also references elsewhere is
    # rejected rather than risked.
    device_globals = {
        d.name for d in program.decls
        if isinstance(d, Declare) and (
            d.space in (MemSpace.GLOBAL, MemSpace.GENERIC)
            or Qual.DEVICE in d.quals)}
    if device_globals:
        writes: Set[str] = set()
        refs: Set[str] = set()

        def _refs(e, into):
            if isinstance(e, Var) and e.name in device_globals:
                into.add(e.name)
            for attr in ("lhs", "rhs", "arg", "array", "index", "addr",
                         "operand", "value", "mask", "predicate", "delta",
                         "cond", "when_true", "when_false", "a", "b", "c"):
                child = getattr(e, attr, None)
                if isinstance(child, Expr):
                    _refs(child, into)
            if isinstance(e, Call):
                for a in e.args:
                    _refs(a, into)

        for node in _iter_nodes(program):
            if isinstance(node, (Assign, CompoundAssign, IncDecStmt)):
                _refs(node.target, writes)
                _refs(getattr(node, "value", None), refs)
                _refs(node.target, refs)
            elif isinstance(node, Atomic):
                _refs(node.addr, writes)
                _refs(node.addr, refs)
                _refs(node.operand, refs)
            elif isinstance(node, Expr):
                _refs(node, refs)
        hazard = sorted(writes & refs)
        if hazard:
            problems.append(
                f"device-global write+read handoff on {hazard} is not mirrored")

    # -- geometry: warp collectives need whole warps -----------------------------
    if program.fetch_kernel() is not None and _uses_warp_collectives(program):
        if cfg.threads_per_block % 32 != 0:
            problems.append(
                f"warp collectives need whole warps ({cfg.threads_per_block} threads/block)")

    # dedup, keeping the first few for a readable error
    seen: Set[str] = set()
    ordered: List[str] = []
    for p in problems:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered[:6]


# ---------------------------------------------------------------------------
# CPU runtime prelude (C++, compiled by g++)
# ---------------------------------------------------------------------------

_RUNTIME_PRELUDE = r"""
/* ---- CPU-oracle runtime: emulates the CUDA execution model deterministically ---- */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <signal.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>

#define __device__
#define __noinline__ __attribute__((noinline))
#define __forceinline__ inline __attribute__((always_inline))
#define __constant__
#define warpSize 32

typedef struct { unsigned x, y, z; } gpu_dim3_t;
static gpu_dim3_t gpu_gridDim, gpu_blockDim;
static __thread gpu_dim3_t t_threadIdx, t_blockIdx;
#define threadIdx (t_threadIdx)
#define blockIdx  (t_blockIdx)
#define blockDim  (gpu_blockDim)
#define gridDim   (gpu_gridDim)

#define __threadfence()        __sync_synchronize()
#define __threadfence_block()  ((void)0)
#define __threadfence_system() __sync_synchronize()
#define __syncwarp(mask) ((void)(mask))

/* ---- min/max (CUDA integer semantics; g++ templates) ----------------------- */
template <class T> static inline T gpu_min(T a, T b) { return a < b ? a : b; }
template <class T> static inline T gpu_max(T a, T b) { return a > b ? a : b; }
static inline int gpu_abs(int x) { return x < 0 ? -x : x; }
#define min gpu_min
#define max gpu_max
#define abs gpu_abs
"""

_RUNTIME_PRELUDE += r"""
/* ---- integer intrinsics (GPU-verified semantics, sm_86 / CUDA 12.6) --------- */
static inline int gpu___clz(unsigned x) { return x ? __builtin_clz(x) : 32; }
static inline int gpu___clzll(unsigned long long x) { return x ? __builtin_clzll(x) : 64; }
static inline int gpu___popc(unsigned x) { return __builtin_popcount(x); }
static inline int gpu___popcll(unsigned long long x) { return __builtin_popcountll(x); }
static inline int gpu___ffs(unsigned x) { return x ? __builtin_ctz(x) + 1 : 0; }
static inline int gpu___ffsll(unsigned long long x) { return x ? __builtin_ctzll(x) + 1 : 0; }

static inline unsigned gpu___brev32(unsigned x) {
    x = ((x & 0x55555555u) << 1) | ((x >> 1) & 0x55555555u);
    x = ((x & 0x33333333u) << 2) | ((x >> 2) & 0x33333333u);
    x = ((x & 0x0F0F0F0Fu) << 4) | ((x >> 4) & 0x0F0F0F0Fu);
    x = ((x & 0x00FF00FFu) << 8) | ((x >> 8) & 0x00FF00FFu);
    return (x << 16) | (x >> 16);
}
static inline unsigned long long gpu___brev64(unsigned long long x) {
    return ((unsigned long long)gpu___brev32((unsigned)x) << 32)
         | (unsigned long long)gpu___brev32((unsigned)(x >> 32));
}
#define __clz(x)    gpu___clz((unsigned)(x))
#define __clzll(x)  gpu___clzll((unsigned long long)(x))
#define __popc(x)   gpu___popc((unsigned)(x))
#define __popcll(x) gpu___popcll((unsigned long long)(x))
#define __ffs(x)    gpu___ffs((unsigned)(x))
#define __ffsll(x)  gpu___ffsll((unsigned long long)(x))
#define __brev(x)   gpu___brev32((unsigned)(x))
#define __brevll(x) gpu___brev64((unsigned long long)(x))

/* __byte_perm: PTX prmt as exposed by the CUDA C intrinsic.  Empirically
 * (sm_86, CUDA 12.6) each selector nibble picks byte (nibble & 7) from the
 * {a,b} pool (a = low word); bit 3 is IGNORED — the intrinsic never does the
 * PTX generic-form sign replication. */
static inline unsigned gpu___byte_perm(unsigned a, unsigned b, unsigned s) {
    unsigned long long pool = ((unsigned long long)b << 32) | a;
    unsigned r = 0;
    for (int i = 0; i < 4; i++) {
        unsigned nib = (s >> (4 * i)) & 0xFu;
        unsigned byte = (unsigned)((pool >> (8 * (nib & 7u))) & 0xFFu);
        r |= byte << (8 * i);
    }
    return r;
}
#define __byte_perm(a, b, s) gpu___byte_perm((a), (b), (s))

/* funnel shifts: (hi:lo) << / >> shift, other word returned.  The plain
 * forms wrap the shift amount modulo 32; the _c forms clamp to min(s, 32)
 * (empirical sm_86 semantics: s>=32 leaves lo for _l and hi for _r). */
static inline unsigned gpu_funnel_l(unsigned lo, unsigned hi, unsigned sh) {
    return (unsigned)((((unsigned long long)hi << 32) | lo) << (sh & 31u) >> 32);
}
static inline unsigned gpu_funnel_lc(unsigned lo, unsigned hi, unsigned sh) {
    unsigned s = sh > 32u ? 32u : sh;
    return (unsigned)((((unsigned long long)hi << 32) | lo) << s >> 32);
}
static inline unsigned gpu_funnel_r(unsigned lo, unsigned hi, unsigned sh) {
    return (unsigned)((((unsigned long long)hi << 32) | lo) >> (sh & 31u));
}
static inline unsigned gpu_funnel_rc(unsigned lo, unsigned hi, unsigned sh) {
    unsigned s = sh > 32u ? 32u : sh;
    return (unsigned)((((unsigned long long)hi << 32) | lo) >> s);
}
#define __funnelshift_l(lo, hi, s)  gpu_funnel_l((lo), (hi), (s))
#define __funnelshift_r(lo, hi, s)  gpu_funnel_r((lo), (hi), (s))
#define __funnelshift_lc(lo, hi, s) gpu_funnel_lc((lo), (hi), (s))
#define __funnelshift_rc(lo, hi, s) gpu_funnel_rc((lo), (hi), (s))

/* 24-bit multiplies: operands truncated to 24 bits (signed variants
 * sign-extended), low 32 bits of the product (PTX mul24 semantics). */
static inline int gpu_s24(int x) { int v = x << 8; return v >> 8; }
static inline int gpu___mul24(int a, int b) {
    return (int)((long long)gpu_s24(a) * (long long)gpu_s24(b));
}
static inline unsigned gpu___umul24(unsigned a, unsigned b) {
    return ((a & 0xFFFFFFu) * (b & 0xFFFFFFu)) & 0xFFFFFFFFu;
}
static inline int gpu___mulhi(int a, int b) {
    return (int)(((long long)a * (long long)b) >> 32);
}
static inline unsigned gpu___umulhi(unsigned a, unsigned b) {
    return (unsigned)((((unsigned long long)a) * b) >> 32);
}
/* mul.hi.s64/u64: high 64 bits of the exact two's-complement product (the
 * hardware/runtime semantics; note nvcc 12.6's ptxas -O3 miscompiles
 * mul.hi.s64 for small negative products — the oracle mirrors the correct
 * math, which is what the differential pipeline wants). */
static inline long long gpu___mul64hi(long long a, long long b) {
    return (__int128)a * b >> 64;
}
static inline unsigned long long gpu___umul64hi(unsigned long long a,
                                                unsigned long long b) {
    return (unsigned __int128)a * b >> 64;
}
#define __mul24(a, b)    gpu___mul24((a), (b))
#define __umul24(a, b)   gpu___umul24((a), (b))
#define __mulhi(a, b)    gpu___mulhi((a), (b))
#define __umulhi(a, b)   gpu___umulhi((a), (b))
#define __mul64hi(a, b)  gpu___mul64hi((a), (b))
#define __umul64hi(a, b) gpu___umul64hi((a), (b))

/* __sad: |x - y| + z with the difference taken over SIGNED integers
 * (empirical sm_86: __sad(0, -1, 0) == 1, not the unsigned 0xFFFFFFFF);
 * __usad is the unsigned variant.  Both wrap the final add modulo 2^32. */
static inline unsigned gpu___sad(int a, int b, unsigned c) {
    long long d = (long long)a - (long long)b;
    d = d < 0 ? -d : d;
    return (unsigned)(d + (unsigned long long)c);
}
static inline unsigned gpu___usad(unsigned a, unsigned b, unsigned c) {
    unsigned d = a > b ? a - b : b - a;
    return d + c;
}
#define __sad(a, b, c)  gpu___sad((a), (b), (c))
#define __usad(a, b, c) gpu___usad((a), (b), (c))

/* unsigned dot products — the generator emits the unambiguous
 * __dp4a/__dp2a_*(unsigned, unsigned, unsigned) CUDA overloads. */
static inline unsigned gpu_u8(unsigned x, int byte) {
    return (x >> (8 * byte)) & 0xFFu;
}
static inline unsigned gpu_u16(unsigned x, int half) {
    return (x >> (16 * half)) & 0xFFFFu;
}
static inline unsigned gpu___dp4a(unsigned a, unsigned b, unsigned c) {
    return gpu_u8(a, 0) * gpu_u8(b, 0) + gpu_u8(a, 1) * gpu_u8(b, 1)
         + gpu_u8(a, 2) * gpu_u8(b, 2) + gpu_u8(a, 3) * gpu_u8(b, 3) + c;
}
static inline unsigned gpu___dp2a_lo(unsigned a, unsigned b, unsigned c) {
    return gpu_u16(a, 0) * gpu_u8(b, 0) + gpu_u16(a, 1) * gpu_u8(b, 1) + c;
}
static inline unsigned gpu___dp2a_hi(unsigned a, unsigned b, unsigned c) {
    return gpu_u16(a, 0) * gpu_u8(b, 2) + gpu_u16(a, 1) * gpu_u8(b, 3) + c;
}
#define __dp4a(a, b, c)     gpu___dp4a((unsigned)(a), (unsigned)(b), (unsigned)(c))
#define __dp2a_lo(a, b, c)  gpu___dp2a_lo((unsigned)(a), (unsigned)(b), (unsigned)(c))
#define __dp2a_hi(a, b, c)  gpu___dp2a_hi((unsigned)(a), (unsigned)(b), (unsigned)(c))

/* cache hints are plain accesses on the CPU */
#define __ldg(p)   (*(p))
#define __ldcg(p)  (*(p))
#define __ldca(p)  (*(p))
#define __ldcs(p)  (*(p))
#define __ldlu(p)  (*(p))
#define __stwb(p, v) ((*(p)) = (v))
#define __stcg(p, v) ((*(p)) = (v))
#define __stcs(p, v) ((*(p)) = (v))
#define __stwt(p, v) ((*(p)) = (v))
"""

_RUNTIME_PRELUDE += r"""
/* ---- shared memory with commit-on-barrier semantics --------------------------
 *
 * GPU warp-synchronous patterns read a warp peer's slot and write their own
 * in the SAME statement, with __syncthreads() only between iterations.  On
 * the GPU every lane reads the pre-statement (last-barrier) value; a plain
 * pthread block cannot reproduce that.  Model:
 *   * writes land in a pending buffer and stamp the word with the writer;
 *   * reads return pending for words the reader itself wrote this phase,
 *     otherwise the last committed value;
 *   * the last thread arriving at __syncthreads() commits every buffer
 *     (pending -> committed, stamps cleared) BEFORE releasing the barrier.
 * For barrier-delimited producer/consumer patterns this is exactly the GPU
 * result, and same-phase cross-warp reads (a real race on the GPU, excluded
 * by the generator's invariants) read the committed value. */
static __thread unsigned g_my_tid;

struct gpu_smem_base {
    gpu_smem_base* next;
    unsigned words;
    unsigned* committed;
    unsigned* pending;
    unsigned* stamp;   /* writer tid + 1, 0 = committed value */
};
static gpu_smem_base* g_smem_head = NULL;

static void gpu_smem_commit_all(void) {
    /* Atomic (relaxed) stores: plain stores could be reordered by the
     * compiler past the block barrier's release atomic, breaking the
     * commit-before-release protocol.  On x86 these compile to plain mov. */
    for (gpu_smem_base* b = g_smem_head; b; b = b->next) {
        for (unsigned i = 0; i < b->words; ++i) {
            __atomic_store_n(&b->committed[i],
                             __atomic_load_n(&b->pending[i], __ATOMIC_RELAXED),
                             __ATOMIC_RELAXED);
            __atomic_store_n(&b->stamp[i], 0u, __ATOMIC_RELAXED);
        }
    }
}

struct gpu_word {
    unsigned* committed; unsigned* pending; unsigned* stamp;
    unsigned idx;
    unsigned rd(void) const {
        unsigned own = __atomic_load_n(&stamp[idx], __ATOMIC_RELAXED);
        unsigned v = own == g_my_tid + 1u
            ? __atomic_load_n(&pending[idx], __ATOMIC_RELAXED)
            : __atomic_load_n(&committed[idx], __ATOMIC_RELAXED);
        return v;
    }
    void wr(unsigned v) const {
        __atomic_store_n(&pending[idx], v, __ATOMIC_RELAXED);
        __atomic_store_n(&stamp[idx], g_my_tid + 1u, __ATOMIC_RELAXED);
    }
};

template <class T> struct gpu_ref_ptr;
template <class T> struct gpu_ref;

template <class T> struct gpu_ref_ptr { gpu_ref<T> r; };

template <class T> struct gpu_ref {
    gpu_word w;
    unsigned shift;   /* byte offset inside the word */
    operator T() const {
        unsigned v = w.rd();
        T out;
        memcpy(&out, (const char*)&v + shift, sizeof(T));   /* little-endian */
        return out;
    }
    /* &(smem[i]) inside atomicAdd(...) etc.: return a by-value wrapper the
     * atomic overloads accept (plain pointer extraction stays unsupported).
     * Field-wise copy: the converting operator= would write through the
     * destination's still-uninitialized word pointers. */
    gpu_ref_ptr<T> operator&(void) const {
        gpu_ref_ptr<T> p;
        p.r.w = w;
        p.r.shift = shift;
        return p;
    }
    const gpu_ref& operator=(T v) const {
        unsigned cur = w.rd();
        memcpy((char*)&cur + shift, &v, sizeof(T));
        w.wr(cur);
        return *this;
    }
    /* smem[b] = smem[a] must copy the VALUE: without this, overload
     * resolution picks the implicit member-wise proxy copy-assignment,
     * which writes nothing. */
    gpu_ref& operator=(const gpu_ref& rhs) {
        T v = (T)rhs;
        operator=(v);
        return *this;
    }
#define GPU_REF_OP(OP, APPLY)                                        \
    template <class U> gpu_ref const& operator OP(U rhs) const {     \
        T cur = operator T();                                        \
        return operator=(T(APPLY));                                  \
    }
    GPU_REF_OP(+=, cur + rhs)
    GPU_REF_OP(-=, cur - rhs)
    GPU_REF_OP(*=, cur * rhs)
    GPU_REF_OP(/=, cur / rhs)
    GPU_REF_OP(%=, cur % rhs)
    GPU_REF_OP(^=, cur ^ rhs)
    GPU_REF_OP(|=, cur | rhs)
    GPU_REF_OP(&=, cur & rhs)
    GPU_REF_OP(<<=, cur << rhs)
    GPU_REF_OP(>>=, cur >> rhs)
#undef GPU_REF_OP
    gpu_ref const& operator++(void) const { T cur = operator T(); operator=(T(cur + 1)); return *this; }
    gpu_ref const& operator--(void) const { T cur = operator T(); operator=(T(cur - 1)); return *this; }
    T operator++(int) const { T cur = operator T(); operator=(T(cur + 1)); return cur; }
    T operator--(int) const { T cur = operator T(); operator=(T(cur - 1)); return cur; }
};

/* scalar shared variable: __shared__ unsigned int bsum; */
template <class T> struct gpu_smem_scalar {
    unsigned c, p, s;
    gpu_smem_base hdr;
    gpu_smem_scalar() {
        c = p = s = 0;
        hdr.next = g_smem_head; hdr.words = 1;
        hdr.committed = &c; hdr.pending = &p; hdr.stamp = &s;
        g_smem_head = &hdr;
    }
    operator T() const { gpu_word w{&c, &p, &s, 0}; gpu_ref<T> r{w, 0}; return r; }
    T operator=(T v) const { gpu_word w{&c, &p, &s, 0}; gpu_ref<T> r{w, 0}; r = v; return v; }
    /* scalar = scalar copies the value, not the wrapper state */
    T operator=(const gpu_smem_scalar& rhs) const { T v = (T)rhs; operator=(v); return v; }
};

/* 1-D shared array */
template <class T, unsigned N> struct gpu_smem_arr {
    unsigned c[N], p[N], s[N];
    gpu_smem_base hdr;
    gpu_smem_arr() {
        memset(c, 0, sizeof(c)); memset(p, 0, sizeof(p)); memset(s, 0, sizeof(s));
        hdr.next = g_smem_head; hdr.words = N;
        hdr.committed = c; hdr.pending = p; hdr.stamp = s;
        g_smem_head = &hdr;
    }
    gpu_ref<T> operator[](unsigned i) {
        gpu_word w{c, p, s, i};
        return gpu_ref<T>{w, (i * sizeof(T)) & 3u};
    }
};

/* 2-D shared array [R][C] */
template <class T, unsigned R, unsigned C> struct gpu_smem_arr2 {
    unsigned c[R * C], p[R * C], s[R * C];
    gpu_smem_base hdr;
    gpu_smem_arr2() {
        memset(c, 0, sizeof(c)); memset(p, 0, sizeof(p)); memset(s, 0, sizeof(s));
        hdr.next = g_smem_head; hdr.words = R * C;
        hdr.committed = c; hdr.pending = p; hdr.stamp = s;
        g_smem_head = &hdr;
    }
    gpu_ref<T>* row_storage(unsigned r) { return NULL; }
    struct row_t {
        gpu_smem_arr2* a; unsigned r;
        gpu_ref<T> operator[](unsigned j) {
            unsigned flat = r * C + j;
            gpu_word w{a->c, a->p, a->s, flat};
            return gpu_ref<T>{w, (flat * sizeof(T)) & 3u};
        }
    };
    row_t operator[](unsigned r) { return row_t{this, r}; }
};

/* dynamic shared: extern __shared__ unsigned int name[]; */
struct gpu_smem_dyn {
    unsigned* c; unsigned* p; unsigned* s;
    gpu_smem_base hdr;
    gpu_smem_dyn(unsigned words) {
        c = (unsigned*)calloc(words, 4); p = (unsigned*)calloc(words, 4);
        s = (unsigned*)calloc(words, 4);
        hdr.next = g_smem_head; hdr.words = words;
        hdr.committed = c; hdr.pending = p; hdr.stamp = s;
        g_smem_head = &hdr;
    }
    gpu_ref<unsigned> operator[](unsigned i) {
        gpu_word w{c, p, s, i};
        return gpu_ref<unsigned>{w, 0};
    }
};

/* ---- shared-memory atomics: lock-serialized commutative RMW ------------------
 * The lock makes the RMW itself atomic; the write follows the pending/stamp
 * path, so barrier-delimited readers see the accumulated value. */
static pthread_mutex_t g_smem_atomic_mu = PTHREAD_MUTEX_INITIALIZER;

#define GPU_SMEM_ATOMIC(OP, EXPR)                                        \
    do {                                                                 \
        pthread_mutex_lock(&g_smem_atomic_mu);                           \
        T cur_ = ref_;                                                   \
        ref_ = (T)(EXPR);                                                \
        pthread_mutex_unlock(&g_smem_atomic_mu);                         \
        return cur_;                                                     \
    } while (0)

template <class T> static inline T gpu_smem_atomic_add(gpu_ref<T> r, T v) {
    gpu_ref<T>& ref_ = r; GPU_SMEM_ATOMIC(+, cur_ + v);
}
template <class T> static inline T gpu_smem_atomic_sub(gpu_ref<T> r, T v) {
    gpu_ref<T>& ref_ = r; GPU_SMEM_ATOMIC(-, cur_ - v);
}
template <class T> static inline T gpu_smem_atomic_and(gpu_ref<T> r, T v) {
    gpu_ref<T>& ref_ = r; GPU_SMEM_ATOMIC(&, cur_ & v);
}
template <class T> static inline T gpu_smem_atomic_or(gpu_ref<T> r, T v) {
    gpu_ref<T>& ref_ = r; GPU_SMEM_ATOMIC(|, cur_ | v);
}
template <class T> static inline T gpu_smem_atomic_xor(gpu_ref<T> r, T v) {
    gpu_ref<T>& ref_ = r; GPU_SMEM_ATOMIC(^, cur_ ^ v);
}
template <class T> static inline T gpu_smem_atomic_min(gpu_ref<T> r, T v) {
    gpu_ref<T>& ref_ = r; GPU_SMEM_ATOMIC(<, v < cur_ ? v : cur_);
}
template <class T> static inline T gpu_smem_atomic_max(gpu_ref<T> r, T v) {
    gpu_ref<T>& ref_ = r; GPU_SMEM_ATOMIC(>, v > cur_ ? v : cur_);
}
"""

_RUNTIME_PRELUDE += r"""
/* ---- block / warp synchronization -------------------------------------------- */
#define GPU_MAX_WARPS 32

static pthread_barrier_t g_bar_launch;   /* master releases a new block      */
static pthread_barrier_t g_bar_done;     /* block finished                   */
static volatile long g_block_index = -2;
static unsigned g_block_acc;             /* __syncthreads_{count,and,or}     */

static unsigned g_blk_cnt, g_blk_gen;
static unsigned g_blk_nt;

/* sense-reversing block barrier; the LAST arriver commits all shared-memory
 * buffers before releasing, so no thread can observe a half-committed state */
static void gpu_syncthreads(void) {
    unsigned gen = __atomic_load_n(&g_blk_gen, __ATOMIC_ACQUIRE);
    if (__atomic_fetch_add(&g_blk_cnt, 1u, __ATOMIC_ACQ_REL) == g_blk_nt - 1u) {
        __atomic_store_n(&g_blk_cnt, 0u, __ATOMIC_RELAXED);
        gpu_smem_commit_all();
        __atomic_fetch_add(&g_blk_gen, 1u, __ATOMIC_RELEASE);
    } else {
        while (__atomic_load_n(&g_blk_gen, __ATOMIC_ACQUIRE) == gen)
            sched_yield();
    }
}
#define __syncthreads() gpu_syncthreads()

static unsigned g_warp_cnt[GPU_MAX_WARPS];
static unsigned g_warp_gen[GPU_MAX_WARPS];
static unsigned g_warp_xchg[GPU_MAX_WARPS][32];
static unsigned g_warp_acc[GPU_MAX_WARPS];
static __thread unsigned t_lane, t_warp;

static void gpu_warp_barrier(void) {
    unsigned w = t_warp;
    unsigned gen = __atomic_load_n(&g_warp_gen[w], __ATOMIC_ACQUIRE);
    if (__atomic_fetch_add(&g_warp_cnt[w], 1u, __ATOMIC_ACQ_REL) == 31u) {
        __atomic_store_n(&g_warp_cnt[w], 0u, __ATOMIC_RELAXED);
        __atomic_fetch_add(&g_warp_gen[w], 1u, __ATOMIC_RELEASE);
    } else {
        while (__atomic_load_n(&g_warp_gen[w], __ATOMIC_ACQUIRE) == gen)
            sched_yield();
    }
}

/* warp votes: three-barrier protocol, so consecutive collectives cannot alias */
static unsigned gpu_ballot_sync(unsigned mask, int pred) {
    (void)mask;   /* full-mask converged warps are guaranteed by generation */
    gpu_warp_barrier();
    if (t_lane == 0u)
        __atomic_store_n(&g_warp_acc[t_warp], 0u, __ATOMIC_RELAXED);
    gpu_warp_barrier();
    if (pred)
        __atomic_fetch_or(&g_warp_acc[t_warp], 1u << t_lane, __ATOMIC_ACQ_REL);
    gpu_warp_barrier();
    return __atomic_load_n(&g_warp_acc[t_warp], __ATOMIC_ACQUIRE);
}
static int gpu_all_sync(unsigned mask, int pred) {
    return gpu_ballot_sync(mask, pred) == 0xFFFFFFFFu;
}
static int gpu_any_sync(unsigned mask, int pred) {
    return gpu_ballot_sync(mask, pred) != 0u;
}

static unsigned gpu_shfl_sync(unsigned mask, unsigned v, int delta, int width) {
    (void)mask;
    unsigned w = (((unsigned)width - 1u) & 31u) + 1u;  /* 32 stays 32 */
    unsigned seg = t_lane & ~(w - 1u);
    gpu_warp_barrier();
    __atomic_store_n(&g_warp_xchg[t_warp][t_lane], v, __ATOMIC_RELAXED);
    gpu_warp_barrier();
    unsigned src = seg + (((unsigned)delta) & (w - 1u));   /* absolute srcLane */
    unsigned r = __atomic_load_n(&g_warp_xchg[t_warp][src], __ATOMIC_RELAXED);
    gpu_warp_barrier();
    return r;
}
static unsigned gpu_shfl_up_sync(unsigned mask, unsigned v, int delta, int width) {
    (void)mask;
    unsigned w = (((unsigned)width - 1u) & 31u) + 1u;  /* 32 stays 32 */
    unsigned seg = t_lane & ~(w - 1u);
    gpu_warp_barrier();
    __atomic_store_n(&g_warp_xchg[t_warp][t_lane], v, __ATOMIC_RELAXED);
    gpu_warp_barrier();
    int src = (int)t_lane - (int)(((unsigned)delta) & (w - 1u));
    unsigned r = (src >= (int)seg)
        ? __atomic_load_n(&g_warp_xchg[t_warp][src], __ATOMIC_RELAXED) : v;
    gpu_warp_barrier();
    return r;
}
static unsigned gpu_shfl_down_sync(unsigned mask, unsigned v, int delta, int width) {
    (void)mask;
    unsigned w = (((unsigned)width - 1u) & 31u) + 1u;  /* 32 stays 32 */
    unsigned seg = t_lane & ~(w - 1u);
    gpu_warp_barrier();
    __atomic_store_n(&g_warp_xchg[t_warp][t_lane], v, __ATOMIC_RELAXED);
    gpu_warp_barrier();
    unsigned src = t_lane + (((unsigned)delta) & (w - 1u));
    unsigned r = (src < seg + w)
        ? __atomic_load_n(&g_warp_xchg[t_warp][src], __ATOMIC_RELAXED) : v;
    gpu_warp_barrier();
    return r;
}
static unsigned gpu_shfl_xor_sync(unsigned mask, unsigned v, int delta, int width) {
    (void)mask;
    unsigned w = (((unsigned)width - 1u) & 31u) + 1u;  /* 32 stays 32 */
    unsigned seg = t_lane & ~(w - 1u);
    gpu_warp_barrier();
    __atomic_store_n(&g_warp_xchg[t_warp][t_lane], v, __ATOMIC_RELAXED);
    gpu_warp_barrier();
    unsigned src = seg + ((t_lane ^ (unsigned)delta) & (w - 1u));
    unsigned r = __atomic_load_n(&g_warp_xchg[t_warp][src], __ATOMIC_RELAXED);
    gpu_warp_barrier();
    return r;
}

/* ---- atomics on ordinary memory (commutative RMW; order-independent) --------- */
static inline unsigned gpu_atomic_add(unsigned* p, unsigned v) { return __atomic_fetch_add(p, v, __ATOMIC_SEQ_CST); }
static inline unsigned gpu_atomic_sub(unsigned* p, unsigned v) { return __atomic_fetch_sub(p, v, __ATOMIC_SEQ_CST); }
static inline unsigned gpu_atomic_and(unsigned* p, unsigned v) { return __atomic_fetch_and(p, v, __ATOMIC_SEQ_CST); }
static inline unsigned gpu_atomic_or(unsigned* p, unsigned v) { return __atomic_fetch_or(p, v, __ATOMIC_SEQ_CST); }
static inline unsigned gpu_atomic_xor(unsigned* p, unsigned v) { return __atomic_fetch_xor(p, v, __ATOMIC_SEQ_CST); }
static inline unsigned gpu_atomic_min(unsigned* p, unsigned v) {
    unsigned old = __atomic_load_n(p, __ATOMIC_SEQ_CST);
    while (v < old) {
        unsigned expected = old;
        if (__atomic_compare_exchange_n(p, &expected, v, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST))
            return expected;
        old = expected;
    }
    return old;
}
static inline unsigned gpu_atomic_max(unsigned* p, unsigned v) {
    unsigned old = __atomic_load_n(p, __ATOMIC_SEQ_CST);
    while (v > old) {
        unsigned expected = old;
        if (__atomic_compare_exchange_n(p, &expected, v, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST))
            return expected;
        old = expected;
    }
    return old;
}

/* atomicX(p, v) overloads: ordinary memory (unsigned*) uses __atomic
 * builtins; shared-memory proxies use the lock-serialized RMW.  Both return
 * the old value, matching CUDA. */
#define GPU_ATOMIC_IMPL(NAME, PTR, REF)                                          \
    static inline unsigned atomic##NAME(unsigned* p, unsigned v)                 \
    { return PTR(p, v); }                                                        \
    template <class T> static inline T atomic##NAME(gpu_ref<T> r, T v)           \
    { return REF(r, v); }                                                        \
    template <class T> static inline T atomic##NAME(gpu_ref_ptr<T> p, T v)       \
    { return REF(p.r, v); }                                                      \
    static inline unsigned atomic##NAME##_block(unsigned* p, unsigned v)         \
    { return PTR(p, v); }                                                        \
    template <class T> static inline T atomic##NAME##_block(gpu_ref<T> r, T v)   \
    { return REF(r, v); }                                                        \
    template <class T> static inline T atomic##NAME##_block(gpu_ref_ptr<T> p, T v) \
    { return REF(p.r, v); }                                                      \
    static inline unsigned atomic##NAME##_system(unsigned* p, unsigned v)        \
    { return PTR(p, v); }                                                        \
    template <class T> static inline T atomic##NAME##_system(gpu_ref<T> r, T v)  \
    { return REF(r, v); }                                                        \
    template <class T> static inline T atomic##NAME##_system(gpu_ref_ptr<T> p, T v) \
    { return REF(p.r, v); }

GPU_ATOMIC_IMPL(Add, gpu_atomic_add, gpu_smem_atomic_add)
GPU_ATOMIC_IMPL(Sub, gpu_atomic_sub, gpu_smem_atomic_sub)
GPU_ATOMIC_IMPL(And, gpu_atomic_and, gpu_smem_atomic_and)
GPU_ATOMIC_IMPL(Or,  gpu_atomic_or,  gpu_smem_atomic_or)
GPU_ATOMIC_IMPL(Xor, gpu_atomic_xor, gpu_smem_atomic_xor)
GPU_ATOMIC_IMPL(Min, gpu_atomic_min, gpu_smem_atomic_min)
GPU_ATOMIC_IMPL(Max, gpu_atomic_max, gpu_smem_atomic_max)

/* ---- counted barriers (__syncthreads_count / _and / _or) ----------------------
 * Thread (0,0,0) resets the accumulator before the first barrier; both
 * barriers commit shared memory, matching __syncthreads. */
static int gpu_syncthreads_impl(int pred) {
    if (t_threadIdx.x == 0u && t_threadIdx.y == 0u && t_threadIdx.z == 0u)
        g_block_acc = 0u;
    gpu_syncthreads();
    if (pred)
        __atomic_fetch_add(&g_block_acc, 1u, __ATOMIC_SEQ_CST);
    gpu_syncthreads();
    return (int)__atomic_load_n(&g_block_acc, __ATOMIC_SEQ_CST);
}
#define __syncthreads_count(p) gpu_syncthreads_impl((p) ? 1 : 0)
#define __syncthreads_and(p)   (gpu_syncthreads_impl((p) ? 1 : 0) == (int)g_blk_nt)
#define __syncthreads_or(p)    (gpu_syncthreads_impl((p) ? 1 : 0) != 0)
"""


# ---------------------------------------------------------------------------
# Emitter
# ---------------------------------------------------------------------------

_SMEM_TYPE_NAMES = {
    "unsigned int": "unsigned", "int": "int",
    "signed char": "signed char", "unsigned char": "unsigned char",
    "short": "short", "unsigned short": "unsigned short",
}


class CpuOracleEmitter(BaseEmitter):
    """Render a Program IR to a self-contained CPU C++ oracle program."""

    def __init__(self, spec: Dict[str, Any], dyn_smem_words: int) -> None:
        super().__init__()
        self.spec = spec
        self.dyn_smem_words = dyn_smem_words

    # -- CUDA API-name mapping -----------------------------------------------

    def signature(self, f: Func) -> str:
        """Kernel/device signatures without CUDA launch attributes.

        ``__device__``/inline hints survive (macro-mapped); ``__global__``,
        ``__launch_bounds__``, ``__maxnreg__`` and ``__cluster_dims__`` are
        GPU resource hints with no CPU meaning and are dropped.
        """
        parts: List[str] = []
        if f.inline_hint:
            parts.append(f.inline_hint)
        if not f.kx_is_kernel() and f.kx_is_device():
            parts.append("__device__")
        parts.append(self.type_name(f.ret))
        params = ", ".join(self._param(p) for p in f.params)
        return " ".join(parts) + f" {f.name}({params})"

    def _warp_vote(self, wv: WarpVote) -> str:
        fn = {"all_sync": "gpu_all_sync", "any_sync": "gpu_any_sync",
              "ballot_sync": "gpu_ballot_sync"}[wv.kind]
        return f"{fn}({self.expr(wv.mask)}, {self.expr(wv.predicate)})"

    def _warp_shuffle(self, ws: WarpShuffle) -> str:
        fn = {"shfl_sync": "gpu_shfl_sync", "shfl_up_sync": "gpu_shfl_up_sync",
              "shfl_down_sync": "gpu_shfl_down_sync",
              "shfl_xor_sync": "gpu_shfl_xor_sync"}[ws.kind]
        return (f"{fn}({self.expr(ws.mask)}, {self.expr(ws.value)}, "
                f"{self.expr(ws.delta)}, {int(ws.width)})")

    # -- statements -------------------------------------------------------------

    @staticmethod
    def _smem_decl_text(d: Declare) -> str:
        """Render a __shared__ declaration as the commit-on-barrier wrapper."""
        t = d.gtype
        # scalar
        if not isinstance(t, TArray):
            return f"static gpu_smem_scalar<{CpuOracleEmitter._cpp_type(t)}> {d.name};"
        dims: List[int] = []
        inner = t
        while isinstance(inner, TArray):
            dims.append(inner.count)
            inner = inner.elem
        base = CpuOracleEmitter._cpp_type(inner)
        if len(dims) == 1:
            return f"static gpu_smem_arr<{base}, {dims[0]}> {d.name};"
        if len(dims) == 2:
            return f"static gpu_smem_arr2<{base}, {dims[0]}, {dims[1]}> {d.name};"
        raise CpuOracleUnsupported(
            f"shared array rank {len(dims)} is not mirrored")

    @staticmethod
    def _cpp_type(t: GType) -> str:
        if isinstance(t, TUInt):
            return "unsigned"
        if isinstance(t, TInt):
            return "int"
        if isinstance(t, TSChar):
            return "signed char"
        if isinstance(t, TChar):
            return "unsigned char"
        if isinstance(t, TShort):
            return "short"
        if isinstance(t, TUShort):
            return "unsigned short"
        return "unsigned"

    def _declare(self, d: Declare) -> None:
        # Shared memory becomes the commit-on-barrier wrapper; everything
        # else matches BaseEmitter (module globals are plain variables).
        if d.space is MemSpace.SHARED:
            self.w.line(self._smem_decl_text(d))
            return
        if d.space is MemSpace.CONSTANT:
            d = Declare(name=d.name, gtype=d.gtype, space=MemSpace.GENERIC,
                        init=d.init, quals=list(d.quals))
        super()._declare(d)

    def _raw_stmt(self, s: RawCode) -> None:
        text = _strip_c_comments(s.text)
        text = _EMPTY_ASM_RE.sub('asm volatile("" ::: "memory");', text)
        text = self._rewrite_raw_smem(text)
        for ln in text.split("\n"):
            ln = ln.strip()
            if ln:
                self.w.line(ln)

    def _rewrite_raw_smem(self, text: str) -> str:
        def decl_repl(m: "re.Match") -> str:
            ty = _SMEM_TYPE_NAMES[m.group(1)]
            name, dim1, dim2 = m.group(2), m.group(3), m.group(4)
            if dim2:
                return f"static gpu_smem_arr2<{ty}, {dim1}, {dim2}> {name};"
            if dim1:
                return f"static gpu_smem_arr<{ty}, {dim1}> {name};"
            return f"static gpu_smem_scalar<{ty}> {name};"

        text = _SMEM_DYN_RE.sub(
            lambda m: f"static gpu_smem_dyn {m.group(1)}({self.dyn_smem_words});",
            text)
        return _SMEM_RE.sub(decl_repl, text)

    def stmt(self, s: Stmt) -> None:
        if isinstance(s, RawCode):
            self._raw_stmt(s)
            return
        if isinstance(s, UnrollHint):
            # CUDA unroll pragmas are unknown to GCC; drop them.
            return
        super().stmt(s)

    # -- whole-file assembly ------------------------------------------------------

    def rendersrc(self, program: Program) -> str:
        cfg = program.launch
        kernel = program.fetch_kernel()
        if kernel is None:
            raise CpuOracleUnsupported("program has no kernel")
        w = SourceWriter()
        self.w = w
        spec = self.spec

        w.raw(_RUNTIME_PRELUDE)
        w.blank()
        # Safe-op + divergence macros shared verbatim with the GPU backends.
        write_safe_macros(w)
        write_diverge_macros(w)
        w.raw(f"static unsigned int g_dyn_smem_words = {max(1, self.dyn_smem_words)};")
        w.blank()

        # Deterministic harness buffers (contract from cuda_driver_spec).
        w.line(f"static unsigned int *g_input;   /* in_size = {spec['in_size']} words */")
        w.line(f"static unsigned int *g_output;  /* buf_size = {spec['buf_size']} words */")
        w.line(f"static int g_n = {int(spec['n'])};")
        for i, arg in enumerate(spec["args"]):
            if arg.get("kind") == "scalar":
                data = bytes.fromhex(arg.get("data_hex", ""))
                body = ", ".join(f"0x{b:02x}" for b in data)
                w.line(f"static const unsigned char k_arg_{i}[{len(data)}] = {{{body}}};")
        w.blank()

        for d in program.decls:
            if isinstance(d, StructDecl):
                self.struct(d)
                w.blank()
        top_level_decls = [d for d in program.decls if isinstance(d, Declare)]
        for d in top_level_decls:
            self.stmt(d)
        if top_level_decls:
            w.blank()
        if cfg.needs_constant_mem:
            w.line("unsigned int _cmem[256];")
            w.blank()
        for d in program.decls:
            if isinstance(d, Func) and not d.kx_is_kernel():
                self.function(d)
                w.blank()
        self.function(kernel)
        w.blank()
        self._emit_harness(program, kernel)
        return w.text()

    # -- deterministic harness (mirrors native/cuda_runner.cpp) ---------------------

    @staticmethod
    def _input_word_fn(dist: str) -> str:
        body = {
            "zeros": "return 0;",
            "small": "return i % 17u;",
            "random": "return (i * 1103515245u + 12345u) ^ (i >> 3);",
            "adversarial": ("return ((i & 1u) ? 0xFFFFFFFFu : 0u)"
                            " ^ ((i & 2u) ? 0x80000000u : 0u);"),
        }.get(dist, "return i * 2654435761u + 12345u;")
        return ("static unsigned int gpu_input_word(unsigned int i) {\n"
                f"    {body}\n"
                "}\n")

    def _emit_harness(self, program: Program, kernel: Func) -> None:
        cfg = program.launch
        spec = self.spec
        grid = [int(x) for x in spec["grid"]]
        block = [int(x) for x in spec["block"]]
        in_size = int(spec["in_size"])
        buf_size = int(spec["buf_size"])
        dist = str(spec.get("input_distribution", "hash"))
        gx, gy, gz = grid
        bx, by, bz = block
        nt = bx * by * bz
        alarm_s = int(spec.get("alarm_seconds", 600))

        w = self.w
        w.raw("// ---- deterministic input/checksum harness (runner contract) ----")
        w.raw(self._input_word_fn(dist))
        w.raw("""static void gpu_apply_input_salt(unsigned int *input, size_t n,
                            unsigned long long salt) {
    if (salt == 0ULL) return;
    for (size_t i = 0; i < n; ++i) {
        unsigned long long z =
            salt + 0x9e3779b97f4a7c15ULL * ((unsigned long long)(unsigned int)i + 1ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z ^= z >> 31;
        unsigned int salt_word = (unsigned int)(z ^ (z >> 32));
        input[i] ^= salt_word ? salt_word : 0xa5a5a5a5u;
    }
}
""")

        args: List[str] = []
        for i, arg in enumerate(spec["args"]):
            kind = arg.get("kind")
            if kind == "output":
                args.append("g_output")
            elif kind == "input":
                args.append("g_input")
            elif kind == "n":
                args.append("g_n")
            elif kind == "scalar":
                args.append(f"s{i}")
            else:
                raise CpuOracleUnsupported(f"spec arg kind {kind!r} is not mirrored")
        decls = "\n".join(
            f"    {arg['type']} s{i}; memcpy(&s{i}, k_arg_{i}, sizeof(s{i}));"
            for i, arg in enumerate(spec["args"]) if arg.get("kind") == "scalar")
        call = f"{kernel.name}({', '.join(args)});"
        w.raw(f"static void gpu_call_kernel(void) {{\n{decls}\n    {call}\n}}\n")

        w.raw(f"""static void *gpu_thread_main(void *arg) {{
    long t = (long)(intptr_t)arg;
    unsigned int tx = (unsigned int)(t % {bx});
    unsigned int ty = (unsigned int)((t / {bx}) % {by});
    unsigned int tz = (unsigned int)(t / ({bx} * {by}));
    unsigned int lin = tx + ty * {bx}u + tz * {bx}u * {by}u;
    g_my_tid = lin;
    t_lane = lin & 31u;
    t_warp = lin >> 5;
    for (;;) {{
        pthread_barrier_wait(&g_bar_launch);
        long blk = g_block_index;
        if (blk < 0) break;
        t_blockIdx.x = (unsigned int)(blk % {gx});
        t_blockIdx.y = (unsigned int)((blk / {gx}) % {gy});
        t_blockIdx.z = (unsigned int)(blk / ({gx} * {gy}));
        t_threadIdx.x = tx; t_threadIdx.y = ty; t_threadIdx.z = tz;
        gpu_call_kernel();
        pthread_barrier_wait(&g_bar_done);
    }}
    return NULL;
}}

static void gpu_alarm_exit(int sig) {{ (void)sig; _Exit(9); }}

int main(void) {{
    signal(SIGALRM, gpu_alarm_exit);
    alarm({alarm_s});

    gpu_gridDim.x = {gx}; gpu_gridDim.y = {gy}; gpu_gridDim.z = {gz};
    gpu_blockDim.x = {bx}; gpu_blockDim.y = {by}; gpu_blockDim.z = {bz};
    g_blk_nt = {nt};

    g_input = (unsigned int *)malloc((size_t){in_size} * sizeof(unsigned int));
    g_output = (unsigned int *)calloc((size_t){buf_size}, sizeof(unsigned int));
    if (!g_input || !g_output) {{ fprintf(stderr, "malloc failed\\n"); return 1; }}
    for (long i = 0; i < {in_size}; i++)
        g_input[i] = gpu_input_word((unsigned int)i);
    {{
        const char *salt_text = getenv("CUDA_STRESS_INPUT_SALT");
        unsigned long long salt = (salt_text && *salt_text)
            ? strtoull(salt_text, NULL, 0) : 0ULL;
        gpu_apply_input_salt(g_input, (size_t){in_size}, salt);
    }}
""")

        # module-global initialization in runner order (zero / cmem_u32)
        for g in spec.get("globals", []):
            name = str(g["name"])
            init = str(g.get("init", "zero"))
            if name == "_cmem" and cfg.needs_constant_mem:
                w.raw("    for (int i = 0; i < 256; i++) "
                      "_cmem[i] = (unsigned int)i * 2654435761u + 40503u;")
                continue
            decl = next((d for d in program.decls
                         if isinstance(d, Declare) and d.name == name), None)
            if decl is None or not isinstance(decl.gtype, TArray):
                continue
            total = 1
            inner = decl.gtype
            while isinstance(inner, TArray):
                total *= inner.count
                inner = inner.elem
            if init == "cmem_u32":
                w.raw(f"    for (long i = 0; i < {total}; i++) "
                      f"{name}[i] = (unsigned int)i * 2654435761u + 40503u;")
            else:
                w.raw(f"    memset({name}, 0, sizeof({name}));")

        w.raw(f"""    pthread_barrier_init(&g_bar_launch, NULL, {nt} + 1);
    pthread_barrier_init(&g_bar_done, NULL, {nt} + 1);

    pthread_t th[{nt}];
    for (long t = 0; t < {nt}; t++)
        if (pthread_create(&th[t], NULL, gpu_thread_main, (void *)(intptr_t)t)) {{
            fprintf(stderr, "pthread_create failed\\n"); return 1;
        }}

    const long total_blocks = (long){gx} * {gy} * {gz};
    for (long blk = 0; blk < total_blocks; blk++) {{
        g_block_index = blk;
        pthread_barrier_wait(&g_bar_launch);
        pthread_barrier_wait(&g_bar_done);
    }}
    g_block_index = -1;
    pthread_barrier_wait(&g_bar_launch);
    for (long t = 0; t < {nt}; t++)
        pthread_join(th[t], NULL);

    unsigned long long hash = 42;
    for (long i = 0; i < {buf_size}; i++) {{
        unsigned long long val = g_output[i];
        hash ^= val + 0x9e3779b9ULL + (hash << 6) + (hash >> 2);
    }}
    printf("checksum=%llu\\n", (unsigned long long)hash);
    return 0;
}}
""")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _dyn_smem_words(program: Program) -> int:
    for text in _iter_raw_text(program):
        if "extern __shared__" in text:
            return max(1024, program.launch.threads_per_block)
    return 0


def translate_cpu_oracle(program: Program) -> str:
    """Emit the CPU oracle C++ program, or raise CpuOracleUnsupported."""
    problems = scan_program(program)
    if problems:
        raise CpuOracleUnsupported("; ".join(problems))

    from generator_ir.emit.host_harness import cuda_driver_spec
    kernel = program.fetch_kernel()
    if kernel is None:
        raise CpuOracleUnsupported("program has no kernel")
    spec = cuda_driver_spec(program, kernel)
    for arg in spec["args"]:
        if arg.get("kind") == "output_fp":
            raise CpuOracleUnsupported("float output buffer is not mirrored")
    spec = dict(spec)
    spec["threads"] = (
        int(spec["block"][0]) * int(spec["block"][1]) * int(spec["block"][2]))
    spec["alarm_seconds"] = 600
    return CpuOracleEmitter(spec, _dyn_smem_words(program)).rendersrc(program)


def run_cpu_oracle_source(source: str, workdir: Optional[Path] = None,
                          cc: str = "g++", timeout: float = 300.0,
                          input_salt: Optional[int] = None) -> int:
    """Compile + execute an oracle program; return its u64 checksum.

    Raises CpuOracleUnsupported on compile failure and RuntimeError on
    execution failure/timeout, so the strategy can fall back cleanly.
    """
    tmp: Optional[tempfile.TemporaryDirectory] = None
    if workdir is None:
        tmp = tempfile.TemporaryDirectory(prefix="cpu_oracle_")
        workdir = Path(tmp.name)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    src = workdir / "oracle_cpu.cc"
    exe = workdir / "oracle_cpu.exe"
    src.write_text(source)
    try:
        proc = subprocess.run(
            [cc, "-O2", "-std=gnu++17", "-pthread",
             str(src), "-o", str(exe), "-lm"],
            capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            tail = (proc.stderr or "")[-800:]
            raise CpuOracleUnsupported(f"compile failed: {tail.strip()}")
        env = dict(os.environ)
        if input_salt is None:
            env.pop("CUDA_STRESS_INPUT_SALT", None)
        else:
            env["CUDA_STRESS_INPUT_SALT"] = str(int(input_salt) & 0xFFFFFFFFFFFFFFFF)
        run = subprocess.run([str(exe)], capture_output=True, text=True,
                             timeout=timeout, env=env)
        if run.returncode == 9:
            raise RuntimeError("cpu oracle watchdog alarm fired")
        if run.returncode != 0:
            raise RuntimeError(
                f"cpu oracle exited {run.returncode}: {(run.stderr or '')[-400:]}")
        m = re.search(r"^checksum=([0-9]+)$", run.stdout, re.M)
        if not m:
            raise RuntimeError(
                f"cpu oracle printed no checksum: {(run.stdout or '')[-400:]}")
        return int(m.group(1))
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("cpu oracle timed out") from exc
    finally:
        if tmp is not None:
            tmp.cleanup()


def cpu_oracle_checksums(program: Program, workdir: Optional[Path] = None,
                         timeout: float = 300.0) -> Dict[str, int]:
    """Compute the (unsalted, salted) oracle checksums for one program."""
    source = translate_cpu_oracle(program)
    base = run_cpu_oracle_source(source, workdir=workdir, timeout=timeout)
    salted = run_cpu_oracle_source(source, workdir=workdir, timeout=timeout,
                                   input_salt=1)
    return {"checksum": base, "salted_checksum": salted}
