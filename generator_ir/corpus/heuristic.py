"""Corpus scoring: property-based prioritization without coverage feedback.

Dimensions (weighted sum, each 0..1):
  register pressure 40%, divergence potential 25%, memory diversity 20%,
  mutation applicability 10%, performance sensitivity 5%.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from generator_ir.ir.access_pattern import AccessPatternAnalyzer
from generator_ir.ir.typedecl import MemSpace, TArray, TPointer
from generator_ir.ir.nodes import (
    Atomic, WarpShuffle, ThreadBuiltin, Fma, Subscript, Var, BinOp, Call,
)
from generator_ir.ir.statements import (
    For, If, While, DoWhile, Switch, Barrier, Fence, IncDecStmt, ExprStmt,
    Declare, UnrollHint, Assign, RawCode, Block, Stmt,
)
from generator_ir.ir.program import Func, Program
from generator_ir.ir.traversal import (
    collect, kernel_of, for_loops, declarations, assignments, subscripts,
    atomics, barriers, if_statements, mutable_for_loops,
)

W_REGISTER = 0.40
W_DIVERGENCE = 0.25
W_MEMORY = 0.20
W_MUTATION = 0.10
W_SENSITIVITY = 0.05


@dataclass
class CorpusScore:
    register_pressure: float = 0.0
    divergence_potential: float = 0.0
    memory_diversity: float = 0.0
    mutation_applicability: float = 0.0
    performance_sensitivity: float = 0.0

    @property
    def total(self) -> float:
        return (W_REGISTER * self.register_pressure
                + W_DIVERGENCE * self.divergence_potential
                + W_MEMORY * self.memory_diversity
                + W_MUTATION * self.mutation_applicability
                + W_SENSITIVITY * self.performance_sensitivity)

    def summary(self) -> str:
        return (f"total={self.total:.3f} "
                f"(reg={self.register_pressure:.2f} "
                f"div={self.divergence_potential:.2f} "
                f"mem={self.memory_diversity:.2f} "
                f"mut={self.mutation_applicability:.2f} "
                f"sens={self.performance_sensitivity:.2f})")


@dataclass
class CorpusFeatures:
    tags: List[str] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)

    def kx_as_set(self) -> Set[str]:
        return set(self.tags)


_INTRINSIC_FAMILIES: Dict[str, Set[str]] = {
    "half_precision": {
        "__hadd", "__hsub", "__hmul", "__hdiv", "__hfma",
        "__float2half_rn", "__half2float", "__float2half",
        "__hadd2", "__hmul2", "__hfma2", "__habs", "__hneg",
    },
    "rounding_mode": {
        "__fadd_rn", "__fadd_rd", "__fadd_ru", "__fadd_rz",
        "__fmul_rn", "__fmul_rd", "__fmul_ru", "__fmul_rz",
        "__fmaf_rn", "__fmaf_rd", "__fmaf_ru", "__fmaf_rz",
        "__fdiv_rn", "__fdiv_rd", "__fdiv_ru", "__fdiv_rz",
        "__fsqrt_rn", "__fsqrt_rd", "__fsqrt_ru", "__fsqrt_rz",
        "__dadd_rn", "__dadd_rd", "__dadd_ru", "__dadd_rz",
        "__dmul_rn", "__dmul_rd", "__dmul_ru", "__dmul_rz",
    },
    "bit_ops": {
        "__popc", "__clz", "__ffs", "__brev", "__byte_perm",
        "__popcll", "__clzll", "__ffsll", "__brevll",
    },
    "wide_mul": {
        "__mulhi", "__umulhi", "__mul64hi", "__umul64hi",
        "__mul24", "__umul24", "__sad", "__usad",
        "__dp4a", "__dp2a_lo", "__dp2a_hi",
    },
    "warp_ops": {
        "__shfl_sync", "__shfl_up_sync", "__shfl_down_sync", "__shfl_xor_sync",
        "__ballot_sync", "__all_sync", "__any_sync", "__match_any_sync",
        "__reduce_add_sync", "__reduce_min_sync", "__reduce_max_sync",
    },
    "cache_ops": {"__ldg", "__ldcg", "__ldca", "__ldcs", "__ldlu"},
    "fast_math": {
        "__sinf", "__cosf", "__tanf", "__expf", "__logf",
        "__log2f", "__powf", "__rsqrtf", "__saturatef",
    },
}


def _kx_type_family(t) -> str:
    if t is None:
        return "unknown"
    if isinstance(t, TArray):
        return f"array:{_kx_type_family(t.elem)}"
    if isinstance(t, TPointer):
        return f"ptr:{_kx_type_family(t.base)}"
    name = type(t).__name__.replace("T", "", 1).lower() or "other"
    if getattr(t, "is_fp", False):
        return f"float:{name}"
    if getattr(t, "is_int", False):
        return f"int:{name}"
    return name


def pull_features(program: Program, template_name: str = "") -> CorpusFeatures:
    kernel = kernel_of(program)
    if kernel is None or kernel.body is None:
        return CorpusFeatures()

    tags: Set[str] = set()
    counts: Dict[str, int] = {}

    def bump(tag: str, value: int = 1) -> None:
        counts[tag] = counts.get(tag, 0) + value
        tags.add(tag)

    stmt_kinds = {
        "for": For, "if": If, "while": While, "do_while": DoWhile,
        "switch": Switch, "sync": Barrier, "fence": Fence,
        "incr_decr": IncDecStmt, "expr": ExprStmt, "assign": Assign,
        "kx_declare": Declare, "unroll": UnrollHint,
    }
    for name, cls in stmt_kinds.items():
        n = len(collect(kernel, lambda node, c=cls: isinstance(node, c)))
        if n:
            bump(f"stmt:{name}", n)

    expr_kinds = {
        "atomic": Atomic, "warp_shuffle": WarpShuffle,
        "thread_id": ThreadBuiltin, "fma": Fma, "binary": BinOp,
        "call": Call, "subscript": Subscript,
    }
    for name, cls in expr_kinds.items():
        n = len(collect(kernel, lambda node, c=cls: isinstance(node, c)))
        if n:
            bump(f"expr:{name}", n)

    for call in collect(kernel, lambda n: isinstance(n, Call)):
        for cat, names in _INTRINSIC_FAMILIES.items():
            if call.fn in names:
                bump(f"intrinsic:{cat}")
                break

    analyzer = AccessPatternAnalyzer(kernel)
    for pat, cnt in analyzer.pattern_summary().items():
        bump(f"mem:{pat.lower()}", cnt)
    coalesced_arrays = sum(1 for name in {r.array_name for r in analyzer.records()
                                          if r.array_name}
                           if analyzer.is_coalesced(name))
    if coalesced_arrays:
        bump("mem:coalesced_array", coalesced_arrays)

    raws = collect(kernel.body, lambda n: isinstance(n, RawCode))
    if raws:
        bump("stmt:raw", len(raws))

    shared_bytes = 0
    for d in declarations(kernel):
        bump(f"space:{d.space.name.lower()}")
        bump(f"type:{_kx_type_family(d.gtype)}")
        if d.space is MemSpace.SHARED and isinstance(d.gtype, TArray):
            shared_bytes += d.gtype.width
    if shared_bytes:
        bump("mem:shared_bytes", max(1, shared_bytes // 256))

    cfg = program.launch
    bump(f"launch:block_{cfg.threads_per_block}")
    bump(f"launch:grid_{cfg.block_count}")
    if cfg.body_repeat > 1:
        bump(f"launch:repeat_{cfg.body_repeat}", cfg.body_repeat)
    if template_name:
        bump(f"template:{template_name}")

    return CorpusFeatures(tags=sorted(tags), counts=counts)


# ---------------------------------------------------------------------------
# scoring dimensions
# ---------------------------------------------------------------------------

def _kx_max_loop_depth(node, depth: int) -> int:
    if node is None:
        return depth
    if isinstance(node, (For, While, DoWhile)):
        return _kx_max_loop_depth(node.body, depth + 1)
    if isinstance(node, Block):
        return max((_kx_max_loop_depth(s, depth) for s in node.stmts), default=depth)
    if isinstance(node, If):
        d1 = _kx_max_loop_depth(node.then, depth)
        d2 = _kx_max_loop_depth(node.otherwise, depth) if node.otherwise else depth
        return max(d1, d2)
    return depth


def _rate_register_pressure(kernel: Func, perf_data: Optional[Dict]) -> float:
    rate = 0.0
    decls = declarations(kernel)
    n_vars = len([d for d in decls
                  if d.space in (MemSpace.GENERIC, MemSpace.REGISTER)])
    rate += min(1.0, n_vars / 20.0) * 0.3
    rate += min(1.0, _kx_max_loop_depth(kernel.body, 0) / 4.0) * 0.2
    fma_count = len(collect(kernel, lambda n: isinstance(n, Fma)))
    rate += min(1.0, fma_count / 5.0) * 0.2
    if barriers(kernel):
        rate += 0.15
    if perf_data and perf_data.get("register_count"):
        reg = perf_data["register_count"]
        if reg > 96:
            rate += 0.15
        elif reg > 64:
            rate += 0.10
        elif reg > 32:
            rate += 0.05
    return min(1.0, rate)


def _rate_divergence(kernel: Func, features: CorpusFeatures) -> float:
    rate = 0.0
    ifs = if_statements(kernel)
    tid_in_cond = sum(1 for i in ifs
                      if collect(i.cond, lambda n: isinstance(n, ThreadBuiltin)))
    rate += min(1.0, tid_in_cond / 5.0) * 0.22
    total_stmts = len(collect(kernel, lambda n: isinstance(n, Stmt)))
    branch_ratio = len(ifs) / max(1, total_stmts)
    rate += min(1.0, branch_ratio * 10) * 0.15
    warp_ops = len(collect(kernel, lambda n: isinstance(n, WarpShuffle)))
    rate += min(1.0, warp_ops / 3.0) * 0.15
    switches = len(collect(kernel, lambda n: isinstance(n, Switch)))
    whiles = (len(collect(kernel, lambda n: isinstance(n, While)))
              + len(collect(kernel, lambda n: isinstance(n, DoWhile))))
    rate += min(1.0, switches / 2.0) * 0.08
    rate += min(1.0, whiles / 3.0) * 0.04
    rate += min(1.0, len(atomics(kernel)) / 3.0) * 0.12
    fences = len(collect(kernel, lambda n: isinstance(n, Fence)))
    rate += min(1.0, fences / 3.0) * 0.04
    tids = len(collect(kernel, lambda n: isinstance(n, ThreadBuiltin)))
    rate += min(1.0, tids / 10.0) * 0.10
    intrinsic_cats = sum(1 for t in features.tags if t.startswith("intrinsic:"))
    rate += min(0.10, intrinsic_cats * 0.02)
    return min(1.0, rate)


def _rate_memory(kernel: Func, program: Program) -> float:
    rate = 0.0
    spaces: Set[MemSpace] = set()
    shared_bytes = 0
    for d in declarations(kernel):
        spaces.add(d.space)
        if d.space is MemSpace.SHARED and isinstance(d.gtype, TArray):
            shared_bytes += d.gtype.width
    rate += min(1.0, len(spaces) / 3.0) * 0.3
    rate += min(1.0, shared_bytes / (32 * 1024)) * 0.25
    summary = AccessPatternAnalyzer(kernel).pattern_summary()
    rate += min(1.0, len(summary) / 4.0) * 0.2          # access-pattern diversity
    uncoalesced = summary.get("STRIDED", 0) + summary.get("RANDOM", 0)
    rate += min(1.0, uncoalesced / 4.0) * 0.05          # uncoalesced stress
    rate += min(1.0, len(barriers(kernel)) / 4.0) * 0.25
    return min(1.0, rate)


def _rate_mutation_applicability(kernel: Func) -> float:
    applicable = 0
    if declarations(kernel):
        applicable += 1
    if subscripts(kernel):
        applicable += 1
    if assignments(kernel):
        applicable += 1
    if mutable_for_loops(kernel):
        applicable += 1
    if collect(kernel, lambda n: isinstance(n, Fma)):
        applicable += 1
    if atomics(kernel):
        applicable += 1
    if collect(kernel, lambda n: isinstance(n, (Switch, Barrier, Fence))):
        applicable += 1
    return min(1.0, applicable / 7.0)


def _rate_sensitivity(perf_data: Optional[Dict]) -> float:
    if not perf_data:
        return 0.5
    old = perf_data.get("old_o3_time_us", 0)
    new = perf_data.get("new_o3_time_us", 0)
    if old <= 0 or new <= 0:
        return 0.5
    ratio = max(old, new) / min(old, new)
    return min(1.0, max(0.0, ratio - 1.0))


def rate_program(program: Program,
                  perf_data: Optional[Dict] = None) -> CorpusScore:
    kernel = kernel_of(program)
    if kernel is None or kernel.body is None:
        return CorpusScore()
    features = pull_features(program)
    return CorpusScore(
        register_pressure=_rate_register_pressure(kernel, perf_data),
        divergence_potential=_rate_divergence(kernel, features),
        memory_diversity=_rate_memory(kernel, program),
        mutation_applicability=_rate_mutation_applicability(kernel),
        performance_sensitivity=_rate_sensitivity(perf_data),
    )
