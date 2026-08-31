"""E0 diagnostic corpus study for the generator_ir generic baseline.

The experiment asks whether template-free random programs accidentally express
the live-ranges-across-synchronization target used in the paper motivation.
It measures source-level trigger primitives, their full conjunction, normalized
IR diversity, and an optional PTX artifact signal.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import dataclasses
import enum
import hashlib
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generator.base import default_registry  # noqa: E402
from generator_ir.corpus.heuristic import pull_features  # noqa: E402
from generator_ir.ir.nodes import (  # noqa: E402
    AddrOf,
    Atomic,
    BinOp,
    Call,
    Cast,
    Deref,
    Expr,
    Fma,
    Lit,
    LogicOp,
    Member,
    RelOp,
    SafeBin,
    Select,
    Subscript,
    ThreadBuiltin,
    UnOp,
    Var,
    WarpShuffle,
    WarpVote,
)
from generator_ir.ir.program import Func, LaunchConfig, Program, StructDecl  # noqa: E402
from generator_ir.ir.statements import (  # noqa: E402
    Assign,
    Barrier,
    Block,
    Break,
    Case,
    CompoundAssign,
    Continue,
    Declare,
    DoWhile,
    ExprStmt,
    Fence,
    For,
    If,
    IncDecStmt,
    RawCode,
    Return,
    Stmt,
    Switch,
    UnrollHint,
    While,
)
from generator_ir.ir.typedecl import (  # noqa: E402
    GType,
    MemSpace,
    Qual,
    TArray,
    TChar,
    TInt,
    TLong,
    TPointer,
    TQualified,
    TSChar,
    TShort,
    TStruct,
    TUShort,
    TULong,
    TUInt,
    TVec,
)
from generator_ir.ir.traversal import collect, kernel_of  # noqa: E402


TARGET_NAME = "live_ranges_across_barrier"
DEFAULT_LIVE_THRESHOLD = 4
ARM_C_DEFAULT_TEMPLATES = (
    "sync_span_live",
    "cf_switch_tree",
    "smem_bank_stress",
    "isa_stress",
)
ARM_B_DEFAULT_TEMPLATES = (
    "rawllm_sync_span_live",
    "rawllm_cf_switch_tree",
    "rawllm_smem_bank_stress",
    "rawllm_isa_stress",
)
ARM_C_TEMPLATE_TARGETS = {
    "sync_span_live": ("sync_live_l4", "shared_write_barrier_read"),
    "cf_switch_tree": ("thread_dependent_switch_ge3", "shared_write_barrier_read"),
    "smem_bank_stress": ("shared_write_barrier_read",),
    "isa_stress": ("byte_perm_call", "mulhi_call"),
    # Additional PassSpec-like families in the catalog.  They are not part of
    # the default Arm-C suite because the current source checker suite does not
    # yet include their artifact-specific target definitions, but callers can
    # include them explicitly with --templates for exploratory runs.
    "perm_round": ("shared_write_barrier_read",),
    "reg_spill_probe": ("sync_live_l4",),
    "vec_load_probe": ("shared_write_barrier_read",),
}
ARM_B_TEMPLATE_TARGETS = {
    "rawllm_sync_span_live": ("sync_live_l4", "shared_write_barrier_read"),
    "rawllm_cf_switch_tree": ("thread_dependent_switch_ge3", "shared_write_barrier_read"),
    "rawllm_smem_bank_stress": ("shared_write_barrier_read",),
    "rawllm_isa_stress": ("byte_perm_call", "mulhi_call"),
}
ALL_TEMPLATE_TARGETS = {**ARM_C_TEMPLATE_TARGETS, **ARM_B_TEMPLATE_TARGETS}
RESERVED_VALUE_NAMES = {
    "output",
    "input",
    "n",
    "tid",
    "_rep",
    "_ci",
    "_fv",
    "_fv_sink",
    "_ov_sink",
}
INLINE_ASM_RE = re.compile(r"\b(?:asm|__asm__)\b\s*(?:volatile\b\s*)?\(")


def _default_templates_for_mode(mode: str) -> Tuple[str, ...]:
    if mode == "arm-b":
        return ARM_B_DEFAULT_TEMPLATES
    if mode == "arm-c":
        return ARM_C_DEFAULT_TEMPLATES
    return ()


def _template_targets_for_mode(mode: str) -> Dict[str, Tuple[str, ...]]:
    if mode == "arm-b":
        return ARM_B_TEMPLATE_TARGETS
    if mode == "arm-c":
        return ARM_C_TEMPLATE_TARGETS
    return ALL_TEMPLATE_TARGETS


def _experiment_name(mode: str) -> str:
    if mode == "arm-b":
        return "RQ1-ArmB"
    if mode == "arm-c":
        return "RQ1-ArmC"
    return "E0"


def _source_has_inline_asm(source: str) -> bool:
    return bool(INLINE_ASM_RE.search(source or ""))


def _simple_signal(full: bool, primitives: Dict[str, bool], **metrics: Any) -> Dict[str, Any]:
    return {
        "full_conjunction": bool(full),
        "primitives": dict(primitives),
        "metrics": metrics,
    }


@dataclass
class Event:
    kind: str
    uses: Set[str] = field(default_factory=set)
    defs: Set[str] = field(default_factory=set)
    dep_defs: Set[str] = field(default_factory=set)
    output_rhs: Set[str] = field(default_factory=set)
    text: str = ""


@dataclass
class SourceSignal:
    input_dependent_def_count: int
    barrier_count: int
    use_after_barrier_count: int
    observable_sink_count: int
    max_cross_barrier_observable_live_defs: int
    full_conjunction: bool
    primitives: Dict[str, bool]
    best_barrier_defs: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class ArtifactSignal:
    compiled: bool
    artifact_hit: bool
    barrier_count: int = 0
    max_live_regs_across_barrier: int = 0
    ptx_sha256: str = ""
    nvcc_version: str = ""
    error: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _is_scalar_type(t: Any) -> bool:
    return isinstance(t, GType) and not isinstance(t, (TPointer, TArray, TStruct, TVec))


def _expr_vars(expr: Optional[Expr]) -> Set[str]:
    if expr is None:
        return set()
    return {v.name for v in collect(expr, lambda node: isinstance(node, Var))}


def _expr_has_source(expr: Optional[Expr], scalar_params: Set[str]) -> bool:
    if expr is None:
        return False
    if collect(expr, lambda node: isinstance(node, ThreadBuiltin)):
        return True
    for sub in collect(expr, lambda node: isinstance(node, Subscript)):
        arr = getattr(sub, "array", None)
        if isinstance(arr, Var) and arr.name == "input":
            return True
    return bool(_expr_vars(expr) & scalar_params)


def _is_output_store(expr: Optional[Expr]) -> bool:
    if not isinstance(expr, Subscript):
        return False
    arr = getattr(expr, "array", None)
    return isinstance(arr, Var) and arr.name == "output"


def _target_name(expr: Optional[Expr]) -> Optional[str]:
    return expr.name if isinstance(expr, Var) else None


def _stmt_type_decls(stmt: Stmt, type_map: Dict[str, Any]) -> None:
    if isinstance(stmt, Declare):
        type_map[stmt.name] = stmt.gtype
    elif isinstance(stmt, For):
        # generator_ir for-loops always use int induction variables.
        from generator_ir.ir.typedecl import T

        type_map[stmt.var] = T.i32


def _flatten_stmt(stmt: Stmt, events: List[Event], type_map: Dict[str, Any]) -> None:
    _stmt_type_decls(stmt, type_map)
    if isinstance(stmt, Declare):
        uses = _expr_vars(stmt.init)
        defs = {stmt.name} if _is_scalar_type(stmt.gtype) else set()
        events.append(Event(kind="def", uses=uses, defs=defs))
    elif isinstance(stmt, Assign):
        uses = _expr_vars(stmt.value) | (_expr_vars(stmt.target) - {_target_name(stmt.target) or ""})
        rhs = _expr_vars(stmt.value)
        if _is_output_store(stmt.target):
            events.append(Event(kind="output", uses=uses, output_rhs=rhs))
        else:
            name = _target_name(stmt.target)
            defs = {name} if name and _is_scalar_type(type_map.get(name)) else set()
            events.append(Event(kind="def", uses=uses, defs=defs))
    elif isinstance(stmt, CompoundAssign):
        name = _target_name(stmt.target)
        target_uses = _expr_vars(stmt.target)
        uses = _expr_vars(stmt.value) | target_uses
        if _is_output_store(stmt.target):
            events.append(Event(kind="output", uses=uses, output_rhs=uses))
        else:
            defs = {name} if name and _is_scalar_type(type_map.get(name)) else set()
            events.append(Event(kind="def", uses=uses, defs=defs))
    elif isinstance(stmt, IncDecStmt):
        name = _target_name(stmt.target)
        uses = _expr_vars(stmt.target)
        defs = {name} if name and _is_scalar_type(type_map.get(name)) else set()
        events.append(Event(kind="def", uses=uses, defs=defs))
    elif isinstance(stmt, ExprStmt):
        events.append(Event(kind="use", uses=_expr_vars(stmt.expr)))
    elif isinstance(stmt, If):
        events.append(Event(kind="use", uses=_expr_vars(stmt.cond)))
        _flatten_block(stmt.then, events, type_map.copy())
        if stmt.otherwise is not None:
            _flatten_block(stmt.otherwise, events, type_map.copy())
    elif isinstance(stmt, For):
        events.append(Event(kind="use", uses=_expr_vars(stmt.start) | _expr_vars(stmt.end) | _expr_vars(stmt.step)))
        _flatten_block(stmt.body, events, type_map.copy())
    elif isinstance(stmt, While):
        events.append(Event(kind="use", uses=_expr_vars(stmt.cond)))
        _flatten_block(stmt.body, events, type_map.copy())
        events.append(Event(kind="use", uses=_expr_vars(stmt.cond)))
    elif isinstance(stmt, DoWhile):
        _flatten_block(stmt.body, events, type_map.copy())
        events.append(Event(kind="use", uses=_expr_vars(stmt.cond)))
    elif isinstance(stmt, Switch):
        events.append(Event(kind="use", uses=_expr_vars(stmt.expr)))
        for case in stmt.cases:
            _flatten_stmt(case, events, type_map.copy())
    elif isinstance(stmt, Case):
        events.append(Event(kind="use", uses=_expr_vars(stmt.value)))
        _flatten_block(stmt.body, events, type_map.copy())
    elif isinstance(stmt, Return):
        events.append(Event(kind="use", uses=_expr_vars(stmt.value)))
    elif isinstance(stmt, Barrier):
        events.append(Event(kind="barrier"))
    elif isinstance(stmt, Fence):
        events.append(Event(kind="fence"))
    elif isinstance(stmt, RawCode):
        # RawCode is excluded from generic baseline. Count it as unknown rather
        # than trying to infer a target match.
        events.append(Event(kind="raw", text=stmt.text))
    elif isinstance(stmt, (Break, Continue, UnrollHint)):
        events.append(Event(kind="use"))
    elif isinstance(stmt, Block):
        _flatten_block(stmt, events, type_map.copy())


def _flatten_block(block: Optional[Block], events: List[Event], type_map: Dict[str, Any]) -> None:
    if block is None:
        return
    for stmt in block.stmts:
        _flatten_stmt(stmt, events, type_map)


def _backward_output_reach(events: Sequence[Event]) -> Set[str]:
    deps_by_var: Dict[str, Set[str]] = defaultdict(set)
    sinks: Set[str] = set()
    for event in events:
        if event.kind == "output":
            sinks.update(event.output_rhs)
            continue
        for name in event.defs:
            deps_by_var[name].update(event.uses)

    reachable = set(sinks)
    work = list(sinks)
    while work:
        name = work.pop()
        for dep in deps_by_var.get(name, ()):
            if dep not in reachable:
                reachable.add(dep)
                work.append(dep)
    return reachable


def _is_visibility_fold(event: Event) -> bool:
    """True for the generic checksum plumbing, not a target expression."""
    return event.kind == "def" and event.defs == {"_fv"} and "_fv" in event.uses


def analyze_live_ranges_across_barrier(program: Program, live_threshold: int = DEFAULT_LIVE_THRESHOLD) -> SourceSignal:
    kernel = kernel_of(program)
    if kernel is None or kernel.body is None:
        return SourceSignal(0, 0, 0, 0, 0, False, {}, [])

    type_map: Dict[str, Any] = {p.name: p.gtype for p in kernel.params}
    scalar_params = {
        p.name for p in kernel.params
        if p.name not in {"output", "input"} and _is_scalar_type(p.gtype)
    }
    events: List[Event] = []
    _flatten_block(kernel.body, events, type_map)
    def_positions: Dict[str, int] = {}
    all_dep_defs = set()

    # Walk the statements in execution order and propagate input/thread/param
    # dependence through scalar definitions.
    events = _flatten_with_dependency(kernel, scalar_params, def_positions, all_dep_defs)
    reach = _backward_output_reach(events)

    barrier_positions = [idx for idx, event in enumerate(events) if event.kind == "barrier"]
    observable_sink_count = sum(1 for event in events if event.kind == "output")
    max_count = 0
    best_defs: List[str] = []
    names_used_after_any_barrier: Set[str] = set()
    for barrier_idx in barrier_positions:
        pre_dep_defs: Set[str] = set()
        for idx, event in enumerate(events[:barrier_idx]):
            pre_dep_defs.update(event.dep_defs)
        used_after: Set[str] = set()
        for event in events[barrier_idx + 1 :]:
            if _is_visibility_fold(event):
                continue
            used_after.update(event.uses)
        live = sorted((pre_dep_defs & used_after & reach) - RESERVED_VALUE_NAMES)
        names_used_after_any_barrier.update(pre_dep_defs & used_after)
        if len(live) > max_count:
            max_count = len(live)
            best_defs = live[:]

    primitives = {
        "input_dependent_defs_ge_L": len(all_dep_defs) >= live_threshold,
        "block_barrier": bool(barrier_positions),
        "uses_after_barrier": bool(names_used_after_any_barrier),
        "observable_sink": observable_sink_count > 0,
    }
    return SourceSignal(
        input_dependent_def_count=len(all_dep_defs),
        barrier_count=len(barrier_positions),
        use_after_barrier_count=len(names_used_after_any_barrier),
        observable_sink_count=observable_sink_count,
        max_cross_barrier_observable_live_defs=max_count,
        full_conjunction=max_count >= live_threshold,
        primitives=primitives,
        best_barrier_defs=best_defs,
    )


def _kernel_context(program: Program) -> Tuple[Optional[Func], Set[str]]:
    kernel = kernel_of(program)
    if kernel is None:
        return None, set()
    scalar_params = {
        p.name for p in kernel.params
        if p.name not in {"output", "input"} and _is_scalar_type(p.gtype)
    }
    return kernel, scalar_params


def _calls(kernel: Optional[Func]) -> List[Call]:
    if kernel is None:
        return []
    return collect(kernel, lambda node: isinstance(node, Call))


def _contains_call(expr: Optional[Expr], names: Set[str]) -> bool:
    if expr is None:
        return False
    return any(c.fn in names for c in collect(expr, lambda node: isinstance(node, Call)))


def _is_64_type(t: Any) -> bool:
    return isinstance(t, (TLong, TULong))


def _is_narrow_int_type(t: Any) -> bool:
    return isinstance(t, (TChar, TSChar, TShort, TUShort))


def _expr_contains_divrem64(expr: Optional[Expr]) -> bool:
    if expr is None:
        return False
    for node in collect(expr, lambda n: isinstance(n, (SafeBin, BinOp))):
        op_name = getattr(getattr(node, "op", None), "name", "")
        if op_name not in {"SDIV", "SMOD", "DIV", "MOD"}:
            continue
        lhs = getattr(node, "lhs", None)
        rhs = getattr(node, "rhs", None)
        if _is_64_type(getattr(node, "ty", None)) or _is_64_type(getattr(lhs, "ty", None)) or _is_64_type(getattr(rhs, "ty", None)):
            return True
    return False


def _expr_contains_shift64(expr: Optional[Expr]) -> bool:
    if expr is None:
        return False
    for node in collect(expr, lambda n: isinstance(n, (SafeBin, BinOp))):
        op_name = getattr(getattr(node, "op", None), "name", "")
        if op_name not in {"SSHL", "SSHR", "SHL", "SHR"}:
            continue
        lhs = getattr(node, "lhs", None)
        if _is_64_type(getattr(node, "ty", None)) or _is_64_type(getattr(lhs, "ty", None)):
            return True
    return False


def _expr_contains_bfe_idiom(expr: Optional[Expr]) -> bool:
    if expr is None:
        return False
    for node in collect(expr, lambda n: isinstance(n, BinOp)):
        if getattr(getattr(node, "op", None), "name", "") != "BAND":
            continue
        children = [getattr(node, "lhs", None), getattr(node, "rhs", None)]
        for child in children:
            for sub in collect(child, lambda n: isinstance(n, (SafeBin, BinOp))):
                op_name = getattr(getattr(sub, "op", None), "name", "")
                if op_name in {"SSHR", "SHR"}:
                    return True
    return False


def _assignment_values(kernel: Optional[Func]) -> Iterable[Expr]:
    if kernel is None:
        return []
    values: List[Expr] = []
    for stmt in collect(kernel, lambda node: isinstance(node, (Declare, Assign, CompoundAssign, ExprStmt, Return))):
        if isinstance(stmt, Declare) and stmt.init is not None:
            values.append(stmt.init)
        elif isinstance(stmt, Assign):
            values.append(stmt.value)
        elif isinstance(stmt, CompoundAssign):
            values.append(stmt.value)
        elif isinstance(stmt, ExprStmt):
            values.append(stmt.expr)
        elif isinstance(stmt, Return) and stmt.value is not None:
            values.append(stmt.value)
    return values


def _shared_names(kernel: Optional[Func]) -> Set[str]:
    if kernel is None:
        return set()
    return {
        decl.name
        for decl in collect(kernel, lambda node: isinstance(node, Declare))
        if decl.space is MemSpace.SHARED
    }


def _shared_accesses(expr: Optional[Expr], shared: Set[str]) -> Set[str]:
    if expr is None:
        return set()
    out: Set[str] = set()
    for sub in collect(expr, lambda node: isinstance(node, Subscript)):
        arr = getattr(sub, "array", None)
        if isinstance(arr, Var) and arr.name in shared:
            out.add(arr.name)
    return out


def _stmt_shared_writes(stmt: Stmt, shared: Set[str]) -> Set[str]:
    writes: Set[str] = set()
    if isinstance(stmt, Assign):
        writes.update(_shared_accesses(stmt.target, shared))
    elif isinstance(stmt, CompoundAssign):
        writes.update(_shared_accesses(stmt.target, shared))
    elif isinstance(stmt, ExprStmt) and isinstance(stmt.expr, Atomic):
        writes.update(_shared_accesses(stmt.expr.addr, shared))
    for child in _child_blocks(stmt):
        writes.update(_block_shared_writes(child, shared))
    return writes


def _block_shared_writes(block: Optional[Block], shared: Set[str]) -> Set[str]:
    if block is None:
        return set()
    out: Set[str] = set()
    for stmt in block.stmts:
        out.update(_stmt_shared_writes(stmt, shared))
    return out


def _stmt_shared_reads(stmt: Stmt, shared: Set[str]) -> Set[str]:
    reads: Set[str] = set()
    if isinstance(stmt, Declare):
        reads.update(_shared_accesses(stmt.init, shared))
    elif isinstance(stmt, Assign):
        reads.update(_shared_accesses(stmt.value, shared))
        target_name = _target_name(stmt.target)
        if target_name is None:
            reads.update(_shared_accesses(stmt.target, shared) - _stmt_shared_writes(stmt, shared))
    elif isinstance(stmt, CompoundAssign):
        reads.update(_shared_accesses(stmt.value, shared))
        reads.update(_shared_accesses(stmt.target, shared))
    elif isinstance(stmt, IncDecStmt):
        reads.update(_shared_accesses(stmt.target, shared))
    elif isinstance(stmt, ExprStmt):
        reads.update(_shared_accesses(stmt.expr, shared))
    elif isinstance(stmt, If):
        reads.update(_shared_accesses(stmt.cond, shared))
        reads.update(_block_shared_reads(stmt.then, shared))
        reads.update(_block_shared_reads(stmt.otherwise, shared))
    elif isinstance(stmt, For):
        reads.update(_shared_accesses(stmt.start, shared))
        reads.update(_shared_accesses(stmt.end, shared))
        reads.update(_shared_accesses(stmt.step, shared))
        reads.update(_block_shared_reads(stmt.body, shared))
    elif isinstance(stmt, While):
        reads.update(_shared_accesses(stmt.cond, shared))
        reads.update(_block_shared_reads(stmt.body, shared))
    elif isinstance(stmt, DoWhile):
        reads.update(_block_shared_reads(stmt.body, shared))
        reads.update(_shared_accesses(stmt.cond, shared))
    elif isinstance(stmt, Switch):
        reads.update(_shared_accesses(stmt.expr, shared))
        for case in stmt.cases:
            reads.update(_stmt_shared_reads(case, shared))
    elif isinstance(stmt, Case):
        reads.update(_shared_accesses(stmt.value, shared))
        reads.update(_block_shared_reads(stmt.body, shared))
    elif isinstance(stmt, Return):
        reads.update(_shared_accesses(stmt.value, shared))
    return reads


def _block_shared_reads(block: Optional[Block], shared: Set[str]) -> Set[str]:
    if block is None:
        return set()
    out: Set[str] = set()
    for stmt in block.stmts:
        out.update(_stmt_shared_reads(stmt, shared))
    return out


def _block_has_shared_barrier_exchange(block: Optional[Block], shared: Set[str]) -> bool:
    if block is None:
        return False
    written_before: Set[str] = set()
    for idx, stmt in enumerate(block.stmts):
        if isinstance(stmt, Barrier):
            reads_after = set()
            for later in block.stmts[idx + 1 :]:
                reads_after.update(_stmt_shared_reads(later, shared))
            if written_before & reads_after:
                return True
        written_before.update(_stmt_shared_writes(stmt, shared))
        for child in _child_blocks(stmt):
            if _block_has_shared_barrier_exchange(child, shared):
                return True
    return False


def _child_blocks(stmt: Stmt) -> Iterable[Block]:
    if isinstance(stmt, If):
        yield stmt.then
        if stmt.otherwise is not None:
            yield stmt.otherwise
    elif isinstance(stmt, For):
        yield stmt.body
    elif isinstance(stmt, While):
        yield stmt.body
    elif isinstance(stmt, DoWhile):
        yield stmt.body
    elif isinstance(stmt, Switch):
        for case in stmt.cases:
            yield case.body
    elif isinstance(stmt, Case):
        yield stmt.body
    elif isinstance(stmt, Block):
        yield stmt


def analyze_source_target_suite(program: Program, live_threshold: int = DEFAULT_LIVE_THRESHOLD) -> Dict[str, Dict[str, Any]]:
    kernel, scalar_params = _kernel_context(program)
    live = analyze_live_ranges_across_barrier(program, live_threshold)
    calls = _calls(kernel)
    call_names = Counter(c.fn for c in calls)
    expr_values = list(_assignment_values(kernel))

    byte_perm = call_names["__byte_perm"]
    mulhi_names = {"__umulhi", "__mulhi", "__umul64hi", "__mul64hi", "__umul24", "__mul24"}
    mulhi_count = sum(call_names[name] for name in mulhi_names)
    ffsll_count = call_names["__ffsll"]
    ffsll_byte_perm_same_expr = sum(
        1 for expr in expr_values
        if _contains_call(expr, {"__ffsll"}) and _contains_call(expr, {"__byte_perm"})
    )
    divrem64_casts = [
        cast for cast in collect(kernel, lambda node: isinstance(node, Cast)) if _is_narrow_int_type(cast.to) and _expr_contains_divrem64(cast.arg)
    ] if kernel is not None else []
    shift64_casts = [
        cast for cast in collect(kernel, lambda node: isinstance(node, Cast)) if _is_narrow_int_type(cast.to) and _expr_contains_shift64(cast.arg)
    ] if kernel is not None else []
    bfe_idiom_exprs = [expr for expr in expr_values if _expr_contains_bfe_idiom(expr)]
    atomics = collect(kernel, lambda node: isinstance(node, Atomic)) if kernel is not None else []
    atomic_xor_output = [
        atom for atom in atomics
        if atom.what == "Xor" and _is_output_store(atom.addr)
    ]
    switches = collect(kernel, lambda node: isinstance(node, Switch)) if kernel is not None else []
    divergent_switches = [
        sw for sw in switches
        if len(sw.cases) >= 3 and (_expr_has_source(sw.expr, scalar_params | {"tid"}) or bool(_expr_vars(sw.expr) & {"tid"}))
    ]
    shared = _shared_names(kernel)
    shared_exchange = bool(kernel and _block_has_shared_barrier_exchange(kernel.body, shared))

    return {
        "sync_live_l4": live.as_dict(),
        "byte_perm_call": _simple_signal(
            byte_perm > 0,
            {"byte_perm_call": byte_perm > 0, "observable_sink": live.observable_sink_count > 0},
            call_count=byte_perm,
        ),
        "mulhi_call": _simple_signal(
            mulhi_count > 0,
            {"mulhi_call": mulhi_count > 0, "observable_sink": live.observable_sink_count > 0},
            call_count=mulhi_count,
        ),
        "ffsll_byte_perm_mix": _simple_signal(
            ffsll_byte_perm_same_expr > 0,
            {"ffsll_call": ffsll_count > 0, "byte_perm_call": byte_perm > 0, "same_expression": ffsll_byte_perm_same_expr > 0},
            ffsll_call_count=ffsll_count,
            byte_perm_call_count=byte_perm,
            same_expression_count=ffsll_byte_perm_same_expr,
        ),
        "u64_divrem_narrow": _simple_signal(
            bool(divrem64_casts),
            {"u64_divrem": any(_expr_contains_divrem64(expr) for expr in expr_values), "narrowing_cast": bool(divrem64_casts)},
            cast_count=len(divrem64_casts),
        ),
        "u64_shift_narrow": _simple_signal(
            bool(shift64_casts),
            {"u64_shift": any(_expr_contains_shift64(expr) for expr in expr_values), "narrowing_cast": bool(shift64_casts)},
            cast_count=len(shift64_casts),
        ),
        "bfe_shift_mask_idiom": _simple_signal(
            bool(bfe_idiom_exprs),
            {"shift_then_mask": bool(bfe_idiom_exprs), "observable_sink": live.observable_sink_count > 0},
            expression_count=len(bfe_idiom_exprs),
        ),
        "atomic_xor_output": _simple_signal(
            bool(atomic_xor_output),
            {"atomic": bool(atomics), "xor_variant": any(a.what == "Xor" for a in atomics), "output_address": bool(atomic_xor_output)},
            atomic_count=len(atomics),
            atomic_xor_output_count=len(atomic_xor_output),
        ),
        "thread_dependent_switch_ge3": _simple_signal(
            bool(divergent_switches),
            {"switch": bool(switches), "ge3_cases": any(len(sw.cases) >= 3 for sw in switches), "thread_dependent_key": bool(divergent_switches)},
            switch_count=len(switches),
            divergent_switch_count=len(divergent_switches),
        ),
        "shared_write_barrier_read": _simple_signal(
            shared_exchange,
            {"shared_array": bool(shared), "block_barrier": live.barrier_count > 0, "same_array_write_then_read": shared_exchange},
            shared_array_count=len(shared),
        ),
    }


def _flatten_with_dependency(
    kernel: Func,
    scalar_params: Set[str],
    def_positions: Dict[str, int],
    all_dep_defs: Set[str],
) -> List[Event]:
    events: List[Event] = []
    type_map: Dict[str, Any] = {p.name: p.gtype for p in kernel.params}
    state = _DependencyState(scalar_params=set(scalar_params), dependent=set(scalar_params))
    _flatten_stmt_with_dependency(kernel.body, events, type_map, state)
    def_positions.clear()
    all_dep_defs.clear()
    for pos, event in enumerate(events):
        for name in event.dep_defs:
            def_positions[name] = pos
            all_dep_defs.add(name)
    return events


@dataclass
class _DependencyState:
    scalar_params: Set[str]
    dependent: Set[str]

    def child(self) -> "_DependencyState":
        return _DependencyState(self.scalar_params, set(self.dependent))


def _expr_dep(expr: Optional[Expr], state: _DependencyState) -> bool:
    return _expr_has_source(expr, state.scalar_params) or bool(_expr_vars(expr) & state.dependent)


def _record_def(event: Event, state: _DependencyState, is_dep: bool) -> None:
    for name in event.defs:
        if name in RESERVED_VALUE_NAMES:
            continue
        if is_dep:
            state.dependent.add(name)
            event.dep_defs.add(name)
        else:
            state.dependent.discard(name)


def _flatten_stmt_with_dependency(
    stmt: Stmt,
    events: List[Event],
    type_map: Dict[str, Any],
    state: _DependencyState,
) -> None:
    _stmt_type_decls(stmt, type_map)
    if isinstance(stmt, Block):
        for child in stmt.stmts:
            _flatten_stmt_with_dependency(child, events, type_map, state)
        return
    if isinstance(stmt, Declare):
        event = Event(kind="def", uses=_expr_vars(stmt.init))
        if _is_scalar_type(stmt.gtype):
            event.defs.add(stmt.name)
        _record_def(event, state, _expr_dep(stmt.init, state))
        events.append(event)
        return
    if isinstance(stmt, Assign):
        target_name = _target_name(stmt.target)
        target_uses = _expr_vars(stmt.target)
        uses = _expr_vars(stmt.value) | (target_uses - ({target_name} if target_name else set()))
        if _is_output_store(stmt.target):
            events.append(Event(kind="output", uses=uses, output_rhs=_expr_vars(stmt.value)))
            return
        event = Event(kind="def", uses=uses)
        if target_name and _is_scalar_type(type_map.get(target_name)):
            event.defs.add(target_name)
        _record_def(event, state, _expr_dep(stmt.value, state) or bool((target_uses - {target_name}) & state.dependent))
        events.append(event)
        return
    if isinstance(stmt, CompoundAssign):
        target_name = _target_name(stmt.target)
        uses = _expr_vars(stmt.target) | _expr_vars(stmt.value)
        if _is_output_store(stmt.target):
            events.append(Event(kind="output", uses=uses, output_rhs=uses))
            return
        event = Event(kind="def", uses=uses)
        if target_name and _is_scalar_type(type_map.get(target_name)):
            event.defs.add(target_name)
        _record_def(event, state, _expr_dep(stmt.value, state) or bool(_expr_vars(stmt.target) & state.dependent))
        events.append(event)
        return
    if isinstance(stmt, IncDecStmt):
        target_name = _target_name(stmt.target)
        event = Event(kind="def", uses=_expr_vars(stmt.target))
        if target_name and _is_scalar_type(type_map.get(target_name)):
            event.defs.add(target_name)
        _record_def(event, state, bool(_expr_vars(stmt.target) & state.dependent))
        events.append(event)
        return
    if isinstance(stmt, ExprStmt):
        events.append(Event(kind="use", uses=_expr_vars(stmt.expr)))
        return
    if isinstance(stmt, If):
        events.append(Event(kind="use", uses=_expr_vars(stmt.cond)))
        then_state = state.child()
        _flatten_stmt_with_dependency(stmt.then, events, type_map.copy(), then_state)
        state.dependent.update(then_state.dependent)
        if stmt.otherwise is not None:
            else_state = state.child()
            _flatten_stmt_with_dependency(stmt.otherwise, events, type_map.copy(), else_state)
            state.dependent.update(else_state.dependent)
        return
    if isinstance(stmt, For):
        events.append(Event(kind="use", uses=_expr_vars(stmt.start) | _expr_vars(stmt.end) | _expr_vars(stmt.step)))
        loop_state = state.child()
        loop_state.dependent.add(stmt.var)
        _flatten_stmt_with_dependency(stmt.body, events, type_map.copy(), loop_state)
        state.dependent.update(loop_state.dependent)
        return
    if isinstance(stmt, While):
        events.append(Event(kind="use", uses=_expr_vars(stmt.cond)))
        body_state = state.child()
        _flatten_stmt_with_dependency(stmt.body, events, type_map.copy(), body_state)
        events.append(Event(kind="use", uses=_expr_vars(stmt.cond)))
        state.dependent.update(body_state.dependent)
        return
    if isinstance(stmt, DoWhile):
        body_state = state.child()
        _flatten_stmt_with_dependency(stmt.body, events, type_map.copy(), body_state)
        events.append(Event(kind="use", uses=_expr_vars(stmt.cond)))
        state.dependent.update(body_state.dependent)
        return
    if isinstance(stmt, Switch):
        events.append(Event(kind="use", uses=_expr_vars(stmt.expr)))
        for case in stmt.cases:
            case_state = state.child()
            _flatten_stmt_with_dependency(case, events, type_map.copy(), case_state)
            state.dependent.update(case_state.dependent)
        return
    if isinstance(stmt, Case):
        events.append(Event(kind="use", uses=_expr_vars(stmt.value)))
        _flatten_stmt_with_dependency(stmt.body, events, type_map.copy(), state)
        return
    if isinstance(stmt, Return):
        events.append(Event(kind="use", uses=_expr_vars(stmt.value)))
        return
    if isinstance(stmt, Barrier):
        events.append(Event(kind="barrier"))
        return
    if isinstance(stmt, Fence):
        events.append(Event(kind="fence"))
        return
    if isinstance(stmt, RawCode):
        events.append(Event(kind="raw", text=stmt.text))
        return
    events.append(Event(kind="use"))


def _type_repr(t: Any) -> Any:
    if isinstance(t, TArray):
        return ["TArray", _type_repr(t.elem), t.count]
    if isinstance(t, TPointer):
        return ["TPointer", _type_repr(t.base)]
    if isinstance(t, TQualified):
        return ["TQualified", _type_repr(t.base), [q.name for q in t.quals]]
    if isinstance(t, TVec):
        return ["TVec", _type_repr(t.elem), t.lanes]
    if isinstance(t, TStruct):
        return ["TStruct", "S", [[f"f{i}", _type_repr(ft)] for i, (_name, ft) in enumerate(t.fields)]]
    if isinstance(t, GType):
        return type(t).__name__
    return repr(t)


def normalized_ir_hash(program: Program) -> str:
    names: Dict[str, str] = {}
    fn_names: Dict[str, str] = {}

    def name_of(name: str) -> str:
        if name in {"input", "output", "n", "tid", "_fv", "_rep", "_ci", "_fv_sink", "_ov_sink"}:
            return name
        if name not in names:
            names[name] = f"v{len(names)}"
        return names[name]

    def fn_of(name: str) -> str:
        if name == "kernel_main":
            return name
        if name.startswith("helper_"):
            if name not in fn_names:
                fn_names[name] = f"helper_{len(fn_names)}"
            return fn_names[name]
        return name

    def canon(obj: Any, key: str = "") -> Any:
        if obj is None or isinstance(obj, (bool, int, float, str)):
            if key in {"name", "var"} and isinstance(obj, str):
                return name_of(obj)
            if key == "fn" and isinstance(obj, str):
                return fn_of(obj)
            if key == "field" and isinstance(obj, str):
                return "field"
            return obj
        if isinstance(obj, enum.Enum):
            return obj.name
        if isinstance(obj, GType):
            return _type_repr(obj)
        if isinstance(obj, Lit):
            return ["Lit", _type_repr(obj.ty), type(obj.value).__name__]
        if isinstance(obj, Var):
            return ["Var", _type_repr(obj.ty), name_of(obj.name), obj.space.name]
        if isinstance(obj, ThreadBuiltin):
            return ["ThreadBuiltin", _type_repr(obj.ty), obj.source, obj.axis]
        if isinstance(obj, LaunchConfig):
            return ["LaunchConfig", obj.block, obj.grid, obj.body_repeat, obj.generic_baseline]
        if isinstance(obj, Program):
            return ["Program", [canon(d) for d in obj.decls]]
        if isinstance(obj, Func):
            return [
                "Func",
                fn_of(obj.name),
                [canon(p) for p in obj.params],
                _type_repr(obj.ret),
                [q.name for q in obj.quals],
                canon(obj.body),
            ]
        if isinstance(obj, StructDecl):
            return ["StructDecl", _type_repr(obj.struct)]
        if dataclasses.is_dataclass(obj):
            out = [type(obj).__name__]
            for field_info in dataclasses.fields(obj):
                if field_info.name in {"uniformity", "emit_ctx"}:
                    continue
                out.append([field_info.name, canon(getattr(obj, field_info.name), field_info.name)])
            return out
        if isinstance(obj, (list, tuple)):
            return [canon(v) for v in obj]
        if isinstance(obj, set):
            return sorted(canon(v) for v in obj)
        return repr(obj)

    blob = json.dumps(canon(program), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


_REG_RE = re.compile(r"%(?:[A-Za-z]+)\d+")
_INST_RE = re.compile(r"^(?:@\S+\s+)?([A-Za-z_.$][\w.$]*)\s*(.*)$")


def _split_ptx_instructions(ptx: str) -> List[Tuple[str, str, str]]:
    instructions: List[Tuple[str, str, str]] = []
    buf = ""
    for raw in ptx.splitlines():
        line = raw.split("//", 1)[0].strip()
        if not line or line.startswith(".") or line in {"{", "}"}:
            continue
        buf = f"{buf} {line}".strip()
        if ";" not in buf:
            continue
        parts = buf.split(";")
        for inst in parts[:-1]:
            inst = inst.strip()
            if not inst:
                continue
            match = _INST_RE.match(inst)
            if match:
                instructions.append((match.group(1), match.group(2), inst))
        buf = parts[-1].strip()
    return instructions


def _operand_groups(rest: str) -> List[str]:
    groups: List[str] = []
    cur: List[str] = []
    depth = 0
    for ch in rest:
        if ch in "[{(":
            depth += 1
        elif ch in "]})":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            groups.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        groups.append("".join(cur).strip())
    return groups


def _ptx_defs_uses(mnemonic: str, rest: str) -> Tuple[Set[str], Set[str]]:
    regs = set(_REG_RE.findall(rest))
    base = mnemonic.split(".", 1)[0]
    no_dest = {
        "st",
        "bra",
        "ret",
        "bar",
        "membar",
        "exit",
        "trap",
        "prefetch",
        "discard",
        "red",
    }
    defs: Set[str] = set()
    if base not in no_dest:
        groups = _operand_groups(rest)
        if groups:
            first = groups[0]
            defs = set(_REG_RE.findall(first))
    return defs, regs - defs


def analyze_ptx_signal(ptx: str, live_threshold: int = DEFAULT_LIVE_THRESHOLD) -> Tuple[int, int, bool]:
    insts = _split_ptx_instructions(ptx)
    barriers = [idx for idx, (mn, _rest, _raw) in enumerate(insts) if mn.startswith("bar.sync") or mn.startswith("barrier.sync")]
    max_live = 0
    for barrier_idx in barriers:
        defined: Set[str] = set()
        for mn, rest, _raw in insts[:barrier_idx]:
            defs, _uses = _ptx_defs_uses(mn, rest)
            defined.update(defs)
        redefined: Set[str] = set()
        live: Set[str] = set()
        for mn, rest, _raw in insts[barrier_idx + 1 :]:
            defs, uses = _ptx_defs_uses(mn, rest)
            for reg in uses:
                if reg in defined and reg not in redefined:
                    live.add(reg)
            redefined.update(defs & defined)
        max_live = max(max_live, len(live))
    return len(barriers), max_live, bool(barriers and max_live >= live_threshold)


def _nvcc_version(nvcc: str) -> str:
    try:
        proc = subprocess.run([nvcc, "--version"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
    except Exception as exc:
        return f"unavailable: {exc}"
    return " | ".join(line.strip() for line in proc.stdout.splitlines() if line.strip())


def compile_and_analyze_ptx(source: str, nvcc: str, arch: str, live_threshold: int, timeout_s: int) -> ArtifactSignal:
    version = _nvcc_version(nvcc)
    with tempfile.TemporaryDirectory(prefix="e0_ptx_") as td:
        src = Path(td) / "test.cu"
        ptx_path = Path(td) / "test.ptx"
        src.write_text(source, encoding="utf-8")
        cmd = [nvcc, "-std=c++17", "-O3", f"-arch={arch}", "-ptx", str(src), "-o", str(ptx_path)]
        try:
            proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return ArtifactSignal(False, False, nvcc_version=version, error=f"nvcc timeout after {timeout_s}s")
        except Exception as exc:
            return ArtifactSignal(False, False, nvcc_version=version, error=str(exc))
        if proc.returncode != 0:
            return ArtifactSignal(False, False, nvcc_version=version, error=proc.stdout[-4000:])
        ptx = ptx_path.read_text(encoding="utf-8", errors="replace")
    barrier_count, max_live, hit = analyze_ptx_signal(ptx, live_threshold)
    return ArtifactSignal(
        compiled=True,
        artifact_hit=hit,
        barrier_count=barrier_count,
        max_live_regs_across_barrier=max_live,
        ptx_sha256=hashlib.sha256(ptx.encode("utf-8")).hexdigest(),
        nvcc_version=version,
    )


def _generate_case(index: int, args_dict: Dict[str, Any]) -> Dict[str, Any]:
    mode = args_dict["mode"]
    params = {
        "generic_baseline": mode == "generic",
        "complexity": args_dict["complexity"],
        "amplify_iters": args_dict["amplify_iters"],
        "scalar_param_count": args_dict["scalar_param_count"],
        "use_full_int_widths": args_dict["use_full_int_widths"],
        "enable_smem": args_dict["enable_smem"],
        "enable_atomics": args_dict["enable_atomics"],
        "enable_warp": args_dict["enable_warp"],
        "enable_barriers": args_dict["enable_barriers"],
        "pass_targeted_only": mode in {"arm-b", "arm-c"},
    }
    intended_template = ""
    intended_targets: Tuple[str, ...] = ()
    if mode == "positive":
        params["generic_baseline"] = False
        params["template"] = "sync_span_live"
        intended_template = "sync_span_live"
        intended_targets = ARM_C_TEMPLATE_TARGETS[intended_template]
    elif mode in {"arm-b", "arm-c"}:
        templates = tuple(args_dict.get("templates") or _default_templates_for_mode(mode))
        if not templates:
            raise ValueError(f"{mode} mode requires at least one template")
        intended_template = templates[index % len(templates)]
        params["generic_baseline"] = False
        params["template"] = intended_template
        if mode == "arm-b":
            params["template_catalog"] = "raw_llm"
        intended_targets = tuple(
            args_dict.get("template_targets", {}).get(
                intended_template,
                ALL_TEMPLATE_TARGETS.get(intended_template, ()),
            )
        )

    strategy = default_registry().get("generator_ir")
    config_seed = args_dict["config_seed_base"] + index
    test_seed = args_dict["test_seed_base"] + index
    case = strategy.generate(config_seed, test_seed, params, "e0", args_dict["arch"])
    program = pickle.loads(case.ir_blob)
    source_signal = analyze_live_ranges_across_barrier(program, args_dict["live_threshold"])
    target_signals = analyze_source_target_suite(program, args_dict["live_threshold"])
    h = normalized_ir_hash(program)
    inline_asm_violation = _source_has_inline_asm(case.source)
    feature_error = ""
    try:
        features = pull_features(program, case.params.get("template", "")).counts
    except Exception as exc:
        features = {}
        feature_error = type(exc).__name__
    any_source_target_hit = any(sig.get("full_conjunction") for sig in target_signals.values())
    record: Dict[str, Any] = {
        "index": index,
        "test_id": case.test_id,
        "case_id": case.case_id(),
        "mode": args_dict["mode"],
        "target": TARGET_NAME,
        "intended_template": intended_template,
        "intended_targets": list(intended_targets),
        "target_sm": args_dict["arch"],
        "config_seed": config_seed,
        "test_seed": test_seed,
        "kernel_seed": case.kernel_seed,
        "params": case.params,
        "source_sha256": case.source_sha256,
        "normalized_ir_sha256": h,
        "inline_assembly_violation": inline_asm_violation,
        "source_signal": source_signal.as_dict(),
        "target_signals": target_signals,
        "features": features,
        "feature_error": feature_error,
        "source": case.source if (any_source_target_hit or args_dict["return_all_sources"]) else "",
    }
    compile_mode = args_dict["compile_mode"]
    should_compile = (
        compile_mode == "all"
        or (compile_mode == "source-hits" and source_signal.full_conjunction)
        or args_dict["mode"] == "positive"
    )
    if args_dict["mode"] == "arm-b" and inline_asm_violation:
        should_compile = False
    if should_compile and args_dict["nvcc"]:
        record["artifact_signal"] = compile_and_analyze_ptx(
            case.source,
            args_dict["nvcc"],
            args_dict["arch"],
            args_dict["live_threshold"],
            args_dict["compile_timeout_s"],
        ).as_dict()
    else:
        record["artifact_signal"] = ArtifactSignal(False, False).as_dict()
    return record


def _wilson(success: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    phat = success / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    margin = z * ((phat * (1 - phat) / total + z * z / (4 * total * total)) ** 0.5) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _rate(success: int, total: int) -> Dict[str, Any]:
    lo, hi = _wilson(success, total)
    return {
        "count": success,
        "total": total,
        "rate": success / total if total else 0.0,
        "wilson95": [lo, hi],
    }


def summarize_records(records: Sequence[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    primitive_counts = Counter()
    suite_primitive_counts: Dict[str, Counter] = defaultdict(Counter)
    suite_metric_max: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    suite_metric_sum: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    suite_hits = Counter()
    feature_counts = Counter()
    feature_errors = Counter()
    max_live_hist = Counter()
    compile_failures = 0
    artifact_compiled = 0
    artifact_hits = 0
    full_hits = 0
    inline_assembly_violations = 0
    arm_c_by_template: Dict[str, Counter] = defaultdict(Counter)
    arm_c_template_targets: Dict[str, Set[str]] = defaultdict(set)
    for rec in records:
        if rec.get("inline_assembly_violation"):
            inline_assembly_violations += 1
        src = rec["source_signal"]
        if src["full_conjunction"]:
            full_hits += 1
        for target, sig in rec.get("target_signals", {}).items():
            if sig.get("full_conjunction"):
                suite_hits[target] += 1
            for name, hit in sig.get("primitives", {}).items():
                if hit:
                    suite_primitive_counts[target][name] += 1
            for name, value in sig.get("metrics", {}).items():
                if isinstance(value, bool):
                    value = int(value)
                if isinstance(value, int):
                    suite_metric_sum[target][name] += value
                    suite_metric_max[target][name] = max(suite_metric_max[target][name], value)
        for name, hit in src["primitives"].items():
            if hit:
                primitive_counts[name] += 1
        max_live_hist[str(src["max_cross_barrier_observable_live_defs"])] += 1
        for name, count in rec.get("features", {}).items():
            feature_counts[name] += count
        if rec.get("feature_error"):
            feature_errors[rec["feature_error"]] += 1
        art = rec.get("artifact_signal", {})
        if art.get("compiled"):
            artifact_compiled += 1
        elif art.get("error"):
            compile_failures += 1
        if art.get("artifact_hit"):
            artifact_hits += 1
        tmpl = rec.get("intended_template") or ""
        intended_targets = tuple(rec.get("intended_targets") or ())
        if tmpl:
            arm_c_by_template[tmpl]["total"] += 1
            arm_c_template_targets[tmpl].update(intended_targets)
            if any(
                rec.get("target_signals", {}).get(target, {}).get("full_conjunction")
                for target in intended_targets
            ):
                arm_c_by_template[tmpl]["intended_hit"] += 1
            for target in intended_targets:
                if rec.get("target_signals", {}).get(target, {}).get("full_conjunction"):
                    arm_c_by_template[tmpl][f"target:{target}"] += 1

    total = len(records)
    unique_ir = len({rec["normalized_ir_sha256"] for rec in records})
    target_names = sorted({target for rec in records for target in rec.get("target_signals", {})})
    target_suite = {}
    for target in target_names:
        primitives = {
            name: _rate(suite_primitive_counts[target][name], total)
            for name in sorted(suite_primitive_counts[target])
        }
        target_suite[target] = {
            "source_full_conjunction": _rate(suite_hits[target], total),
            "primitives": primitives,
            "metric_sums": dict(sorted(suite_metric_sum[target].items())),
            "metric_max": dict(sorted(suite_metric_max[target].items())),
        }
    arm_c_summary = {}
    if arm_c_by_template:
        family_hits = 0
        for tmpl in sorted(arm_c_by_template):
            c = arm_c_by_template[tmpl]
            tmpl_total = c["total"]
            intended_hit = c["intended_hit"]
            if intended_hit:
                family_hits += 1
            target_rates = {
                target: _rate(c[f"target:{target}"], tmpl_total)
                for target in sorted(arm_c_template_targets[tmpl])
            }
            arm_c_summary[tmpl] = {
                "intended_targets": sorted(arm_c_template_targets[tmpl]),
                "instances": tmpl_total,
                "intended_instance_hit": _rate(intended_hit, tmpl_total),
                "target_instance_hits": target_rates,
            }
        arm_c_summary["_overall"] = {
            "families": len(arm_c_by_template),
            "family_hit": _rate(family_hits, len(arm_c_by_template)),
            "intended_instance_hit": _rate(
                sum(c["intended_hit"] for c in arm_c_by_template.values()),
                sum(c["total"] for c in arm_c_by_template.values()),
            ),
        }
    return {
        "experiment": _experiment_name(args.mode),
        "target": TARGET_NAME,
        "definition": {
            "live_threshold_L": args.live_threshold,
            "source_full_conjunction": [
                "InputDependentDefs_ge_L",
                "BlockBarrier",
                "UsesAfterBarrier",
                "ObservableSink",
            ],
            "artifact_signal": "PTX contains bar.sync/barrier.sync and at least L PTX registers live across that barrier",
            "inline_assembly": (
                "target-instruction inline assembly is not generated; source-level checkers do not count embedded target mnemonics"
                if args.mode in {"arm-b", "arm-c"}
                else "excluded by generic_baseline mode"
            ),
        },
        "generator": {
            "strategy": "generator_ir",
            "mode": args.mode,
            "template_catalog": "raw_llm" if args.mode == "arm-b" else "default",
            "pass_targeted_only": args.mode in {"arm-b", "arm-c"},
            "complexity": args.complexity,
            "amplify_iters": args.amplify_iters,
            "arch": args.arch,
            "count": total,
            "config_seed_base": args.config_seed_base,
            "test_seed_base": args.test_seed_base,
        },
        "compiler": {
            "compile_mode": args.compile_mode,
            "nvcc": args.nvcc or "",
            "compiled": artifact_compiled,
            "compile_failures": compile_failures,
            "nvcc_version": next((rec["artifact_signal"]["nvcc_version"] for rec in records if rec.get("artifact_signal", {}).get("nvcc_version")), ""),
        },
        "diversity": {
            "unique_normalized_ir_count": unique_ir,
            "unique_normalized_ir_rate": unique_ir / total if total else 0.0,
        },
        "source_primitives": {
            name: _rate(primitive_counts[name], total)
            for name in [
                "input_dependent_defs_ge_L",
                "block_barrier",
                "uses_after_barrier",
                "observable_sink",
            ]
        },
        "source_full_conjunction": _rate(full_hits, total),
        "target_suite": target_suite,
        "artifact_signal": _rate(artifact_hits, total),
        "inline_assembly_violations": _rate(inline_assembly_violations, total),
        "max_cross_barrier_observable_live_defs_histogram": dict(sorted(max_live_hist.items(), key=lambda kv: int(kv[0]))),
        "top_features": dict(feature_counts.most_common(30)),
        "feature_errors": dict(feature_errors),
        "template_families": arm_c_summary,
        "arm_b": arm_c_summary if args.mode == "arm-b" else {},
        "arm_c": arm_c_summary if args.mode == "arm-c" else {},
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _write_readme(out_dir: Path, summary: Dict[str, Any]) -> None:
    src_full = summary["source_full_conjunction"]
    artifact = summary["artifact_signal"]
    inline_asm = summary.get("inline_assembly_violations", {"count": 0, "total": summary["generator"]["count"], "rate": 0.0})
    suite_lines = []
    for name, sig in sorted(summary.get("target_suite", {}).items()):
        hit = sig["source_full_conjunction"]
        suite_lines.append(
            f"- `{name}`: {hit['count']}/{hit['total']} "
            f"({hit['rate']:.6%}, Wilson 95% "
            f"[{hit['wilson95'][0]:.6%}, {hit['wilson95'][1]:.6%}])"
        )
    suite_text = "\n".join(suite_lines) if suite_lines else "- no target-suite records"
    family_lines = []
    families = summary.get("template_families", {})
    overall = families.get("_overall") if isinstance(families, dict) else None
    if overall:
        fh = overall["family_hit"]
        ih = overall["intended_instance_hit"]
        family_lines.append(
            f"- family hit: {fh['count']}/{fh['total']} "
            f"({fh['rate']:.6%}, Wilson 95% "
            f"[{fh['wilson95'][0]:.6%}, {fh['wilson95'][1]:.6%}])"
        )
        family_lines.append(
            f"- intended instance hit: {ih['count']}/{ih['total']} "
            f"({ih['rate']:.6%}, Wilson 95% "
            f"[{ih['wilson95'][0]:.6%}, {ih['wilson95'][1]:.6%}])"
        )
        for tmpl, data in sorted(families.items()):
            if tmpl == "_overall":
                continue
            rate = data["intended_instance_hit"]
            family_lines.append(
                f"- `{tmpl}` -> {data['intended_targets']}: "
                f"{rate['count']}/{rate['total']} ({rate['rate']:.6%})"
            )
    family_text = "\n".join(family_lines) if family_lines else "- not a template-family run"
    if summary.get("experiment") == "RQ1-ArmC":
        title = "RQ1 Arm-C PassSpec-guided template-family corpus"
    elif summary.get("experiment") == "RQ1-ArmB":
        title = "RQ1 Arm-B Raw-LLM template-family corpus"
    else:
        title = "E0 generic-baseline diagnostic corpus"
    text = f"""# {title}

Target: `{summary['target']}`

- cases: {summary['generator']['count']}
- unique normalized IR: {summary['diversity']['unique_normalized_ir_count']} ({summary['diversity']['unique_normalized_ir_rate']:.6f})
- source full conjunction: {src_full['count']}/{src_full['total']} ({src_full['rate']:.6%}, Wilson 95% [{src_full['wilson95'][0]:.6%}, {src_full['wilson95'][1]:.6%}])
- artifact signal: {artifact['count']}/{artifact['total']} ({artifact['rate']:.6%}, Wilson 95% [{artifact['wilson95'][0]:.6%}, {artifact['wilson95'][1]:.6%}])
- inline assembly violations: {inline_asm['count']}/{inline_asm['total']} ({inline_asm['rate']:.6%})
- compile mode: {summary['compiler']['compile_mode']}; compiled {summary['compiler']['compiled']} cases; compile failures {summary['compiler']['compile_failures']}

Target suite:

{suite_text}

Template-family intended hits:

{family_text}

Files:

- `records.jsonl`: one record per generated case
- `summary.json`: aggregate counts and confidence intervals
- `samples/`: source for source-level hits and configured examples
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def _resolve_nvcc(value: str) -> str:
    if value:
        return value
    candidates = [
        "/usr/local/cuda-13.3/bin/nvcc",
        "/usr/local/cuda-13.0/bin/nvcc",
        "/usr/local/cuda-12.6/bin/nvcc",
        "/usr/local/cuda/bin/nvcc",
        shutil.which("nvcc") or "",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return ""


def _copy_args(args: argparse.Namespace) -> Dict[str, Any]:
    templates = tuple(args.templates or _default_templates_for_mode(args.mode))
    return {
        "mode": args.mode,
        "complexity": args.complexity,
        "amplify_iters": args.amplify_iters,
        "scalar_param_count": args.scalar_param_count,
        "use_full_int_widths": args.use_full_int_widths,
        "enable_smem": args.enable_smem,
        "enable_atomics": args.enable_atomics,
        "enable_warp": args.enable_warp,
        "enable_barriers": args.enable_barriers,
        "config_seed_base": args.config_seed_base,
        "test_seed_base": args.test_seed_base,
        "arch": args.arch,
        "live_threshold": args.live_threshold,
        "compile_mode": args.compile_mode,
        "compile_timeout_s": args.compile_timeout_s,
        "nvcc": args.nvcc,
        "return_all_sources": args.return_all_sources,
        "templates": templates,
        "template_targets": _template_targets_for_mode(args.mode),
    }


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = out_dir / "samples"
    sample_dir.mkdir(exist_ok=True)
    args.nvcc = _resolve_nvcc(args.nvcc)
    if args.compile_mode != "none" and not args.nvcc:
        print("warning: no nvcc found; artifact_signal will be skipped", file=sys.stderr)
        args.compile_mode = "none"

    records_path = out_dir / "records.jsonl"
    records: List[Dict[str, Any]] = []
    start = time.time()
    worker_args = _copy_args(args)

    def persist_record(rec: Dict[str, Any]) -> None:
        src = rec.pop("source", "")
        with records_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
        if src and (len(list(sample_dir.glob("*.cu"))) < args.max_saved_sources or rec["source_signal"]["full_conjunction"]):
            (sample_dir / f"{rec['index']:06d}_{rec['case_id']}.cu").write_text(src, encoding="utf-8")
        records.append(rec)

    if records_path.exists() and not args.append:
        records_path.unlink()

    if args.workers <= 1:
        for i in range(args.count):
            rec = _generate_case(i, worker_args)
            persist_record(rec)
            if (i + 1) % args.progress_every == 0 or i + 1 == args.count:
                elapsed = time.time() - start
                print(f"progress {i + 1}/{args.count} elapsed={elapsed:.1f}s", flush=True)
    else:
        with futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_generate_case, i, worker_args) for i in range(args.count)]
            for done, fut in enumerate(futures.as_completed(futs), 1):
                rec = fut.result()
                persist_record(rec)
                if done % args.progress_every == 0 or done == args.count:
                    elapsed = time.time() - start
                    print(f"progress {done}/{args.count} elapsed={elapsed:.1f}s", flush=True)

    summary = summarize_records(records, args)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_readme(out_dir, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["generic", "positive", "arm-b", "arm-c"], default="generic")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--out", default="runs/e0_generic_baseline")
    parser.add_argument("--arch", default="sm_86")
    parser.add_argument("--complexity", choices=["low", "medium", "high", "stress"], default="high")
    parser.add_argument("--amplify-iters", type=int, default=1)
    parser.add_argument("--scalar-param-count", type=int, default=12)
    parser.add_argument("--use-full-int-widths", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-smem", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-atomics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-warp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-barriers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--config-seed-base", type=int, default=0xE0000000)
    parser.add_argument("--test-seed-base", type=int, default=0xE1000000)
    parser.add_argument("--live-threshold", type=int, default=DEFAULT_LIVE_THRESHOLD)
    parser.add_argument("--compile-mode", choices=["none", "source-hits", "all"], default="source-hits")
    parser.add_argument("--nvcc", default="")
    parser.add_argument("--compile-timeout-s", type=int, default=60)
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--max-saved-sources", type=int, default=10)
    parser.add_argument("--return-all-sources", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--templates", nargs="+", default=None,
                        help="Template families to cycle through in Arm-B/Arm-C modes")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
