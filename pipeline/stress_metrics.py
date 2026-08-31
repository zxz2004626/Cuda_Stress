"""Compiler-stress gating and online-corpus novelty scoring.

A long source file is not automatically "stressful": every candidate must clear a
configurable gate (SASS diversity across ptxas profiles, minimum instruction counts,
target-opcode survival, input sensitivity) before it is retained as high-value.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .config import Config
from generator.base import GeneratedCase


def collect_build_metrics(test_dir: Path) -> List[Dict[str, Any]]:
    """Gather metrics.json for every build under test_dir."""
    out = []
    builds = test_dir / "builds"
    if not builds.exists():
        return out
    for bdir in sorted(builds.glob("*__*")):
        mfile = bdir / "metrics.json"
        if mfile.exists():
            try:
                m = json.loads(mfile.read_text())
                m["build"] = bdir.name
                out.append(m)
            except Exception:  # noqa: BLE001
                pass
    return out


def evaluate_stress_gate(config: Config, case: GeneratedCase, test_dir: Path,
                         metrics_list: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    gate_cfg = config.stress_gate
    metrics_list = metrics_list if metrics_list is not None else collect_build_metrics(test_dir)
    if not metrics_list:
        return {"passed": False, "reasons": ["no build metrics available"]}

    sass_hashes = {m.get("sass_normalized_hash") for m in metrics_list if m.get("sass_normalized_hash")}
    sass_counts = [m.get("sass_instruction_count", 0) for m in metrics_list]
    ptx_lines = [m.get("ptx_lines", 0) for m in metrics_list]

    checks: Dict[str, Any] = {}
    checks["min_ptx_lines"] = max(ptx_lines) if ptx_lines else 0
    checks["min_sass_instructions"] = max(sass_counts) if sass_counts else 0
    checks["distinct_sass_hashes"] = len(sass_hashes)
    if sass_counts:
        lo, hi = min(sass_counts), max(sass_counts)
        checks["sass_count_delta_ratio"] = round((hi - lo) / max(1, hi), 3)
    else:
        checks["sass_count_delta_ratio"] = 0.0

    reasons = []
    if checks["min_ptx_lines"] < gate_cfg.get("min_ptx_lines", 40):
        reasons.append(f"ptx_lines {checks['min_ptx_lines']} < {gate_cfg.get('min_ptx_lines', 40)}")
    if checks["min_sass_instructions"] < gate_cfg.get("min_sass_instructions", 50):
        reasons.append(f"sass_instructions {checks['min_sass_instructions']} < {gate_cfg.get('min_sass_instructions', 50)}")
    if gate_cfg.get("min_distinct_sass_hashes", 2) > 1 and \
       checks["distinct_sass_hashes"] < gate_cfg["min_distinct_sass_hashes"]:
        reasons.append(f"distinct_sass_hashes {checks['distinct_sass_hashes']} < {gate_cfg['min_distinct_sass_hashes']}")
    if gate_cfg.get("min_optimization_delta", 0.15) > 0 and \
       checks["sass_count_delta_ratio"] < gate_cfg["min_optimization_delta"]:
        reasons.append(f"sass_count_delta_ratio {checks['sass_count_delta_ratio']} < {gate_cfg['min_optimization_delta']}")

    # target opcode survival: strategy-specific opcodes must appear in PTX (not DCE'd).
    if gate_cfg.get("target_opcode_present", True):
        reg = None
        try:
            from generator.base import default_registry
            reg = default_registry().get(case.strategy)
        except Exception:  # noqa: BLE001
            reg = None
        if reg is not None and reg.target_opcodes:
            present = reg.target_opcodes[:1]
            if any(m.get("ptx_target_opcodes_present") for m in metrics_list):
                present = [t for m in metrics_list for t in m.get("ptx_target_opcodes_present", [])]
                present = sorted(set(present))[:8]
            checks["target_opcodes_present"] = present
            if not present:
                reasons.append("target opcodes fully DCE'd")

    passed = not reasons
    return {"passed": passed, "reasons": reasons, "checks": checks}


def novelty_score(config: Config, case: GeneratedCase, metrics_list: List[Dict[str, Any]],
                  corpus_hashes: Set[str], corpus_sass_hashes: Set[str]) -> float:
    """Weighted retention score (higher = more valuable to keep)."""
    w = config.novelty
    if not metrics_list:
        return 0.0
    score = 0.0

    # optimization_delta: normalized distinct SASS hashes (0..1)
    distinct = {m.get("sass_normalized_hash") for m in metrics_list if m.get("sass_normalized_hash")}
    opt_delta = len(distinct) / max(1, len(config.profiles))
    score += w.get("optimization_delta", 1.0) * opt_delta

    # sass_novelty: 1 - overlap with existing corpus SASS hashes
    novel_share = sum(1 for h in distinct if h not in corpus_sass_hashes) / max(1, len(distinct))
    score += w.get("sass_novelty", 1.0) * novel_share

    # register_pressure: max registers across builds (normalized to 255)
    regs = [m.get("registers", 0) or 0 for m in metrics_list]
    score += w.get("register_pressure", 0.5) * (max(regs) / 255.0 if regs else 0.0)

    # predicate_pressure: fraction of predicated SASS instructions
    pred_frac = 0.0
    for m in metrics_list:
        cnt = m.get("sass_instruction_count", 0) or 1
        pred_frac = max(pred_frac, m.get("sass_predicated_count", 0) / cnt)
    score += w.get("predicate_pressure", 0.6) * pred_frac

    # control_flow_complexity: branch fraction
    cf = 0.0
    for m in metrics_list:
        cnt = m.get("sass_instruction_count", 0) or 1
        cf = max(cf, m.get("sass_branch_count", 0) / cnt)
    score += w.get("control_flow_complexity", 0.5) * cf

    # rare_opcode_bonus: presence of strategy target opcodes in SASS histogram
    rare = 0
    for m in metrics_list:
        hist = m.get("sass_opcode_histogram", {})
        for op in ("PRMT", "SHF", "BFE", "LOP3", "POPC", "F2I", "MUFU", "VABSD", "LEA", "ISETP", "PSETP"):
            if op in hist:
                rare += 1
                break
    score += w.get("rare_opcode_bonus", 0.7) * min(1.0, rare)

    # compile_time_penalty
    elapsed = [m.get("elapsed_s", 0) or m.get("compile_elapsed_s", 0) for m in metrics_list]
    score -= w.get("compile_time_penalty", 0.05) * math.log1p(max(elapsed))

    # duplicate_penalty: exact source hash already in corpus
    if case.source_sha256 in corpus_hashes:
        score -= w.get("duplicate_penalty", 2.0)
    return round(score, 4)
