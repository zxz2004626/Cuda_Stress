"""Differential comparison: per-build stability, CPU-oracle equality, cross-matrix equality.

Verdict precedence (highest first):
    NONDETERMINISTIC > POTENTIAL_COMPILER_BUG > ORACLE_MISMATCH
    > RUN_TIMEOUT > RUN_FAILED > COMPILE_FAILED > INCOMPLETE_MATRIX > PASS

POTENTIAL_COMPILER_BUG requires multiple *stable* builds to produce different outputs
-- a single odd run or a timeout never masquerades as a wrong-code regression.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import Config
from .models import BuildResult, CaseResult, Status, Category
from .compiler import build_key
from .executor import parse_output_blob, BLOB_HEADER
from generator.base import GeneratedCase


def _read_output_words(path: Path) -> Tuple[str, Optional[bytes], int]:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        execution_path = path.with_name(
            path.name.replace("output_run_", "execution_run_").replace(".bin", ".json"))
        try:
            status = str(json.loads(execution_path.read_text()).get("status", "MISSING"))
        except Exception:  # noqa: BLE001
            status = "MISSING"
        if status in {"TIMEOUT", "CUDA_ERROR", "OUT_OF_BOUNDS_WRITE", "BAD_SPEC"}:
            return (status, None, 0)
        return ("MISSING", None, 0)
    parsed = parse_output_blob(data)
    return (parsed.status, parsed.output_words, parsed.meaningful_words)


def compare_build(config: Config, case: GeneratedCase, tc_id: str, profile: str,
                  build_dir: Path, runs: Optional[int] = None) -> BuildResult:
    br = BuildResult(toolchain=tc_id, profile=profile)
    compile_json = build_dir / "compile_result.json"
    if not compile_json.exists():
        # compile_result.json not written yet -> treat as compile failure / incomplete
        br.status = Status.INCOMPLETE_MATRIX
        return br
    try:
        cj = json.loads(compile_json.read_text())
    except Exception:  # noqa: BLE001
        cj = {"ok": False}
    br.compile_ok = bool(cj.get("ok", False))
    br.compile_error = cj.get("error")
    br.ptxas_registers = cj.get("registers")
    if not br.compile_ok:
        br.status = Status.COMPILE_FAILED
        return br

    # runs
    runs = max(1, int(runs or config.execution.get("runs_per_build", 3)))
    outputs: List[Optional[bytes]] = []
    statuses: List[str] = []
    for i in range(1, runs + 1):
        status, words, _ = _read_output_words(build_dir / f"output_run_{i}.bin")
        statuses.append(status)
        outputs.append(words)
    br.run_statuses = statuses
    br.run_output_hashes = [hashlib.sha256(o).hexdigest() if o else None for o in outputs]

    if any(s != "OK" for s in statuses):
        if any(s == "TIMEOUT" for s in statuses):
            br.status = Status.RUN_TIMEOUT
        elif any(s == "OUT_OF_BOUNDS_WRITE" for s in statuses):
            br.status = Status.RUN_FAILED
        elif any(s in ("CUDA_ERROR", "MISSING") for s in statuses):
            br.status = Status.RUN_FAILED
        else:
            br.status = Status.RUN_FAILED
        return br

    # all runs OK -> stability
    distinct = {h for h in br.run_output_hashes if h}
    br.stable = len(distinct) == 1
    if not br.stable:
        br.status = Status.NONDETERMINISTIC
        return br

    br.run_output_files = [str(build_dir / f"output_run_{i}.bin") for i in range(1, runs + 1)]
    if case.oracle_mode == "cross_build":
        # Stable within-build output participates in the matrix consensus
        # below.  There is deliberately no synthetic expected.bin oracle.
        br.status = Status.PASS
        return br

    expected = (build_dir.parent.parent / "expected.bin").read_bytes()
    expected_hash = hashlib.sha256(expected).hexdigest()
    if br.run_output_hashes[0] == expected_hash:
        br.status = Status.PASS
    else:
        br.status = Status.ORACLE_MISMATCH
    return br


def compare_matrix(config: Config, case: GeneratedCase, test_dir: Path,
                   runnable_ids: Optional[List[str]] = None,
                   runs: Optional[int] = None) -> CaseResult:
    """Compare every build under test_dir and produce the case verdict."""
    oracle_hash = None
    if case.oracle_mode != "cross_build":
        oracle_hash = hashlib.sha256((test_dir / "expected.bin").read_bytes()).hexdigest()
    result = CaseResult(test_id=case.test_id, strategy=case.strategy,
                        oracle_hash=oracle_hash)
    try:
        execution_plan = json.loads((test_dir / "execution_plan.json").read_text())
    except Exception:  # noqa: BLE001
        execution_plan = {}
    configured_ids = {t.id for t in config.toolchains}
    selected_ids = execution_plan.get("toolchains")
    if not isinstance(selected_ids, list) or not selected_ids:
        selected_ids = [t.id for t in config.toolchains]
    selected_profiles = execution_plan.get("profiles")
    if not isinstance(selected_profiles, list) or not selected_profiles:
        selected_profiles = [p.name for p in config.profiles]
    runs = max(1, int(runs if runs is not None else execution_plan.get(
        "runs_per_build", config.execution.get("runs_per_build", 3))))

    runnable = set(runnable_ids if runnable_ids is not None
                   else [t.id for t in config.runnable_toolchains()])
    expected_keys = {
        build_key(str(tc_id), str(profile))
        for tc_id in selected_ids for profile in selected_profiles
    }
    build_results: Dict[str, BuildResult] = {}
    builds_root = test_dir / "builds"
    for key in sorted(expected_keys):
        tc_id, profile = key.rsplit("__", 1)
        bdir = builds_root / key
        if bdir.is_dir():
            br = compare_build(config, case, tc_id, profile, bdir, runs=runs)
        else:
            br = BuildResult(toolchain=tc_id, profile=profile,
                             status=Status.INCOMPLETE_MATRIX)
        if tc_id not in runnable:
            br.runnable_toolchain = False
            br.status = Status.UNSUPPORTED
        build_results[key] = br

    # Keep stale or out-of-contract directories visible for diagnostics, but
    # never let them create consensus or a false differential.
    for bdir in sorted(builds_root.glob("*__*")):
        if not bdir.is_dir() or bdir.name in expected_keys:
            continue
        tc_id, profile = bdir.name.rsplit("__", 1)
        br = BuildResult(toolchain=tc_id, profile=profile,
                         runnable_toolchain=False, status=Status.UNSUPPORTED)
        if tc_id not in configured_ids:
            br.compile_error = "toolchain is not part of the active matrix"
        else:
            br.compile_error = "profile is not part of the active matrix"
        build_results[bdir.name] = br
    result.builds = build_results

    # Classify per-build state across runnable toolchains.
    unstable = False
    has_compile_fail = False
    has_run_fail = False
    has_timeout = False
    has_incomplete = False
    # Every *stable* build (PASS or ORACLE_MISMATCH) contributes its output hash.
    stable_by_hash: Dict[str, List[str]] = {}
    for key, br in build_results.items():
        tc_id, profile = key.split("__", 1)
        if tc_id not in runnable:
            br.status = Status.UNSUPPORTED
            continue
        if br.status == Status.NONDETERMINISTIC:
            unstable = True
        elif br.status == Status.COMPILE_FAILED:
            has_compile_fail = True
        elif br.status == Status.RUN_TIMEOUT:
            has_timeout = True
        elif br.status == Status.RUN_FAILED:
            has_run_fail = True
        elif br.status == Status.INCOMPLETE_MATRIX:
            has_incomplete = True
        elif br.status in (Status.PASS, Status.ORACLE_MISMATCH) and br.run_output_hashes \
                and br.run_output_hashes[0]:
            stable_by_hash.setdefault(br.run_output_hashes[0], []).append(key)

    oracle_hash = result.oracle_hash
    if unstable:
        result.status = Status.NONDETERMINISTIC
        result.note = "at least one build produced non-deterministic output"
    elif len(stable_by_hash) > 1:
        # Multiple stable builds disagree with each other: the differential we hunt for.
        result.status = Status.POTENTIAL_COMPILER_BUG
        result.diff_detail = {
            "distinct_outputs": {h: keys for h, keys in stable_by_hash.items()},
            "note": "stable builds disagree with each other across the matrix",
        }
        oracle_note = f"oracle_hash={oracle_hash[:16]}" if oracle_hash else "cross-build oracle"
        result.note = (f"{len(stable_by_hash)} distinct stable outputs across builds; "
                       f"{oracle_note}")
    elif has_timeout and len(stable_by_hash) >= 1:
        # Some stable builds completed while others hung: a differential hang
        # (finite loop -> infinite in one ptxas -O path / toolchain) is itself
        # a wrong-code candidate, distinct from a checksum divergence.
        hung = [k for k, br in build_results.items() if br.status == Status.RUN_TIMEOUT]
        ok = [k for k, br in build_results.items()
              if br.status in (Status.PASS, Status.ORACLE_MISMATCH)]
        result.status = Status.HANG_DIVERGENCE
        result.diff_detail = {
            "hung_builds": hung, "ok_builds": ok,
            "note": "some builds hang while others complete (differential hang)",
        }
        result.note = (f"{len(hung)} build(s) timed out while {len(ok)} build(s) "
                       f"completed — differential hang")
    elif len(stable_by_hash) == 1 and case.oracle_mode != "cross_build" \
            and next(iter(stable_by_hash)) != oracle_hash:
        # Oracle disagreement outranks an unrelated missing/failed matrix arm.
        result.status = Status.ORACLE_MISMATCH
        result.note = "all stable builds agree with each other but disagree with the CPU oracle"
    elif has_timeout:
        result.status = Status.RUN_TIMEOUT
    elif has_run_fail:
        result.status = Status.RUN_FAILED
    elif has_compile_fail:
        result.status = Status.COMPILE_FAILED
    elif has_incomplete:
        result.status = Status.INCOMPLETE_MATRIX
    elif len(stable_by_hash) == 1:
        if case.oracle_mode == "cross_build":
            stable_arms = sum(len(keys) for keys in stable_by_hash.values())
            if stable_arms < 2:
                result.status = Status.INCOMPLETE_MATRIX
                result.note = "cross-build comparison requires at least two stable build arms"
            else:
                result.status = Status.PASS
                result.note = "all stable builds agree (cross-build checksum consensus; no CPU oracle)"
        else:
            result.status = Status.PASS
    else:
        result.status = Status.INCOMPLETE_MATRIX

    result.category = classify(result.status)
    return result


def classify(status: str) -> str:
    if status in (Status.POTENTIAL_COMPILER_BUG, Status.ORACLE_MISMATCH,
                  Status.HANG_DIVERGENCE):
        return Category.POTENTIAL_BUG
    if status == Status.NONDETERMINISTIC:
        return Category.FLAKY
    if status == Status.PASS:
        return Category.REDUNDANT          # upgraded to interesting by the stress gate
    if status == Status.COMPILE_FAILED:
        return Category.INVALID
    if status == Status.UNSUPPORTED:
        return Category.UNSUPPORTED
    return Category.INVALID


def first_diff_word(output_bytes: bytes, expected_bytes: bytes) -> Optional[Tuple[int, int, int]]:
    """First word index where outputs differ, with got/expected values (or None)."""
    if len(output_bytes) != len(expected_bytes):
        return None
    n = len(output_bytes) // 8
    g = struct.unpack(f"<{n}Q", output_bytes)
    e = struct.unpack(f"<{n}Q", expected_bytes)
    for i in range(n):
        if g[i] != e[i]:
            return (i, g[i], e[i])
    return None
