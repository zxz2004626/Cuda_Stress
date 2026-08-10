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
        return ("MISSING", None, 0)
    return (parse_output_blob(data).status,
            parse_output_blob(data).output_words,
            parse_output_blob(data).meaningful_words)


def compare_build(config: Config, case: GeneratedCase, tc_id: str, profile: str,
                  build_dir: Path) -> BuildResult:
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
    runs = config.execution.get("runs_per_build", 3)
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
    expected = (build_dir.parent.parent / "expected.bin").read_bytes()
    expected_hash = hashlib.sha256(expected).hexdigest()
    if br.run_output_hashes[0] == expected_hash:
        br.status = Status.PASS
    else:
        br.status = Status.ORACLE_MISMATCH
    return br


def compare_matrix(config: Config, case: GeneratedCase, test_dir: Path,
                   runnable_ids: Optional[List[str]] = None) -> CaseResult:
    """Compare every build under test_dir and produce the case verdict."""
    result = CaseResult(test_id=case.test_id, strategy=case.strategy,
                        oracle_hash=hashlib.sha256((test_dir / "expected.bin").read_bytes()).hexdigest())
    build_results: Dict[str, BuildResult] = {}
    for bdir in sorted((test_dir / "builds").glob("*__*")):
        if not bdir.is_dir():
            continue
        tc_id, profile = bdir.name.rsplit("__", 1)
        br = compare_build(config, case, tc_id, profile, bdir)
        build_results[build_key(tc_id, profile)] = br
    result.builds = build_results

    runnable = runnable_ids if runnable_ids is not None else [t.id for t in config.runnable_toolchains()]

    # Classify per-build state across runnable toolchains.
    unstable = False
    has_compile_fail = False
    has_run_fail = False
    has_timeout = False
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
        result.note = (f"{len(stable_by_hash)} distinct stable outputs across builds; "
                       f"oracle_hash={oracle_hash[:16]}")
    elif len(stable_by_hash) == 1:
        # All stable builds agree.  Do they agree with the CPU oracle?
        (only_hash, _), = stable_by_hash.items()
        if only_hash == oracle_hash:
            result.status = Status.PASS
        else:
            result.status = Status.ORACLE_MISMATCH
            result.note = "all stable builds agree with each other but disagree with the CPU oracle"
    elif has_timeout:
        result.status = Status.RUN_TIMEOUT
    elif has_run_fail:
        result.status = Status.RUN_FAILED
    elif has_compile_fail:
        result.status = Status.COMPILE_FAILED
    else:
        result.status = Status.INCOMPLETE_MATRIX

    result.category = classify(result.status)
    return result


def classify(status: str) -> str:
    if status == Status.POTENTIAL_COMPILER_BUG or status == Status.ORACLE_MISMATCH:
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
