"""Run archive: results.jsonl ledger, per-testcase artifacts, differential folders and
reproduce.sh scripts.  Also the disk-budget garbage collector."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .config import Config
from .models import CaseResult, Category, Status, ALL_CATEGORIES, PROTECTED_CATEGORIES


def new_run_id(config: Config) -> str:
    import time
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_id = config.generation.get("run_id") or f"run-{ts}-{abs(hash(time.time())) % 10000:04x}"
    return run_id


def environment_json(config: Config) -> Dict[str, Any]:
    import os
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "driver": subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=30).stdout.strip() or "",
        "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30).stdout.strip() or "",
        "toolchains": [t.as_dict() for t in config.toolchains],
        "profiles": [{"name": p.name, "ptxas_opt": p.ptxas_opt} for p in config.profiles],
        "time": datetime.now(timezone.utc).isoformat(),
        "cwd": str(Path.cwd()),
        "env_cuda": {k: os.environ.get(k) for k in ("CUDA_HOME", "CUDA_PATH", "LD_LIBRARY_PATH")},
    }


def init_run(run_dir: Path, config: Config) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "environment.json").write_text(json.dumps(environment_json(config), indent=2))
    snapshot = yaml.safe_dump(_config_snapshot(config), sort_keys=False)
    (run_dir / "config.snapshot.yaml").write_text(snapshot)
    ledger = run_dir / "results.jsonl"
    if not ledger.exists():
        ledger.write_text("")


def _config_snapshot(config: Config) -> Dict[str, Any]:
    return {
        "project": {"root": str(config.project_root)},
        "toolchains": [t.as_dict() for t in config.toolchains],
        "optimization_profiles": [{"name": p.name, "ptxas_opt": p.ptxas_opt} for p in config.profiles],
        "generation": config.generation,
        "compilation": config.compilation,
        "execution": config.execution,
        "stress_gate": config.stress_gate,
        "novelty": config.novelty,
        "storage": config.storage,
    }


def test_dir_for(run_dir: Path, case_id: str) -> Path:
    return run_dir / "tests" / case_id


def write_testcase_artifacts(test_dir: Path, case) -> None:
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / "test.cu").write_text(case.source)
    (test_dir / "input.bin").write_bytes(case.input_bytes)
    (test_dir / "expected.bin").write_bytes(case.expected_bytes)
    (test_dir / "manifest.json").write_text(json.dumps(case.manifest(), indent=2, sort_keys=True))


def append_ledger(run_dir: Path, record: Dict[str, Any]) -> None:
    ledger = run_dir / "results.jsonl"
    with open(ledger, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def write_case_result(run_dir: Path, case_id: str, result: CaseResult) -> None:
    summary = {
        "case_id": case_id,
        "strategy": result.strategy,
        "status": result.status,
        "category": result.category,
        "oracle_hash": result.oracle_hash,
        "novel_score": result.novel_score,
        "note": result.note,
        "diff_detail": result.diff_detail,
        "stress_gate": result.stress_gate,
        "builds": {k: _build_summary(v) for k, v in result.builds.items()},
    }
    d = test_dir_for(run_dir, case_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "result_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    append_ledger(run_dir, {"case_id": case_id, **summary})


def _build_summary(br) -> Dict[str, Any]:
    return {
        "toolchain": br.toolchain, "profile": br.profile,
        "compile_ok": br.compile_ok, "status": br.status,
        "stable": br.stable,
        "run_statuses": br.run_statuses,
        "run_output_hashes": br.run_output_hashes,
        "sass_normalized_hash": br.sass_normalized_hash,
        "sass_instruction_count": br.sass_instruction_count,
        "registers": br.ptxas_registers,
        "metrics": br.metrics,
    }


def write_differential(config: Config, run_dir: Path, case, result: CaseResult) -> Path:
    """Snapshot all artifacts for a confirmed differential into differentials/<id>."""
    diff_dir = run_dir / "differentials" / case.case_id()
    if diff_dir.exists():
        shutil.rmtree(diff_dir)
    diff_dir.mkdir(parents=True)
    src_dir = test_dir_for(run_dir, case.case_id())

    shutil.copy2(src_dir / "manifest.json", diff_dir / "manifest.json")
    shutil.copy2(src_dir / "test.cu", diff_dir / "test.cu")
    shutil.copy2(src_dir / "input.bin", diff_dir / "input.bin")
    shutil.copy2(src_dir / "expected.bin", diff_dir / "expected.bin")

    builds = src_dir / "builds"
    if builds.exists():
        shutil.copytree(builds, diff_dir / "builds")

    summary = {
        "case_id": case.case_id(),
        "strategy": case.strategy,
        "status": result.status,
        "category": result.category,
        "note": result.note,
        "diff_detail": result.diff_detail,
        "generator_version": case.generator_version,
        "kernel_seed": case.kernel_seed,
        "config_seed": case.config_seed,
        "test_seed": case.test_seed,
        "params": case.params,
        "toolchains": [t.as_dict() for t in config.toolchains],
    }
    (diff_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (diff_dir / "reproduce.sh").write_text(make_reproduce_sh(config, case))
    return diff_dir


def make_reproduce_sh(config: Config, case) -> str:
    """A standalone script that regenerates the case and re-runs the full matrix."""
    root = config.project_root
    lines = [
        "#!/usr/bin/env bash",
        "# Regenerate and re-run one differential case.",
        "# Usage: ./reproduce.sh [test_id]",
        "set -euo pipefail",
        f"ROOT={root}",
        f"CASE_ID={case.case_id()}",
        f"STRATEGY={case.strategy}",
        f"TEST_SEED={case.test_seed}",
        f"CONFIG_SEED={case.config_seed}",
        f"PARAMS={json.dumps(case.params)}",
        "",
        "cd \"$ROOT\"",
        f"python3 -m cuda_stress generate --strategy \"$STRATEGY\" --test-seed \"$TEST_SEED\" \\",
        "    --config-seed \"$CONFIG_SEED\" --params \"$PARAMS\" --out /tmp/cuda_stress_repro",
        f"python3 -m cuda_stress run --test /tmp/cuda_stress_repro/{case.case_id()} --runs 3",
        f"python3 -m cuda_stress compare --test /tmp/cuda_stress_repro/{case.case_id()}",
        f"python3 -m cuda_stress report --test /tmp/cuda_stress_repro/{case.case_id()}",
        "",
        "echo \"reproduce complete; inspect the report above.\"",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Disk-budget garbage collection
# ---------------------------------------------------------------------------
def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def run_gc(config: Config, run_dir: Path, state: Dict[str, Any]) -> Dict[str, Any]:
    """Enforce storage budget; never delete protected (bug/flaky) or unique repro material."""
    storage = config.storage
    budget_gb = float(storage.get("max_total_gb", 8))
    budget = int(budget_gb * 1024 * 1024 * 1024)
    corpus_root = config.project_root / "corpus"
    queue_root = config.project_root / "queue"

    sizes = {}
    total = 0
    for name, root in (("corpus", corpus_root), ("runs", run_dir.parent), ("queue", queue_root)):
        if root.exists():
            s = _dir_size(root)
            sizes[name] = s
            total += s

    report = {"budget_bytes": budget, "used_bytes": total, "components": sizes, "deleted": []}
    if total <= budget:
        report["status"] = "under_budget"
        return report

    # Delete redundant per-build artifacts for passing cases first (safe, regenerable).
    tests = run_dir / "tests"
    deleted = 0
    if tests.exists():
        for d in sorted(tests.iterdir(), key=lambda p: -_dir_size(p)):
            if not d.is_dir():
                continue
            result_file = d / "result_summary.json"
            if not result_file.exists():
                continue
            try:
                rs = json.loads(result_file.read_text())
            except Exception:  # noqa: BLE001
                continue
            if rs.get("category") != Category.REDUNDANT:
                continue
            if storage.get("delete_redundant_cubin", True):
                for b in (d / "builds").glob("*__*"):
                    for art in ("kernel.cubin", "kernel.sass", "kernel.ptx"):
                        f = b / art
                        if f.exists():
                            f.unlink()
                            deleted += 1
            report["deleted"].append(f"redundant:{d.name}")

    # Remove passing testcase dirs beyond keep_passing_sources.
    keep = int(storage.get("keep_passing_sources", 5000))
    passing = []
    if tests.exists():
        for d in tests.iterdir():
            rf = d / "result_summary.json"
            if rf.exists():
                try:
                    if json.loads(rf.read_text()).get("category") == Category.REDUNDANT:
                        passing.append(d)
                except Exception:  # noqa: BLE001
                    pass
    for d in sorted(passing, key=lambda p: -_dir_size(p))[keep:]:
        ledger_note = {"case_id": d.name, "action": "gc_delete_redundant", "size_bytes": _dir_size(d)}
        append_ledger(run_dir, ledger_note)
        shutil.rmtree(d, ignore_errors=True)
        report["deleted"].append(f"gc:{d.name}")
        deleted += 1

    # Trim invalid samples per reason.
    keep_invalid = int(storage.get("keep_invalid_samples_per_reason", 20))
    rejected_root = queue_root / "rejected" if queue_root.exists() else None
    if rejected_root and rejected_root.exists():
        by_reason: Dict[str, List[Path]] = {}
        for d in rejected_root.iterdir():
            taskf = d / "task.json"
            if not taskf.exists():
                continue
            try:
                reason = json.loads(taskf.read_text()).get("reject_reason", "unknown")
            except Exception:  # noqa: BLE001
                reason = "unknown"
            by_reason.setdefault(reason, []).append(d)
        for reason, dirs in by_reason.items():
            for d in sorted(dirs, key=lambda p: -_dir_size(p))[keep_invalid:]:
                shutil.rmtree(d, ignore_errors=True)
                report["deleted"].append(f"gc_invalid:{d.name}")

    report["status"] = "cleaned"
    return report
