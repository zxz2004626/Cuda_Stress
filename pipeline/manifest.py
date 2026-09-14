"""Standard testcase directory format: test.cu + manifest.json + input/expected bins.

A testcase dir is the atomic unit of work in the queue and the archive:
    <test_dir>/
        test.cu
        manifest.json
        input.bin
        expected.bin
        generation.json
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

from generator.base import GeneratedCase


def write_testcase_dir(test_dir: Path, case: GeneratedCase, generator_version: str,
                       config_seed: int) -> Dict[str, Any]:
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / "test.cu").write_text(case.source)
    (test_dir / "input.bin").write_bytes(case.input_bytes)
    (test_dir / "expected.bin").write_bytes(case.expected_bytes)
    oracle_source = getattr(case, "oracle_source", None)
    if oracle_source:
        # Standalone CPU oracle (generator_ir cpu_oracle mode): the C twin of
        # test.cu whose u64 checksum is expected.bin (8 bytes LE).
        (test_dir / "oracle_cpu.c").write_text(oracle_source)
    ir_blob = getattr(case, "ir_blob", None)
    if ir_blob is not None:
        (test_dir / "program.pkl").write_bytes(ir_blob)
        _write_program_ir_text(test_dir / "program.ir.txt", ir_blob)
    manifest = case.manifest()
    (test_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    gen = {
        "generator_version": generator_version,
        "config_seed": config_seed,
        "test_seed": case.test_seed,
        "strategy": case.strategy,
        "strategy_params": case.params,
        "execution_mode": case.execution_mode,
        "oracle_mode": case.oracle_mode,
    }
    (test_dir / "generation.json").write_text(json.dumps(gen, indent=2, sort_keys=True))
    return manifest


def _write_program_ir_text(path: Path, ir_blob: bytes) -> None:
    """Best-effort human-readable IR dump; never fails the case write."""
    try:
        program = pickle.loads(ir_blob)
        from generator_ir.ir.traversal import child_nodes, traverse
        lines: list = []
        for node in traverse(program):
            lines.append(_fmt_ir_node(node))
            for child in child_nodes(node):
                lines.append(f"    child: {_fmt_ir_node(child)}")
        path.write_text("\n".join(lines) + "\n")
    except Exception:  # noqa: BLE001
        path.unlink(missing_ok=True)


def _fmt_ir_node(node) -> str:
    if node is None:
        return "<None>"
    name = type(node).__name__
    try:
        if hasattr(node, "name") and isinstance(getattr(node, "name"), str):
            return f"{name}(name={getattr(node, 'name')!r})"
        if hasattr(node, "stmts") and isinstance(getattr(node, "stmts"), list):
            return f"{name}(stmts={len(getattr(node, 'stmts'))})"
        if hasattr(node, "op") and hasattr(node, "value"):
            return f"{name}(op={getattr(node, 'op')!r}, value={getattr(node, 'value')!r})"
        if hasattr(node, "op"):
            return f"{name}(op={getattr(node, 'op')!r})"
        if hasattr(node, "ty") and hasattr(node, "value"):
            return f"{name}(ty={getattr(node, 'ty')!r}, value={getattr(node, 'value')!r})"
    except Exception:  # noqa: BLE001
        pass
    return name


def read_testcase_dir(test_dir: Path) -> Dict[str, Any]:
    m = json.loads((test_dir / "manifest.json").read_text())
    m["_dir"] = str(test_dir)
    return m


def verify_hashes(test_dir: Path) -> Dict[str, bool]:
    import hashlib
    src = (test_dir / "test.cu").read_text()
    inp = (test_dir / "input.bin").read_bytes()
    exp = (test_dir / "expected.bin").read_bytes()
    return {
        "source": hashlib.sha256(src.encode()).hexdigest() == manifest_hash(test_dir, "source"),
        "input": hashlib.sha256(inp).hexdigest() == manifest_hash(test_dir, "input"),
        "expected": hashlib.sha256(exp).hexdigest() == manifest_hash(test_dir, "expected"),
    }


def manifest_hash(test_dir: Path, which: str) -> str:
    m = json.loads((test_dir / "manifest.json").read_text())
    h = m.get("hashes", {}).get(which, "")
    return h.split(":", 1)[1] if ":" in h else h


def case_from_dir(test_dir: Path) -> GeneratedCase:
    """Rebuild a GeneratedCase from a testcase dir (for reproduce / re-run)."""
    m = json.loads((test_dir / "manifest.json").read_text())
    return GeneratedCase(
        test_id=m["test_id"], config_seed=m["config_seed"], test_seed=m["test_seed"],
        strategy=m["strategy"], params=m["strategy_params"],
        generator_version=m["generator_version"], kernel_seed=m.get("kernel_seed", 0),
        source=(test_dir / "test.cu").read_text(),
        input_bytes=(test_dir / "input.bin").read_bytes(),
        expected_bytes=(test_dir / "expected.bin").read_bytes(),
        grid=m["grid"], block=m["block"], input_words=m["input_word_count"],
        output_words=m["output_word_count"], output_words_per_thread=m.get("output_words_per_thread", 2),
        target_sm=m["target_sm"], cpu_oracle=m["cpu_oracle"],
        comparison_policy=m["comparison_policy"], cuda_features=m.get("cuda_features", []),
        kernel_entry=m.get("kernel_entry", "test_kernel"),
        sanitizer_policy=m.get("sanitizer_policy", "none"),
        valid=bool(m.get("valid", True)), reject_reason=m.get("reject_reason", ""),
        execution_mode=m.get("execution_mode", "driver_kernel"),
        oracle_mode=m.get("oracle_mode", "cpu_exact"),
        execution_spec=m.get("execution_spec", {}),
        oracle_note=m.get("cpu_oracle_note", ""),
        oracle_source=(test_dir / "oracle_cpu.c").read_text()
        if (test_dir / "oracle_cpu.c").exists() else None,
    )
