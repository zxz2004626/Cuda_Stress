"""Load and save IR artifacts for the reducer."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

from generator_ir.ir.program import Program
from generator_ir.ir.traversal import child_nodes, traverse


class IrLoadError(Exception):
    pass


def load_program_from_dir(case_dir: Path) -> Program:
    """Load a Program from <case_dir>, preferring the archived pickle.

    The pickle is a fast exact path, not a long-term stable format.  If it is
    missing or unreadable the generator_ir strategy is used as a pure-function
    rebuild path; the rebuilt source must match the stored test.cu byte-for-byte.
    """
    case_dir = Path(case_dir)
    manifest = _read_json(case_dir / "manifest.json")
    generation = _read_json(case_dir / "generation.json")
    strategy = manifest.get("strategy") or generation.get("strategy")
    execution_mode = manifest.get("execution_mode") or generation.get("execution_mode")
    if strategy != "generator_ir":
        raise IrLoadError(
            f"case strategy is {strategy!r}; only generator_ir cases are supported")
    if execution_mode != "driver_checksum":
        raise IrLoadError(
            f"case execution_mode is {execution_mode!r}; reducer requires driver_checksum")

    pkl_path = case_dir / "program.pkl"
    if pkl_path.exists():
        try:
            with open(pkl_path, "rb") as f:
                program = pickle.load(f)
            if isinstance(program, Program):
                _check_emit_matches(program, case_dir, pkl_path)
                return program
        except Exception as exc:  # noqa: BLE001
            # Fall through to the deterministic rebuild path; do not trust a
            # corrupt/version-mismatched pickle silently.
            pass

    # Rebuild from seeds/params.  generator_ir.generate is pure and now also
    # carries the IR blob, so this reconstructs the exact Program.
    from generator.base import default_registry
    info = generation if generation is not None else manifest
    if info is None:
        raise IrLoadError(
            f"no manifest.json or generation.json in {case_dir}; cannot rebuild IR")
    try:
        case = default_registry().get("generator_ir").generate(
            config_seed=int(info["config_seed"]),
            test_seed=int(info["test_seed"]),
            params=dict(info.get("strategy_params") or {}),
            gen_version=str(info["generator_version"]),
            target_sm=str(manifest.get("target_sm") or info.get("target_sm")),
        )
    except Exception as exc:  # noqa: BLE001
        raise IrLoadError(
            f"program.pkl unreadable and seed rebuild failed: {exc}. "
            "Provide a valid program.pkl or a reproducible generator_ir case."
        ) from exc

    blob = getattr(case, "ir_blob", None)
    if not blob:
        raise IrLoadError(
            "seed rebuild succeeded but generated case did not carry ir_blob; "
            "please regenerate with the current generator_ir adapter")
    try:
        program = pickle.loads(blob)
    except Exception as exc:  # noqa: BLE001
        raise IrLoadError(f"seed rebuild produced an unreadable ir_blob: {exc}") from exc

    _check_emit_matches(program, case_dir, None)
    return program


def _check_emit_matches(program: Program, case_dir: Path, pkl_path: Optional[Path]) -> None:
    from generator_ir.emit.nvidia import CudaEmitter
    stored = (case_dir / "test.cu").read_text()
    emitted = CudaEmitter().rendersrc(program)
    if emitted != stored:
        src = str(pkl_path) if pkl_path is not None else "seed rebuild"
        raise IrLoadError(
            f"IR drift: emitted source from {src} does not match {case_dir / 'test.cu'}"
        )


def save_reduced_program(out_dir: Path, program: Program) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "program.pkl").write_bytes(
        pickle.dumps(program, protocol=pickle.HIGHEST_PROTOCOL))
    try:
        (out_dir / "program.ir.txt").write_text(program_to_text(program))
    except Exception:  # noqa: BLE001
        pass


def program_to_text(program: Program) -> str:
    lines: List[str] = []
    for node in traverse(program):
        lines.append(_fmt_node(node))
        for child in child_nodes(node):
            lines.append(f"    child: {_fmt_node(child)}")
    return "\n".join(lines) + "\n"


def _fmt_node(node) -> str:
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


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None
