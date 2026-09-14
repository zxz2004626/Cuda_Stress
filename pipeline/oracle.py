"""CPU-oracle verification: recompute expected output from the manifest + input, and
check input sensitivity (the runtime input really influences the output)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from generator.base import default_registry
from .manifest import case_from_dir


def recompute_expected(test_dir: Path) -> Dict[str, Any]:
    """Recompute expected.bin from test.cu/input.bin via the strategy's CPU oracle."""
    case = case_from_dir(test_dir)
    if case.oracle_mode == "cross_build":
        raise ValueError(
            f"{case.strategy} has no CPU oracle; use cross-build checksum comparison")
    strat = default_registry().get(case.strategy)
    kwargs: Dict[str, Any] = {}
    if case.strategy == "generator_ir":
        # generator_ir rebuilds its program from normalized params and needs
        # the original target SM for arch-gated intrinsics.
        kwargs["target_sm"] = case.target_sm
    expected = strat.cpu_oracle(
        input_words=case.input_words, seed=case.kernel_seed, params=case.params,
        input_bytes=case.input_bytes, grid=case.grid, block=case.block,
        output_words_per_thread=case.output_words_per_thread, **kwargs)
    return {
        "recomputed_sha256": hashlib.sha256(expected).hexdigest(),
        "stored_sha256": hashlib.sha256(case.expected_bytes).hexdigest(),
        "match": hashlib.sha256(expected).hexdigest() == hashlib.sha256(case.expected_bytes).hexdigest(),
        "bytes": expected,
    }


def oracle_audit(test_dir: Path, fix: bool = False) -> Dict[str, Any]:
    res = recompute_expected(test_dir)
    out = {"oracle_recompute": {k: v for k, v in res.items() if k != "bytes"},
           "status": "OK" if res["match"] else "ORACLE_REGEN_MISMATCH"}
    if fix and not res["match"]:
        (test_dir / "expected.bin").write_bytes(res["bytes"])
        out["status"] = "FIXED"
    return out


def verify_input_sensitivity(test_dir: Path, config_seed: int) -> Dict[str, Any]:
    """Regenerate input.bin with a tweaked seed; recompute expected; ensure it changed.

    Guards against candidates whose output does not actually depend on the input
    (which would make them near-zero-value for differential testing).
    """
    case = case_from_dir(test_dir)
    if case.oracle_mode == "cross_build":
        raise ValueError(
            "cross-build cases require verify_runtime_input_sensitivity(config, test_dir)")
    strat = default_registry().get(case.strategy)
    if case.execution_mode == "driver_checksum":
        # generator_ir checksum cases derive their input from the recorded
        # distribution internally; the salted-oracle rerun is the analogous
        # CPU-side sensitivity check (the GPU side has its own input-salt run).
        kwargs: Dict[str, Any] = {}
        if case.strategy == "generator_ir":
            kwargs["target_sm"] = case.target_sm
        base = strat.cpu_oracle(
            input_words=case.input_words, seed=case.kernel_seed,
            params=case.params, input_bytes=case.input_bytes, grid=case.grid,
            block=case.block, output_words_per_thread=case.output_words_per_thread,
            input_salt=None, **kwargs)
        salted = strat.cpu_oracle(
            input_words=case.input_words, seed=case.kernel_seed,
            params=case.params, input_bytes=case.input_bytes, grid=case.grid,
            block=case.block, output_words_per_thread=case.output_words_per_thread,
            input_salt=1, **kwargs)
        return {
            "input_sensitive": base != salted,
            "base_sha256": hashlib.sha256(base).hexdigest(),
            "salted_sha256": hashlib.sha256(salted).hexdigest(),
        }
    # A different input, derived deterministically from the same seeds + 1.
    rng = strat._new_rng(config_seed, case.test_seed + 1)
    from generator.strategies.helpers import make_input_words, pack_words
    words = make_input_words(rng, case.input_words, case.params.get("boundary_words", 8))
    alt_input = pack_words(words)
    alt_expected = strat.cpu_oracle(
        input_words=case.input_words, seed=case.kernel_seed, params=case.params,
        input_bytes=alt_input, grid=case.grid, block=case.block,
        output_words_per_thread=case.output_words_per_thread)
    sensitive = alt_expected != case.expected_bytes
    return {
        "input_sensitive": sensitive,
        "alt_input_sha256": hashlib.sha256(alt_input).hexdigest(),
        "alt_expected_sha256": hashlib.sha256(alt_expected).hexdigest(),
    }


def verify_runtime_input_sensitivity(config, test_dir: Path,
                                     input_salt: int = 1) -> Dict[str, Any]:
    """Re-run one stable checksum build with deterministically changed input.

    This is not presented as a CPU oracle.  It only verifies that the checksum
    observed by the differential pipeline actually depends on runtime input.
    """
    from . import executor

    case = case_from_dir(test_dir)
    if case.execution_mode not in {"standalone_checksum", "driver_checksum"}:
        raise ValueError("runtime input sensitivity is only defined for checksum cases")

    for build_dir in sorted((test_dir / "builds").glob("*__*")):
        if not build_dir.is_dir():
            continue
        try:
            tc_id, _profile = build_dir.name.rsplit("__", 1)
            tc = config.find_toolchain(tc_id)
        except (ValueError, KeyError):
            continue
        if not tc.runnable:
            continue
        try:
            compile_result = json.loads((build_dir / "compile_result.json").read_text())
        except Exception:  # noqa: BLE001
            continue
        if not compile_result.get("ok"):
            continue

        try:
            baseline = executor.parse_output_blob(
                (build_dir / "output_run_1.bin").read_bytes())
            salted = executor.parse_output_blob(
                (build_dir / "output_run_input_sensitivity.bin").read_bytes())
        except OSError:
            continue
        if baseline.status != "OK" or baseline.output_words is None:
            continue
        if salted.status != "OK" or salted.output_words is None:
            raise RuntimeError(
                f"input-sensitivity data missing for {build_dir.name}: "
                f"{salted.status} {salted.error}")

        baseline_hash = hashlib.sha256(baseline.output_words).hexdigest()
        changed_hash = hashlib.sha256(salted.output_words).hexdigest()
        result = {
            "method": f"{case.execution_mode}_input_salt",
            "representative_build": build_dir.name,
            "input_salt": int(input_salt) & 0xFFFFFFFF,
            "input_sensitive": baseline.output_words != salted.output_words,
            "baseline_output_sha256": baseline_hash,
            "salted_output_sha256": changed_hash,
        }
        (test_dir / "input_sensitivity.json").write_text(
            json.dumps(result, indent=2, sort_keys=True))
        return result

    raise RuntimeError("no successful runnable checksum build for input-sensitivity check")
