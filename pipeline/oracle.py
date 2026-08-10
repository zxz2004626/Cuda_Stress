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
    strat = default_registry().get(case.strategy)
    expected = strat.cpu_oracle(
        input_words=case.input_words, seed=case.kernel_seed, params=case.params,
        input_bytes=case.input_bytes, grid=case.grid, block=case.block,
        output_words_per_thread=case.output_words_per_thread)
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
    strat = default_registry().get(case.strategy)
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
