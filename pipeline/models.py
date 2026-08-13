"""Shared data models: statuses, classifications, and the manifest schema."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Outcome statuses (per testcase / per build).  Order matters for reporting.
# ---------------------------------------------------------------------------
class Status:
    PASS = "PASS"
    POTENTIAL_COMPILER_BUG = "POTENTIAL_COMPILER_BUG"
    HANG_DIVERGENCE = "HANG_DIVERGENCE"
    ORACLE_MISMATCH = "ORACLE_MISMATCH"
    NONDETERMINISTIC = "NONDETERMINISTIC"
    COMPILE_FAILED = "COMPILE_FAILED"
    RUN_FAILED = "RUN_FAILED"
    RUN_TIMEOUT = "RUN_TIMEOUT"
    INCOMPLETE_MATRIX = "INCOMPLETE_MATRIX"
    UNSUPPORTED = "UNSUPPORTED"
    INVALID_TEST = "INVALID_TEST"


ALL_STATUSES = [
    Status.PASS,
    Status.POTENTIAL_COMPILER_BUG,
    Status.HANG_DIVERGENCE,
    Status.ORACLE_MISMATCH,
    Status.NONDETERMINISTIC,
    Status.COMPILE_FAILED,
    Status.RUN_FAILED,
    Status.RUN_TIMEOUT,
    Status.INCOMPLETE_MATRIX,
    Status.UNSUPPORTED,
    Status.INVALID_TEST,
]


# ---------------------------------------------------------------------------
# Corpus classification (what to do with a testcase after triage).
# ---------------------------------------------------------------------------
class Category:
    INTERESTING = "interesting"
    POTENTIAL_BUG = "potential_bug"
    FLAKY = "flaky"
    REDUNDANT = "redundant"
    INVALID = "invalid"
    UNSUPPORTED = "unsupported"


ALL_CATEGORIES = [
    Category.INTERESTING,
    Category.POTENTIAL_BUG,
    Category.FLAKY,
    Category.REDUNDANT,
    Category.INVALID,
    Category.UNSUPPORTED,
]

# Never garbage-collect these categories.
PROTECTED_CATEGORIES = {Category.POTENTIAL_BUG, Category.FLAKY}


@dataclass
class BuildResult:
    """Outcome of one (toolchain x ptxas-profile) build of a testcase."""

    toolchain: str
    profile: str
    compile_ok: bool = False
    compile_error: Optional[str] = None
    runnable_toolchain: bool = True
    run_statuses: List[str] = field(default_factory=list)  # per-repetition status
    run_output_hashes: List[str] = field(default_factory=list)
    run_output_files: List[str] = field(default_factory=list)
    stable: bool = False          # all repetitions produced identical outputs
    status: str = Status.INCOMPLETE_MATRIX
    sass_normalized_hash: Optional[str] = None
    sass_instruction_count: int = 0
    ptxas_registers: Optional[int] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseResult:
    """Full per-testcase verdict across the matrix."""

    test_id: str
    strategy: str
    status: str = Status.INCOMPLETE_MATRIX
    category: str = Category.REDUNDANT
    oracle_hash: Optional[str] = None
    builds: Dict[str, BuildResult] = field(default_factory=dict)  # key f"{toolchain}__{profile}"
    diff_detail: Dict[str, Any] = field(default_factory=dict)
    novel_score: float = 0.0
    stress_gate: Dict[str, Any] = field(default_factory=dict)
    note: str = ""
