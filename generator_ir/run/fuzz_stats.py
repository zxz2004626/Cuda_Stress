"""Run-statistics accumulator for fuzzing campaigns.

Separated from the loop logic so the counter/reporting concerns can evolve
independently of the generation pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class CampaignCounters:
    """Running tallies updated during a fuzz loop."""
    built: int = 0
    compiled_ok: int = 0
    compiled_fail: int = 0
    ran_ok: int = 0
    ran_fail: int = 0
    mutators_tested: int = 0
    hits: int = 0
    time_accumulated_ms: float = 0.0
    begun_at: float = field(default_factory=time.monotonic)

    def note_compile(self, ok: bool) -> None:
        if ok:
            self.compiled_ok += 1
        else:
            self.compiled_fail += 1

    def note_run(self, ok: bool, runtime_ms: float = 0.0) -> None:
        if ok:
            self.ran_ok += 1
            self.time_accumulated_ms += runtime_ms
        else:
            self.ran_fail += 1

    @property
    def compile_pct(self) -> float:
        total = self.compiled_ok + self.compiled_fail
        return self.compiled_ok / total * 100.0 if total else 0.0

    @property
    def run_pct(self) -> float:
        total = self.ran_ok + self.ran_fail
        return self.ran_ok / total * 100.0 if total else 0.0

    @property
    def mean_runtime_ms(self) -> float:
        return self.time_accumulated_ms / self.ran_ok if self.ran_ok else 0.0

    @property
    def wall_seconds(self) -> float:
        return time.monotonic() - self.begun_at

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_generated": self.built,
            "compile_success": self.compiled_ok,
            "compile_fail": self.compiled_fail,
            "compile_rate_pct": round(self.compile_pct, 1),
            "run_success": self.ran_ok,
            "run_fail": self.ran_fail,
            "run_rate_pct": round(self.run_pct, 1),
            "mutations_tested": self.mutators_tested,
            "regressions_found": self.hits,
            "avg_runtime_ms": round(self.mean_runtime_ms, 2),
            "elapsed_s": round(self.wall_seconds, 1),
        }

    def report(self) -> str:
        return (
            f"built {self.built}, assemble {self.compile_pct:.1f}%, "
            f"run {self.run_pct:.1f}%, {self.hits} regressions, "
            f"{self.wall_seconds:.1f}s"
        )

    def progress_line(self, current: int, total: int) -> str:
        pct = current / total * 100 if total else 0
        return (
            f"[{pct:3.0f}%] {current}/{total} | assemble {self.compile_pct:.0f}% "
            f"| run {self.run_pct:.0f}% | avg {self.mean_runtime_ms:.1f}ms "
            f"| hits {self.hits} | {self.wall_seconds:.0f}s"
        )


# Backward-compatible alias.
FuzzerStats = CampaignCounters
