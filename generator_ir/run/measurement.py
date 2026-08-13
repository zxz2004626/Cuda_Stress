"""Benchmark orchestration and statistical comparison (Mann-Whitney U)."""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

try:
    from scipy import stats as _sp_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


@dataclass
class BenchResult:
    """Timing + metadata for one kernel variant."""
    seed: int
    template: str
    opt_level: str
    avg_time_ms: float = 0.0
    std_time_ms: float = 0.0
    min_time_ms: float = 0.0
    max_time_ms: float = 0.0
    all_times_ms: List[float] = field(default_factory=list)
    checksum: Optional[int] = None
    compile_success: bool = False
    run_success: bool = False
    compile_time_s: float = 0.0
    source_path: str = ""

    @property
    def kx_is_valid(self) -> bool:
        return self.compile_success and self.run_success and self.avg_time_ms > 0


@dataclass
class RegressionResult:
    """Baseline-vs-mutated comparison outcome."""
    seed: int
    mutation_name: str
    mutation_param: Any = None
    baseline_time_ms: float = 0.0
    mutated_time_ms: float = 0.0
    ratio: float = 1.0
    slowdown_pct: float = 0.0
    p_value: float = 1.0
    is_significant: bool = False
    is_regression: bool = False
    baseline_times: List[float] = field(default_factory=list)
    mutated_times: List[float] = field(default_factory=list)


class Benchmarker:
    """Compile + repeatedly run variants, then compare distributions."""

    def __init__(self, compiler, num_repeats: int = 5,
                 run_timeout: int = 300, alpha: float = 0.01) -> None:
        self.compiler = compiler
        self.num_repeats = num_repeats
        self.run_timeout = run_timeout
        self.alpha = alpha

    def bench_single(self, source: str, opt_level: str,
                     seed: int = 0, template: str = "",
                     output_dir = None) -> 'BenchResult':
        return self._profile_single_opt_real(source, opt_level, seed, template, output_dir)

    def _profile_single_opt_real(self, source: str, opt_level: str,
                     seed: int = 0, template: str = "",
                     output_dir: Optional[str] = None) -> BenchResult:
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="gpuperf_bench_")
        cr = self.compiler.assemble(source, opt_level, output_dir)
        result = BenchResult(seed=seed, template=template, opt_level=opt_level,
                             compile_success=cr.success,
                             compile_time_s=cr.compile_time_s,
                             source_path=cr.source_path)
        if not cr.success:
            return result

        times: List[float] = []
        checksum: Optional[int] = None
        for _ in range(self.num_repeats):
            rr = self.compiler.run(cr.output_path, timeout=self.run_timeout)
            if not rr.success:
                continue
            if rr.all_times_ms:
                times.extend(rr.all_times_ms)
            elif rr.avg_time_ms is not None:
                times.append(rr.avg_time_ms)
            if rr.checksum is not None:
                if checksum is not None and rr.checksum != checksum:
                    log.warning("checksum changed across runs seed=%d", seed)
                checksum = rr.checksum

        if not times:
            return result
        arr = np.array(times)
        result.run_success = True
        result.all_times_ms = times
        result.avg_time_ms = float(np.mean(arr))
        result.std_time_ms = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        result.min_time_ms = float(np.min(arr))
        result.max_time_ms = float(np.max(arr))
        result.checksum = checksum
        return result

    def bench_opt_levels(self, source: str, opt_levels: List[str],
                             seed: int = 0, template: str = "") -> Dict[str, BenchResult]:
        """Benchmark one source at each optimization level."""
        out: Dict[str, BenchResult] = {}
        for opt in opt_levels:
            out_dir = tempfile.mkdtemp(
                prefix=f"gpuperf_bench_{opt.replace('-', '')}_")
            out[opt] = self._profile_single_opt_real(source, opt, seed=seed,
                                                     template=template,
                                                     output_dir=out_dir)
        return out

    # Backward-compatible alias.
    _profile_all_opts = bench_opt_levels

    def find_opt_regression(self, source: str, seed: int = 0,
                              opt_levels: Tuple[str, str] = ("-O0", "-O3"),
                              threshold: float = 0.05) -> RegressionResult:
        """Same source, low vs high opt level; regression = the higher opt
        level is significantly *slower* by more than ``threshold``."""
        lo, hi = opt_levels
        b_lo = self._profile_single_opt_real(source, lo, seed=seed)
        b_hi = self._profile_single_opt_real(source, hi, seed=seed)
        return self.find_regression(b_lo, b_hi, f"opt_regression_{lo}_vs_{hi}",
                                      threshold=threshold)

    def compare(self, times_a: List[float], times_b: List[float],
                alpha: Optional[float] = None) -> Tuple[float, bool]:
        """Two-sided Mann-Whitney U; returns (p_value, significant)."""
        if not _HAS_SCIPY:
            raise ImportError("scipy is required for statistical comparison")
        if alpha is None:
            alpha = self.alpha
        if len(times_a) < 3 or len(times_b) < 3:
            return 1.0, False
        try:
            _stat, p = _sp_stats.mannwhitneyu(times_a, times_b,
                                              alternative="two-sided")
        except ValueError:
            return 1.0, False
        return float(p), p < alpha

    def find_regression(self, baseline: BenchResult, mutated: BenchResult,
                          mutation_name: str, mutation_param: Any = None,
                          threshold: float = 0.05) -> RegressionResult:
        """Regression = significant AND mutated slower by > threshold."""
        result = RegressionResult(seed=baseline.seed,
                                  mutation_name=mutation_name,
                                  mutation_param=mutation_param)
        if not baseline.kx_is_valid or not mutated.kx_is_valid:
            return result
        result.baseline_time_ms = baseline.avg_time_ms
        result.mutated_time_ms = mutated.avg_time_ms
        result.baseline_times = baseline.all_times_ms
        result.mutated_times = mutated.all_times_ms
        if baseline.avg_time_ms > 0:
            result.ratio = mutated.avg_time_ms / baseline.avg_time_ms
            result.slowdown_pct = (result.ratio - 1.0) * 100.0
        p, sig = self.compare(baseline.all_times_ms, mutated.all_times_ms)
        result.p_value = p
        result.is_significant = sig
        result.is_regression = sig and result.ratio > (1.0 + threshold)
        return result

    # Backward-compatible alias.
    _compare_for_slowdown = find_regression
