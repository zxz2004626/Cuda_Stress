"""Regression oracles.

Two strategies:

1. ``RatioOracle`` -- cross-compiler ratio-of-ratios:
     baseline_ratio = T_new(src) / T_old(src)
     opt_gap_ratio  = (T_new(opt)/T_new(src)) / (T_old(opt)/T_old(src))
     deopt_gap_ratio likewise.  A ratio past its threshold is a regression.

2. ``DifferentialOracle`` -- single-compiler scan/opt-level comparison via
   the benchmark statistics.

Phase names are part of the public contract and shared with the harness:
"baseline_regression" | "opt_gap" | "deopt_gap"  (a single vocabulary used
by both verdict production and consumption -- the audited implementation
lost baseline regressions to a "baseline" vs "baseline_regression" mismatch).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from generator_ir.run.measurement import Benchmarker, RegressionResult

log = logging.getLogger(__name__)

PHASE_BASELINE = "baseline_regression"
PHASE_OPT_GAP = "opt_gap"
PHASE_DEOPT_GAP = "deopt_gap"
PHASE_PASS = "pass"
PHASE_CHECKSUM_MISMATCH = "checksum_mismatch"


@dataclass
class RatioVerdict:
    seed: int
    phase: str
    is_regression: bool
    ratio: float
    threshold: float
    src_t_old: float
    src_t_new: float
    opt_t_old: Optional[float] = None
    opt_t_new: Optional[float] = None
    deopt_t_old: Optional[float] = None
    deopt_t_new: Optional[float] = None
    opt_mutator: str = ""
    deopt_mutator: str = ""
    opt_gap_ratio: float = 0.0
    deopt_gap_ratio: float = 0.0

    def summary(self) -> str:
        tag = "REGRESSION" if self.is_regression else "ok"
        lines = [f"RatioVerdict({tag}, phase={self.phase}, "
                 f"ratio={self.ratio:.3f}, threshold={self.threshold:.2f})",
                 f"  src: old={self.src_t_old:.1f}us new={self.src_t_new:.1f}us"]
        if self.phase == PHASE_OPT_GAP and self.opt_t_old is not None:
            lines.append(
                f"  opt: old={self.opt_t_old:.1f}us new={self.opt_t_new:.1f}us "
                f"gap={self.opt_gap_ratio:.3f} mutator={self.opt_mutator}")
        if self.phase == PHASE_DEOPT_GAP and self.deopt_t_old is not None:
            lines.append(
                f"  deopt: old={self.deopt_t_old:.1f}us new={self.deopt_t_new:.1f}us "
                f"gap={self.deopt_gap_ratio:.3f} mutator={self.deopt_mutator}")
        return "\n".join(lines)


class RatioOracle:
    """Cross-compiler ratio-of-ratios regression oracle."""

    def __init__(self, baseline_threshold: float = 1.25,
                 opt_threshold: float = 1.20,
                 deopt_threshold: float = 1.20) -> None:
        self.baseline_threshold = baseline_threshold
        self.opt_threshold = opt_threshold
        self.deopt_threshold = deopt_threshold

    def verify_baseline(self, src_t_old: float, src_t_new: float,
                       seed: int = 0) -> RatioVerdict:
        if src_t_old <= 0:
            ratio = float("inf") if src_t_new > 0 else 1.0
        else:
            ratio = src_t_new / src_t_old
        return RatioVerdict(
            seed=seed, phase=PHASE_BASELINE,
            is_regression=ratio > self.baseline_threshold,
            ratio=ratio, threshold=self.baseline_threshold,
            src_t_old=src_t_old, src_t_new=src_t_new)

    def verify_opt_gap(self, src_t_old: float, src_t_new: float,
                      opt_t_old: float, opt_t_new: float,
                      mutator: str = "", seed: int = 0) -> RatioVerdict:
        ratio_old = opt_t_old / src_t_old if src_t_old > 0 else 1.0
        ratio_new = opt_t_new / src_t_new if src_t_new > 0 else 1.0
        gap = ratio_new / ratio_old if ratio_old > 0 else 1.0
        cross = opt_t_new / opt_t_old if opt_t_old > 0 else 1.0
        hit = gap > self.opt_threshold or cross > self.baseline_threshold
        return RatioVerdict(
            seed=seed, phase=PHASE_OPT_GAP, is_regression=hit,
            ratio=max(gap, cross), threshold=self.opt_threshold,
            src_t_old=src_t_old, src_t_new=src_t_new,
            opt_t_old=opt_t_old, opt_t_new=opt_t_new,
            opt_mutator=mutator, opt_gap_ratio=gap)

    def verify_deopt_gap(self, src_t_old: float, src_t_new: float,
                        deopt_t_old: float, deopt_t_new: float,
                        mutator: str = "", seed: int = 0) -> RatioVerdict:
        ratio_old = deopt_t_old / src_t_old if src_t_old > 0 else 1.0
        ratio_new = deopt_t_new / src_t_new if src_t_new > 0 else 1.0
        gap = ratio_new / ratio_old if ratio_old > 0 else 1.0
        cross = deopt_t_new / deopt_t_old if deopt_t_old > 0 else 1.0
        hit = gap > self.deopt_threshold or cross > self.baseline_threshold
        return RatioVerdict(
            seed=seed, phase=PHASE_DEOPT_GAP, is_regression=hit,
            ratio=max(gap, cross), threshold=self.deopt_threshold,
            src_t_old=src_t_old, src_t_new=src_t_new,
            deopt_t_old=deopt_t_old, deopt_t_new=deopt_t_new,
            deopt_mutator=mutator, deopt_gap_ratio=gap)

    def full_verify(self, src_t_old: float, src_t_new: float,
                   opt_t_old: Optional[float] = None,
                   opt_t_new: Optional[float] = None,
                   deopt_t_old: Optional[float] = None,
                   deopt_t_new: Optional[float] = None,
                   opt_mutator: str = "", deopt_mutator: str = "",
                   seed: int = 0) -> List[RatioVerdict]:
        """All applicable checks; returns only the regressions."""
        hits: List[RatioVerdict] = []
        base = self.verify_baseline(src_t_old, src_t_new, seed=seed)
        if base.is_regression:
            hits.append(base)
        if opt_t_old is not None and opt_t_new is not None:
            opt = self.verify_opt_gap(src_t_old, src_t_new, opt_t_old,
                                     opt_t_new, mutator=opt_mutator, seed=seed)
            if opt.is_regression:
                hits.append(opt)
        if deopt_t_old is not None and deopt_t_new is not None:
            deopt = self.verify_deopt_gap(src_t_old, src_t_new, deopt_t_old,
                                         deopt_t_new, mutator=deopt_mutator,
                                         seed=seed)
            if deopt.is_regression:
                hits.append(deopt)
        return hits


# ---------------------------------------------------------------------------
# Sweep / differential oracle (single- or cross-compiler benchmark based)
# ---------------------------------------------------------------------------

@dataclass
class SweepPoint:
    param_value: Any
    baseline_time_ms: float = 0.0
    mutated_time_ms: float = 0.0
    delta: float = 1.0
    p_value: float = 1.0
    is_significant: bool = False
    is_regression: bool = False


@dataclass
class SweepResult:
    mutation_name: str
    seed: int
    template: str
    opt_level: str
    points: List[SweepPoint] = field(default_factory=list)
    regression_points: List[SweepPoint] = field(default_factory=list)

    @property
    def kx_has_regression(self) -> bool:
        return bool(self.regression_points)

    @property
    def kx_worst_regression(self) -> Optional[SweepPoint]:
        if not self.regression_points:
            return None
        return max(self.regression_points, key=lambda p: p.delta)

    def summary(self) -> str:
        lines = [f"Sweep: {self.mutation_name} (seed={self.seed}, opt={self.opt_level})",
                 f"  points={len(self.points)} regressions={len(self.regression_points)}"]
        for p in self.regression_points:
            lines.append(f"    param={p.param_value}: base={p.baseline_time_ms:.2f}ms "
                         f"mut={p.mutated_time_ms:.2f}ms delta={p.delta:.3f} p={p.p_value:.4f}")
        return "\n".join(lines)


@dataclass
class DifferentialResult:
    """Result of comparing scan curves between two compiler versions."""
    mutation_name: str
    seed: int
    sweep_old: SweepResult
    sweep_new: SweepResult
    divergence_points: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def kx_has_divergence(self) -> bool:
        return bool(self.divergence_points)


@dataclass
class OracleVerdict:
    """Aggregate verdict for a single generated kernel."""
    seed: int
    template: str
    baseline_valid: bool = False
    num_mutations_tested: int = 0
    regressions: List[RegressionResult] = field(default_factory=list)
    sweeps: List[SweepResult] = field(default_factory=list)
    differentials: List[DifferentialResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def kx_has_regression(self) -> bool:
        return (any(r.is_regression for r in self.regressions)
                or any(s.kx_has_regression for s in self.sweeps)
                or any(d.kx_has_divergence for d in self.differentials))

    @property
    def kx_num_regressions(self) -> int:
        count = sum(1 for r in self.regressions if r.is_regression)
        count += sum(len(s.regression_points) for s in self.sweeps)
        count += sum(len(d.divergence_points) for d in self.differentials)
        return count


class DifferentialOracle:
    """Benchmark-based oracle for single-mutation and scan testing."""

    def __init__(self, benchmark: Benchmarker,
                 regression_threshold: float = 0.05,
                 alpha: float = 0.01) -> None:
        self.benchmark = benchmark
        self.regression_threshold = regression_threshold
        self.alpha = alpha

    def xtest_single(self, original_source: str, mutated_source: str,
                    mutation_name: str, mutation_param: Any = None,
                    seed: int = 0, template: str = "",
                    opt_levels: Optional[List[str]] = None) -> List[RegressionResult]:
        opt_levels = opt_levels or ["-O0", "-O3"]
        out: List[RegressionResult] = []
        for opt in opt_levels:
            base = self.benchmark.bench_single(original_source, opt,
                                               seed=seed, template=template)
            if not base.kx_is_valid:
                out.append(RegressionResult(seed=seed, mutation_name=mutation_name,
                                            mutation_param=mutation_param))
                continue
            mut = self.benchmark.bench_single(mutated_source, opt,
                                              seed=seed, template=template)
            if not mut.kx_is_valid:
                out.append(RegressionResult(seed=seed, mutation_name=mutation_name,
                                            mutation_param=mutation_param))
                continue
            out.append(self.benchmark.find_regression(
                base, mut, mutation_name, mutation_param,
                threshold=self.regression_threshold))
        return out

    def xtest_scan(self, original_source: str,
                   mutated_sources: Dict[Any, str],
                   mutation_name: str, seed: int = 0, template: str = "",
                   opt_level: str = "-O3") -> SweepResult:
        scan = SweepResult(mutation_name=mutation_name, seed=seed,
                            template=template, opt_level=opt_level)
        baseline = self.benchmark.bench_single(original_source, opt_level,
                                               seed=seed, template=template)
        if not baseline.kx_is_valid:
            return scan
        for param, source in mutated_sources.items():
            mutated = self.benchmark.bench_single(source, opt_level,
                                                  seed=seed, template=template)
            point = SweepPoint(param_value=param,
                               baseline_time_ms=baseline.avg_time_ms)
            if not mutated.kx_is_valid:
                scan.points.append(point)
                continue
            point.mutated_time_ms = mutated.avg_time_ms
            point.delta = (mutated.avg_time_ms / baseline.avg_time_ms
                           if baseline.avg_time_ms > 0 else 1.0)
            p, sig = self.benchmark.compare(baseline.all_times_ms,
                                            mutated.all_times_ms,
                                            alpha=self.alpha)
            point.p_value = p
            point.is_significant = sig
            point.is_regression = sig and point.delta > (1.0 + self.regression_threshold)
            scan.points.append(point)
            if point.is_regression:
                scan.regression_points.append(point)
        return scan

    def xtest_cross_opt(self, source: str, seed: int = 0, template: str = "",
                       opt_levels: Optional[List[str]] = None) -> List[RegressionResult]:
        """Flag when a higher opt level is *slower* than the previous one."""
        opt_levels = opt_levels or ["-O0", "-O1", "-O2", "-O3"]
        benches = {opt: self.benchmark.bench_single(source, opt, seed=seed,
                                                    template=template)
                   for opt in opt_levels}
        out: List[RegressionResult] = []
        for lo, hi in zip(opt_levels, opt_levels[1:]):
            b_lo, b_hi = benches[lo], benches[hi]
            if not b_lo.kx_is_valid or not b_hi.kx_is_valid:
                continue
            if b_hi.avg_time_ms > b_lo.avg_time_ms:
                p, sig = self.benchmark.compare(b_lo.all_times_ms, b_hi.all_times_ms)
                ratio = (b_hi.avg_time_ms / b_lo.avg_time_ms
                         if b_lo.avg_time_ms > 0 else 1.0)
                out.append(RegressionResult(
                    seed=seed,
                    mutation_name=f"opt_regression_{lo}_vs_{hi}",
                    baseline_time_ms=b_lo.avg_time_ms,
                    mutated_time_ms=b_hi.avg_time_ms,
                    ratio=ratio,
                    slowdown_pct=(ratio - 1.0) * 100.0,
                    p_value=p, is_significant=sig,
                    is_regression=sig and ratio > (1.0 + self.regression_threshold),
                    baseline_times=b_lo.all_times_ms,
                    mutated_times=b_hi.all_times_ms))
        return out

    def xtest_differential(self, original_source: str,
                          mutated_sources: Dict[Any, str],
                          mutation_name: str,
                          compiler_old, compiler_new,
                          seed: int = 0, template: str = "",
                          opt_level: str = "-O3",
                          divergence_threshold: float = 0.10) -> DifferentialResult:
        """Compare scan curves between two compiler versions.

        A divergence is flagged when the normalised performance ratios
        differ by more than *divergence_threshold* at any scan point.
        """
        bench_old = Benchmarker(compiler_old, num_repeats=self.benchmark.num_repeats,
                                run_timeout=self.benchmark.run_timeout,
                                alpha=self.alpha)
        bench_new = Benchmarker(compiler_new, num_repeats=self.benchmark.num_repeats,
                                run_timeout=self.benchmark.run_timeout,
                                alpha=self.alpha)
        oracle_old = DifferentialOracle(bench_old, self.regression_threshold,
                                        self.alpha)
        oracle_new = DifferentialOracle(bench_new, self.regression_threshold,
                                        self.alpha)
        sweep_old = oracle_old.xtest_scan(
            original_source, mutated_sources, mutation_name,
            seed=seed, template=template, opt_level=opt_level)
        sweep_new = oracle_new.xtest_scan(
            original_source, mutated_sources, mutation_name,
            seed=seed, template=template, opt_level=opt_level)
        result = DifferentialResult(mutation_name=mutation_name, seed=seed,
                                    sweep_old=sweep_old, sweep_new=sweep_new)
        old_deltas = {p.param_value: p.delta for p in sweep_old.points}
        new_deltas = {p.param_value: p.delta for p in sweep_new.points}
        for param in sorted(set(old_deltas) & set(new_deltas), key=str):
            d_old, d_new = old_deltas[param], new_deltas[param]
            diff = abs(d_old - d_new)
            if diff > divergence_threshold:
                result.divergence_points.append({
                    "param": param, "delta_old": d_old,
                    "delta_new": d_new, "divergence": diff,
                })
                log.info("differential divergence: %s param=%s old=%.3f new=%.3f",
                         mutation_name, param, d_old, d_new)
        return result
