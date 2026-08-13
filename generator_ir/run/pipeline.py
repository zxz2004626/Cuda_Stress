"""Cross-compiler regression harnesses.

``execute_opt_deopt_pipeline`` implements the 4-phase screen-then-confirm
protocol over 6 binaries (src/opt/deopt x old/new compiler).

``execute_simple_cross_pipeline`` compares one (possibly mutated) source across
the two compilers.

Both verify same-source checksum equality between the two compilers before
timing: a mismatch means miscompile/UB, which is reported as its own finding
class (``checksum_mismatch``).
"""

from __future__ import annotations

import json
import logging
import pickle
import statistics
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from generator_ir.run.toolchain import (
    BaseCompiler, CompileResult, RunResult,
    pull_cycles, _kx_same_source_checksum_error, _purge_binaries,
)
from generator_ir.run.detector import (
    RatioOracle, PHASE_BASELINE, PHASE_OPT_GAP, PHASE_DEOPT_GAP,
    PHASE_PASS, PHASE_CHECKSUM_MISMATCH,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timing extraction
# ---------------------------------------------------------------------------

def extract_timing_us(rr: RunResult, use_ci: bool,
                      min_time_us: float = 1000.0) -> Optional[float]:
    """Kernel time in microseconds; None when below the DCE floor.

    The floor applies to BOTH the CI and the simple paths: a kernel whose
    body was optimized away shows near-zero time on either, and such a
    measurement must not feed a ratio.
    """
    if use_ci:
        if rr.ci_mean_us is not None:
            # CI kernels are calibrated per-launch; a 50us floor separates
            # real work from eliminated bodies.
            if rr.ci_mean_us < 50.0:
                return None
            return rr.ci_mean_us
        # fall back to parsing stdout for robustness
        from generator_ir.run.toolchain import parse_ci
        raw = parse_ci(rr.stdout).get("KERNEL_TIME_MEAN_US")
        if raw is not None:
            try:
                v = float(raw)
                return v if v >= 50.0 else None
            except ValueError:
                pass
    if rr.avg_time_ms is not None:
        us = rr.avg_time_ms * 1000.0
        if us < min_time_us:
            return None
        return us
    return None


def _checksum_str(rr: RunResult) -> str:
    return str(rr.checksum) if rr.checksum is not None else ""


def _kx_checksum_mismatches(checksums: Dict[str, str]) -> List[str]:
    out = []
    for pair in ("src", "opt", "deopt"):
        old = checksums.get(f"{pair}_old")
        new = checksums.get(f"{pair}_new")
        if old and new and old != new:
            out.append(f"{pair}: old={old} new={new}")
    return out


def _kx_safe_ratio(num: Optional[float], den: Optional[float]) -> float:
    if num is None or den is None:
        return 1.0
    if den == 0.0:
        return float("inf") if num > 0 else 1.0
    return num / den


def _gap_ratio(t_mut_old, t_src_old, t_mut_new, t_src_new) -> float:
    d_old = _kx_safe_ratio(t_mut_old, t_src_old)
    d_new = _kx_safe_ratio(t_mut_new, t_src_new)
    if d_old == 0.0:
        return 1.0
    return d_new / d_old


def _assemble_one(compiler: BaseCompiler, source: str, opt: str,
                 label: str, out_dir: str) -> Optional[CompileResult]:
    try:
        cr = compiler.assemble(source, opt, output_dir=out_dir, basename=label)
        return cr if cr.success else None
    except Exception as exc:
        log.debug("assemble exception [%s]: %s", label, exc)
        return None


def _execute_one(compiler: BaseCompiler, exe: str, timeout: int = 90) -> Optional[RunResult]:
    try:
        rr = compiler.run(exe, timeout=timeout)
        return rr if rr.success else None
    except Exception as exc:
        log.debug("run exception [%s]: %s", exe, exc)
        return None


# ---------------------------------------------------------------------------
# opt/deopt pipeline result
# ---------------------------------------------------------------------------

@dataclass
class CrossCaseResult:
    seed: int
    template: str
    phase: str = PHASE_PASS
    success: bool = True     # True = no regression found

    src_t_old: Optional[float] = None
    src_t_new: Optional[float] = None
    opt_t_old: Optional[float] = None
    opt_t_new: Optional[float] = None
    deopt_t_old: Optional[float] = None
    deopt_t_new: Optional[float] = None

    baseline_ratio: float = 0.0
    opt_gap_ratio: float = 0.0
    deopt_gap_ratio: float = 0.0

    opt_mutator: str = ""
    deopt_mutator: str = ""
    mutation_mode: str = "full"

    checksum_old: str = ""
    checksum_new: str = ""
    errors: List[str] = field(default_factory=list)


def execute_opt_deopt_pipeline(program, seed: int, template: str, kx_translator,
                           old_compiler: BaseCompiler, new_compiler: BaseCompiler,
                           mutation_pipeline,
                           oracle: Optional[RatioOracle] = None,
                           use_ci_mode: bool = True,
                           use_clock64: bool = False,
                           use_loop: bool = False,
                           opt_level: str = "-O3",
                           perf_runs: int = 3,
                           run_timeout: int = 90,
                           output_dir: Optional[str] = None,
                           baseline_threshold: float = 0.10,
                           gap_threshold: float = 0.15,
                           mutation_mode: str = "full",
                           save_ir: bool = False) -> CrossCaseResult:
    """4-phase cross-compiler regression test for one generated kernel."""
    result = CrossCaseResult(seed=seed, template=template,
                             mutation_mode=mutation_mode)
    # Backend-appropriate artifact extension (fixes the audited always-.hip bug).
    ext = getattr(old_compiler, "extension", None) or ".cu"

    launch = program.launch
    if use_clock64:
        launch.time_clock64 = True
    elif use_loop:
        launch.time_loop = True
    else:
        launch.time_ci = use_ci_mode

    try:
        src_code = kx_translator.rendersrc(program)
    except Exception as exc:
        result.errors.append(f"translation failed: {exc}")
        return result

    # -- opt/deopt mutations ------------------------------------------------
    def _mutate(direction: str):
        try:
            if mutation_mode != "full" and hasattr(mutation_pipeline, "perturb_by_mode"):
                prog, applied = mutation_pipeline.perturb_by_mode(
                    program, mode=mutation_mode, num_mutations=1, paper_only=True)
                name = applied[0] if applied else "none"
            elif hasattr(mutation_pipeline, f"perturb_{direction}_by_mode"):
                prog, applied = getattr(mutation_pipeline,
                                        f"perturb_{direction}_by_mode")(
                    program, mode=mutation_mode, num_mutations=1)
                name = applied[0] if applied else "none"
            else:
                prog = getattr(mutation_pipeline,
                               f"perturb_{direction}_random")(program, paper_only=True)
                name = f"random_{direction}"
            sub = prog.launch
            if use_clock64:
                sub.time_clock64 = True
            elif use_loop:
                sub.time_loop = True
            else:
                sub.time_ci = use_ci_mode
            return prog, kx_translator.rendersrc(prog), name
        except Exception as exc:
            log.debug("%s mutation failed seed=%d: %s", direction, seed, exc)
            result.errors.append(f"{direction} mutation failed: {exc}")
            return None, None, ""

    opt_program, opt_code, opt_name = _mutate("opt")
    deopt_program, deopt_code, deopt_name = _mutate("deopt")
    result.opt_mutator = opt_name
    result.deopt_mutator = deopt_name

    # -- Phase 1: assemble -----------------------------------------------------
    work_dir = output_dir or tempfile.mkdtemp(prefix="gpuperf_harness_")
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    binaries: Dict[str, Optional[CompileResult]] = {}
    binaries["src_old"] = _assemble_one(old_compiler, src_code, opt_level, "src_old", work_dir)
    binaries["src_new"] = _assemble_one(new_compiler, src_code, opt_level, "src_new", work_dir)
    if binaries["src_old"] is None or binaries["src_new"] is None:
        result.errors.append("baseline compilation failed")
        return result
    if opt_code is not None:
        binaries["opt_old"] = _assemble_one(old_compiler, opt_code, opt_level, "opt_old", work_dir)
        binaries["opt_new"] = _assemble_one(new_compiler, opt_code, opt_level, "opt_new", work_dir)
    else:
        binaries["opt_old"] = binaries["opt_new"] = None
    if deopt_code is not None:
        binaries["deopt_old"] = _assemble_one(old_compiler, deopt_code, opt_level, "deopt_old", work_dir)
        binaries["deopt_new"] = _assemble_one(new_compiler, deopt_code, opt_level, "deopt_new", work_dir)
    else:
        binaries["deopt_old"] = binaries["deopt_new"] = None

    # -- Phase 2: screening runs ---------------------------------------------
    timings: Dict[str, Optional[float]] = {}
    checksums: Dict[str, str] = {}
    below_floor: List[str] = []   # ran fine, but timing under the DCE floor
    for label, cr in binaries.items():
        if cr is None or cr.output_path is None:
            timings[label] = None
            continue
        runner = old_compiler if label.endswith("_old") else new_compiler
        rr = _execute_one(runner, cr.output_path, timeout=run_timeout)
        if rr is None:
            timings[label] = None
            result.errors.append(f"run failed: {label}")
            continue
        checksums[label] = _checksum_str(rr)
        if use_clock64:
            # Cycle counts only -- never mix units with µs timings.
            cyc = pull_cycles(rr)
            timings[label] = float(cyc) if cyc is not None else None
        else:
            timings[label] = extract_timing_us(rr, use_ci_mode)
        if timings[label] is None:
            below_floor.append(label)

    result.src_t_old = timings.get("src_old")
    result.src_t_new = timings.get("src_new")
    result.opt_t_old = timings.get("opt_old")
    result.opt_t_new = timings.get("opt_new")
    result.deopt_t_old = timings.get("deopt_old")
    result.deopt_t_new = timings.get("deopt_new")
    result.checksum_old = checksums.get("src_old", "")
    result.checksum_new = checksums.get("src_new", "")

    # Same-source checksum divergence = miscompile/UB finding.
    mismatches = _kx_checksum_mismatches(checksums)
    if mismatches:
        detail = "; ".join(mismatches)
        result.errors.append(f"checksum mismatch -- {detail}")
        result.phase = PHASE_CHECKSUM_MISMATCH
        result.success = False
        if output_dir is not None:
            kx_save_regression_case(result, output_dir, src_code,
                                 opt_code=opt_code, deopt_code=deopt_code,
                                 source_program=program,
                                 opt_program=opt_program,
                                 deopt_program=deopt_program, save_ir=save_ir,
                                 ext=ext)
        return result

    if result.src_t_old is None or result.src_t_new is None:
        if "src_old" in below_floor or "src_new" in below_floor:
            # A fast-but-real kernel whose body may have been optimized away:
            # recorded as a filter diagnostic, not an execution failure.
            result.errors.append(
                "baseline timing below the DCE floor; case filtered")
        else:
            result.errors.append(
                "baseline execution failed on one or both compilers")
        return result

    # -- Phase 3: screen -------------------------------------------------------
    result.baseline_ratio = _kx_safe_ratio(result.src_t_new, result.src_t_old)
    result.opt_gap_ratio = _gap_ratio(result.opt_t_old, result.src_t_old,
                                      result.opt_t_new, result.src_t_new)
    result.deopt_gap_ratio = _gap_ratio(result.deopt_t_old, result.src_t_old,
                                        result.deopt_t_new, result.src_t_new)

    def _screen(baseline_ratio: float, opt_gap: float, deopt_gap: float):
        """(baseline_hit, opt_hit, deopt_hit) under the active decision rule.

        The oracle (ratio-form thresholds) is used when supplied; otherwise
        the simple delta thresholds apply.  Gap checks are one-sided: an
        *improvement* in the new compiler's response is not a regression.
        """
        if oracle is not None:
            verdicts = oracle.full_verify(
                src_t_old=result.src_t_old or 0,
                src_t_new=result.src_t_new or 0,
                opt_t_old=result.opt_t_old, opt_t_new=result.opt_t_new,
                deopt_t_old=result.deopt_t_old, deopt_t_new=result.deopt_t_new,
                opt_mutator=result.opt_mutator,
                deopt_mutator=result.deopt_mutator, seed=seed)
            return (any(v.phase == PHASE_BASELINE for v in verdicts),
                    any(v.phase == PHASE_OPT_GAP for v in verdicts),
                    any(v.phase == PHASE_DEOPT_GAP for v in verdicts))
        return (baseline_ratio > (1.0 + baseline_threshold),
                opt_gap > (1.0 + gap_threshold),
                deopt_gap > (1.0 + gap_threshold))

    baseline_hit, opt_hit, deopt_hit = _screen(
        result.baseline_ratio, result.opt_gap_ratio, result.deopt_gap_ratio)
    # A gap derived from a missing timing is not evidence.  _gap_ratio
    # propagates a missing mut-* value to a neutral 1.0 that, divided by a
    # real d_old < 1, inflates past threshold (the 2.0 case in test_gap_ratio);
    # the oracle path skips None already, but the simple-threshold path does
    # not, so a run failure on the new compiler's mutated kernel would
    # otherwise manufacture an opt/deopt-gap regression.  Only honour a gap
    # hit when both sides actually produced a timing.
    if result.opt_t_old is None or result.opt_t_new is None:
        opt_hit = False
    if result.deopt_t_old is None or result.deopt_t_new is None:
        deopt_hit = False

    if not (baseline_hit or opt_hit or deopt_hit):
        return result

    # -- Phase 4: confirmation ---------------------------------------------------
    log.info("seed=%d screening HIT: baseline=%.3f opt=%.3f deopt=%.3f -- confirming",
             seed, result.baseline_ratio, result.opt_gap_ratio, result.deopt_gap_ratio)

    if use_clock64:
        confirmed = {k: timings.get(k) for k in timings}
    else:
        labels = ["src_old", "src_new"]
        if opt_hit and binaries.get("opt_old") and binaries.get("opt_new"):
            labels += ["opt_old", "opt_new"]
        if deopt_hit and binaries.get("deopt_old") and binaries.get("deopt_new"):
            labels += ["deopt_old", "deopt_new"]
        extra_runs = max(1, perf_runs - 1)
        all_runs: Dict[str, List[float]] = {}
        for label in labels:
            runs = [timings[label]] if timings.get(label) is not None else []
            cr = binaries.get(label)
            if cr is not None and cr.output_path is not None:
                runner = old_compiler if label.endswith("_old") else new_compiler
                for _ in range(extra_runs):
                    rr = _execute_one(runner, cr.output_path, timeout=run_timeout)
                    if rr is not None:
                        t = extract_timing_us(rr, use_ci_mode)
                        if t is not None:
                            runs.append(t)
            all_runs[label] = runs
        confirmed = {}
        for label, runs in all_runs.items():
            if runs:
                confirmed[label] = (statistics.mean(runs) if use_ci_mode
                                    else statistics.median(runs))
            else:
                confirmed[label] = timings.get(label)

    result.src_t_old = confirmed.get("src_old", result.src_t_old)
    result.src_t_new = confirmed.get("src_new", result.src_t_new)
    field_for = {"opt_old": "opt_t_old", "opt_new": "opt_t_new",
                 "deopt_old": "deopt_t_old", "deopt_new": "deopt_t_new"}
    for k, field_name in field_for.items():
        if confirmed.get(k) is not None:
            setattr(result, field_name, confirmed[k])

    result.baseline_ratio = _kx_safe_ratio(result.src_t_new, result.src_t_old)
    result.opt_gap_ratio = _gap_ratio(result.opt_t_old, result.src_t_old,
                                      result.opt_t_new, result.src_t_new)
    result.deopt_gap_ratio = _gap_ratio(result.deopt_t_old, result.src_t_old,
                                        result.deopt_t_new, result.src_t_new)

    # Final verdict with confirmed timings, using the SAME decision rule as
    # screening (oracle when supplied, simple thresholds otherwise), and each
    # phase confirmed only if it was also a screening hit.
    c_base, c_opt, c_deopt = _screen(
        result.baseline_ratio, result.opt_gap_ratio, result.deopt_gap_ratio)
    confirmed_phases: List[str] = []
    if baseline_hit and c_base:
        confirmed_phases.append(PHASE_BASELINE)
    if opt_hit and c_opt:
        confirmed_phases.append(PHASE_OPT_GAP)
    if deopt_hit and c_deopt:
        confirmed_phases.append(PHASE_DEOPT_GAP)

    if confirmed_phases:
        result.phase = confirmed_phases[0]
        result.success = False
        log.info("seed=%d CONFIRMED %s (baseline=%.3f opt=%.3f deopt=%.3f)",
                 seed, result.phase, result.baseline_ratio,
                 result.opt_gap_ratio, result.deopt_gap_ratio)
    else:
        log.info("seed=%d screening hit NOT confirmed", seed)

    if not result.success and output_dir is not None:
        kx_save_regression_case(result, output_dir, src_code,
                             opt_code=opt_code, deopt_code=deopt_code,
                             source_program=program, opt_program=opt_program,
                             deopt_program=deopt_program, save_ir=save_ir,
                             ext=ext)
    _purge_binaries(*binaries.values())
    return result


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def kx_save_regression_case(result: CrossCaseResult, output_dir: str,
                         src_code: str, opt_code: Optional[str] = None,
                         deopt_code: Optional[str] = None,
                         source_program: Any = None,
                         opt_program: Any = None,
                         deopt_program: Any = None,
                         save_ir: bool = False,
                         ext: str = ".cu",
                         mutation_chain: Optional[str] = None) -> str:
    """Persist a regression case (sources + result.json + optional IR pickles)."""
    case_dir = Path(output_dir) / f"seed_{result.seed}_{result.phase}"
    case_dir.mkdir(parents=True, exist_ok=True)

    (case_dir / f"kernel_src{ext}").write_text(src_code)
    if opt_code:
        (case_dir / f"kernel_opt{ext}").write_text(opt_code)
    if deopt_code:
        (case_dir / f"kernel_deopt{ext}").write_text(deopt_code)

    mutated = opt_program if result.phase == PHASE_OPT_GAP else (
        deopt_program if result.phase == PHASE_DEOPT_GAP else None)
    ir_files: List[str] = []
    if save_ir:
        for name, prog in (("source.ir.pkl", source_program),
                           ("mutated.ir.pkl", mutated)):
            if prog is None:
                continue
            try:
                with (case_dir / name).open("wb") as fh:
                    pickle.dump(prog, fh, protocol=pickle.HIGHEST_PROTOCOL)
                ir_files.append(name)
            except Exception as exc:
                log.warning("failed to save IR pickle %s: %s", name, exc)

    report = asdict(result)
    if save_ir:
        report["ir_pickles"] = ir_files
        report["mutation_chain"] = mutation_chain or (
            result.opt_mutator if result.phase == PHASE_OPT_GAP else
            result.deopt_mutator if result.phase == PHASE_DEOPT_GAP else
            "+".join(p for p in (f"opt:{result.opt_mutator}" if result.opt_mutator else "",
                                 f"deopt:{result.deopt_mutator}" if result.deopt_mutator else "")
                     if p))
    (case_dir / "result.json").write_text(json.dumps(report, indent=2, default=str))
    log.info("regression case saved: %s", case_dir)
    return str(case_dir)


# ---------------------------------------------------------------------------
# paired old/new measurement (public helper)
# ---------------------------------------------------------------------------

@dataclass
class PairMeasurementResult:
    """Compile/run result for one source tested across two compilers."""
    compile_success_old: bool = False
    compile_success_new: bool = False
    run_success_old: bool = False
    run_success_new: bool = False
    valid: bool = False
    dry_run: bool = False
    timing_unit: str = "us"
    t_old: Optional[float] = None
    t_new: Optional[float] = None
    ratio: Optional[float] = None
    checksum_old: str = ""
    checksum_new: str = ""
    timings_old: List[float] = field(default_factory=list)
    timings_new: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    old_source_path: str = ""
    new_source_path: str = ""


def time_pair(source_code: str,
                 old_compiler: BaseCompiler,
                 new_compiler: BaseCompiler,
                 opt_level: str = "-O3",
                 work_dir: Optional[str] = None,
                 basename: str = "kernel",
                 perf_runs: int = 3,
                 run_timeout: int = 45,
                 use_ci_mode: bool = True,
                 use_clock64: bool = False,
                 dry_run: bool = False) -> PairMeasurementResult:
    """Compile one source with old/new compilers and measure new/old timing.

    Timings are in microseconds unless clock64 cycle counts are available
    (unit ``cycles``).  In dry-run mode only compilation is validated.
    """
    out_dir = Path(work_dir or tempfile.mkdtemp(prefix="gpuperf_pair_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    result = PairMeasurementResult(dry_run=dry_run)

    cr_old = _assemble_one(old_compiler, source_code, opt_level,
                          f"{basename}_old", str(out_dir))
    cr_new = _assemble_one(new_compiler, source_code, opt_level,
                          f"{basename}_new", str(out_dir))
    result.compile_success_old = cr_old is not None
    result.compile_success_new = cr_new is not None
    if cr_old is not None:
        result.old_source_path = cr_old.source_path
    if cr_new is not None:
        result.new_source_path = cr_new.source_path
    if cr_old is None or cr_new is None:
        result.errors.append(
            f"assemble failed: old={cr_old is not None} new={cr_new is not None}")
        return result
    if dry_run:
        result.valid = True
        _purge_binaries(cr_old, cr_new)
        return result

    actual_runs = 1 if use_clock64 else max(1, int(perf_runs))
    timing_unit = "cycles" if use_clock64 else "us"

    for _ in range(actual_runs):
        rr_old = _execute_one(old_compiler, cr_old.output_path, timeout=run_timeout)
        rr_new = _execute_one(new_compiler, cr_new.output_path, timeout=run_timeout)
        if rr_old is None or rr_new is None:
            result.errors.append("run failed or timed out on one or both compilers")
            _purge_binaries(cr_old, cr_new)
            return result
        result.run_success_old = True
        result.run_success_new = True
        checksum_error = _kx_same_source_checksum_error(rr_old, rr_new)
        if checksum_error is not None:
            result.errors.append(checksum_error)
            _purge_binaries(cr_old, cr_new)
            return result
        result.checksum_old = _checksum_str(rr_old)
        result.checksum_new = _checksum_str(rr_new)

        if use_clock64:
            c_old, c_new = pull_cycles(rr_old), pull_cycles(rr_new)
            if c_old is not None and c_new is not None:
                result.timings_old.append(float(c_old))
                result.timings_new.append(float(c_new))
                timing_unit = "cycles"
                continue
        t_old = extract_timing_us(rr_old, use_ci_mode)
        t_new = extract_timing_us(rr_new, use_ci_mode)
        if t_old is None or t_new is None:
            result.errors.append("no timing data collected (below DCE floor?)")
            _purge_binaries(cr_old, cr_new)
            return result
        result.timings_old.append(t_old)
        result.timings_new.append(t_new)
        timing_unit = "us"

    _purge_binaries(cr_old, cr_new)
    if not result.timings_old or not result.timings_new:
        result.errors.append("no timing data collected")
        return result
    result.timing_unit = timing_unit
    result.t_old = float(statistics.median(result.timings_old))
    result.t_new = float(statistics.median(result.timings_new))
    if result.t_old <= 0:
        result.errors.append("invalid timing (t_old <= 0)")
        return result
    result.ratio = result.t_new / result.t_old
    result.valid = True
    return result


# ---------------------------------------------------------------------------
# simple cross-compiler pipeline
# ---------------------------------------------------------------------------


# Split into simple_cross.py
from generator_ir.run.simple_cross import execute_simple_cross_pipeline  # noqa
