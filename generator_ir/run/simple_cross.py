"""Simple cross-compiler pipeline: perturb + compare timing.

Split from pipeline.py for modularity.
"""

from __future__ import annotations

import json
import logging
import pickle
import statistics
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from generator_ir.run.toolchain import (
    BaseCompiler, RunResult,
    _purge_binaries, _kx_same_source_checksum_error, pull_cycles,
    _write_assemble_args_once,
)

log = logging.getLogger(__name__)

def execute_simple_cross_pipeline(program, seed: int, template: str, kx_translator,
                              old_compiler: BaseCompiler, new_compiler: BaseCompiler,
                              mutation_pipeline,
                              mutation_mode: str = "full",
                              paper_only: bool = False,
                              use_ci_mode: bool = True,
                              use_clock64: bool = False,
                              use_loop: bool = False,
                              use_profiler: bool = False,
                              profiler_path: Optional[str] = None,
                              backend: str = "hip",
                              threshold: float = 1.25,
                              perf_runs: int = 3,
                              mutations_per_case: int = 2,
                              fresh_mutations: int = 1,
                              output_dir: Optional[str] = None,
                              ext: str = ".hip",
                              skip_mutation: bool = False,
                              save_ir: bool = False) -> Dict[str, Any]:
    """Compile one (mutated or fresh) source with both compilers and compare."""
    result: Dict[str, Any] = {
        "seed": seed, "template": template, "regressions": [],
        "mutations_tested": 0, "original_source": "", "errors": [],
    }

    launch = program.launch
    if use_profiler:
        launch.time_profiler = True
    elif use_clock64:
        launch.time_clock64 = True
    elif use_loop:
        launch.time_loop = True
    else:
        launch.time_ci = use_ci_mode

    try:
        original_source = kx_translator.rendersrc(program)
        result["original_source"] = original_source
    except Exception as exc:
        result["errors"].append(f"translation failed: {exc}")
        return result

    num_mutations = fresh_mutations if skip_mutation else mutations_per_case
    num_mutations = max(0, int(num_mutations))
    mutated_program = None
    applied: List[str] = []

    if num_mutations <= 0:
        mutator_name = "none"
        test_source = original_source
    else:
        try:
            mutated_program, applied = mutation_pipeline.perturb_by_mode(
                program, mode=mutation_mode, num_mutations=num_mutations,
                paper_only=paper_only)
            mutator_name = "+".join(applied) if applied else "none"
        except Exception as exc:
            result["errors"].append(f"mutation failed: {exc}")
            return result
        sub = mutated_program.launch
        if use_profiler:
            sub.time_profiler = True
        elif use_clock64:
            sub.time_clock64 = True
        elif use_loop:
            sub.time_loop = True
        else:
            sub.time_ci = use_ci_mode
        try:
            test_source = kx_translator.rendersrc(mutated_program)
        except Exception as exc:
            result["errors"].append(f"mutated translation failed: {exc}")
            return result
        result["mutated_program"] = mutated_program
        result["mutated_source"] = test_source

    result["mutations_tested"] = num_mutations if num_mutations > 0 else 0
    result["mutations_applied"] = len(applied)
    result["mutator"] = mutator_name
    result["mutations"] = applied

    work_dir = Path(output_dir) if output_dir else Path(
        tempfile.mkdtemp(prefix="gpuperf_simple_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    case_dir = work_dir / f"seed_{seed}_{mutator_name}"
    case_dir.mkdir(parents=True, exist_ok=True)

    cr_old = old_compiler.assemble(test_source, "-O3", str(case_dir / "old"))
    cr_new = new_compiler.assemble(test_source, "-O3", str(case_dir / "new"))
    _write_assemble_args_once(work_dir, cr_old, cr_new, "-O3")

    if not cr_old.success or not cr_new.success:
        result["errors"].append(
            f"assemble failed: old={cr_old.success} new={cr_new.success}")
        return result

    run_timeout = 90
    times_old: List[float] = []
    times_new: List[float] = []
    timing_unit = "ms"

    if use_profiler:
        from generator_ir.run.external_prof import profile_hip, profile_cuda
        # Enforce the same miscompile guard as the direct-run branches: a
        # checksum mismatch on identical source must abort before profiler
        # timings can mask it as a performance regression.
        cs_old = old_compiler.run(cr_old.output_path, timeout=run_timeout)
        cs_new = new_compiler.run(cr_new.output_path, timeout=run_timeout)
        err = _kx_same_source_checksum_error(cs_old, cs_new)
        if err is not None:
            result["errors"].append(err)
            _purge_binaries(cr_old, cr_new)
            return result
        if backend == "hip":
            prof = profiler_path or "rocprofv3"
            fn = lambda exe: profile_hip(exe, profiler=prof, timeout=run_timeout)
        else:
            prof = profiler_path or "nsys"
            fn = lambda exe: profile_cuda(exe, profiler=prof, timeout=run_timeout)
        for _ in range(perf_runs):
            t = fn(cr_old.output_path)
            if t is not None:
                times_old.append(t / 1000.0)  # profiler reports us; pipeline is ms
            t = fn(cr_new.output_path)
            if t is not None:
                times_new.append(t / 1000.0)
    else:
        actual_runs = 1 if use_clock64 else perf_runs

        def kx_measured_ms(rr: RunResult) -> Optional[float]:
            if not rr.success:
                return None
            if use_ci_mode and rr.ci_mean_us is not None:
                mean_us = rr.ci_mean_us
                if mean_us <= 0 or mean_us < 50.0:
                    return None
                return mean_us / 1000.0
            if rr.avg_time_ms is None or rr.avg_time_ms < 1.0:
                return None
            return rr.avg_time_ms

        # One ignored warmup pair before measured repetitions (board settling).
        if use_ci_mode and not use_clock64 and actual_runs > 1:
            so = old_compiler.run(cr_old.output_path, timeout=run_timeout)
            sn = new_compiler.run(cr_new.output_path, timeout=run_timeout)
            err = _kx_same_source_checksum_error(so, sn)
            if err is not None:
                result["errors"].append(err)
                _purge_binaries(cr_old, cr_new)
                return result

        for _ in range(actual_runs):
            rr_old = old_compiler.run(cr_old.output_path, timeout=run_timeout)
            rr_new = new_compiler.run(cr_new.output_path, timeout=run_timeout)
            err = _kx_same_source_checksum_error(rr_old, rr_new)
            if err is not None:
                result["errors"].append(err)
                _purge_binaries(cr_old, cr_new)
                return result

            if use_clock64:
                c_old, c_new = pull_cycles(rr_old), pull_cycles(rr_new)
                if c_old is not None and c_new is not None:
                    times_old.append(float(c_old))
                    times_new.append(float(c_new))
                    timing_unit = "cycles"
                    continue
            t_old, t_new = kx_measured_ms(rr_old), kx_measured_ms(rr_new)
            # Keep old/new paired: both must be present to append.
            if t_old is not None and t_new is not None:
                times_old.append(t_old)
                times_new.append(t_new)

    _purge_binaries(cr_old, cr_new)

    if not times_old or not times_new:
        result["errors"].append(
            "no timing data collected (runs failed or below DCE floor)")
        return result

    t_old = float(statistics.median(times_old))
    t_new = float(statistics.median(times_new))
    if t_old <= 0:
        result["errors"].append("invalid timing (t_old <= 0)")
        return result

    ratio = t_new / t_old
    result["ratio"] = ratio
    result["t_old"] = t_old
    result["t_new"] = t_new

    if ratio > threshold:
        unit = timing_unit
        log.info("REGRESSION: %s seed=%d ratio=%.3f (%.2f%s -> %.2f%s)",
                 mutator_name, seed, ratio, t_old, unit, t_new, unit)

        safe_mut = "".join(c if (c.isalnum() or c in "._-+") else "_"
                           for c in str(mutator_name))[:80]
        if skip_mutation:
            reg_dir = work_dir / f"regression_{seed}"
        else:
            reg_dir = work_dir / f"regression_{seed}_{safe_mut}"
        reg_dir.mkdir(parents=True, exist_ok=True)

        if skip_mutation:
            (reg_dir / f"source_seed_{seed}{ext}").write_text(test_source)
        else:
            (reg_dir / f"original_{safe_mut}{ext}").write_text(original_source)
            (reg_dir / f"mutated_{safe_mut}{ext}").write_text(test_source)

        ir_files: List[str] = []
        if save_ir:
            for name, prog in (("source.ir.pkl", program),
                               ("mutated.ir.pkl", mutated_program)):
                if prog is None:
                    continue
                try:
                    with (reg_dir / name).open("wb") as fh:
                        pickle.dump(prog, fh, protocol=pickle.HIGHEST_PROTOCOL)
                    ir_files.append(name)
                except Exception as exc:
                    log.warning("failed to save IR pickle: %s", exc)

        t_old_key = "t_old_cycles" if unit == "cycles" else "t_old_ms"
        t_new_key = "t_new_cycles" if unit == "cycles" else "t_new_ms"
        report = {
            "seed": seed,
            "mode": "fresh" if skip_mutation else "corpus",
            "mutation": mutator_name,
            "mutation_mode": mutation_mode,
            "opt_level": "-O3",
            "timing_unit": unit,
            t_old_key: t_old,
            t_new_key: t_new,
            "ratio": ratio,
        }
        if save_ir:
            report["ir_pickles"] = ir_files
            report["mutation_chain"] = mutator_name
        (reg_dir / "result.json").write_text(
            json.dumps(report, indent=2, default=str))

        result["regressions"].append({
            "mutator": mutator_name, "ratio": ratio,
            "t_old": t_old, "t_new": t_new,
        })
        log.info("regression saved: %s", reg_dir)

    return result

