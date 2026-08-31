"""gpuperf command-line interface.

Subcommands: kx_generate / fuzz / perturb / scan / benchmark / list.

Timing-mode flags (--ci / --clock64 / --loop / --profiler-mode) select the
benchmark harness emitted into each kernel.  Cross-compiler fuzzing compares
an old and a new compiler (--old-compiler / --new-compiler) with three
methods: opt-deopt (default 6-binary ratio-of-ratios), simple, and static.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

from generator_ir.gen.weights import kx_schedule_for_seed

log = logging.getLogger("gpuperf")

DEFAULT_ARCH = {"cuda": "sm_75", "hip": "gfx908"}


def _resolve_arch(args) -> str:
    """Backend-aware arch default (fixes the audited bug where a CUDA run
    without --arch silently inherited the HIP default gfx908)."""
    arch = getattr(args, "arch", None)
    if arch:
        return arch
    return DEFAULT_ARCH.get(getattr(args, "backend", "hip"), "gfx908")


def _kx_diversity_profile(seed: int, backend: str, enabled: bool) -> dict:
    """Deterministic per-seed diversity knobs for CUDA codegen campaigns."""
    if not enabled or backend != "cuda":
        return {}
    import random
    r = random.Random(seed ^ 0x5EEDD1F0)
    return {
        "prob_float_pool": 1.0 if r.random() < 0.40 else 0.0,
        "prob_raw_div": r.choice([0.0, 0.0, 0.5, 0.8]),
        "prob_wide_consts": r.choice([0.0, 0.4, 0.7]),
        "input_distribution": r.choice(["hash", "hash", "zeros", "small",
                                        "random", "adversarial"]),
        "randomize_launch": r.random() < 0.5,
    }


def _make_config(args, seed: int):
    from generator_ir.gen.options import GenConfig
    diversity = _kx_diversity_profile(seed, args.backend,
                                   getattr(args, "codegen", False))
    if getattr(args, "vary_launch", False):
        diversity["randomize_launch"] = True
    return GenConfig(
        seed=seed,
        complexity=getattr(args, "complexity", "medium"),
        target=args.backend,
        isa=_resolve_arch(args),
        observable_fold=getattr(args, "observable_fold", False),
        allow_volatile=getattr(args, "allow_volatile", False),
        codegen=getattr(args, "codegen", False),
        generic_baseline=getattr(args, "generic_baseline", False),
        **diversity,
    )


def _apply_timing_mode(args, program, config) -> None:
    """Set the harness timing mode flags on the generated program."""
    if getattr(args, "clock64", False):
        program.launch.time_clock64 = True
    elif getattr(args, "loop", False):
        program.launch.time_loop = True
    elif getattr(args, "ci", False):
        program.launch.time_ci = True


def _cap_repeat_for_timing(args, config) -> None:
    """CI/clock64 harnesses batch many launches per process, so per-launch
    work must stay small (the fuzz driver's per-mode caps)."""
    if getattr(args, "clock64", False):
        config.amplify_iters = min(config.amplify_iters, 2)
    elif getattr(args, "ci", False):
        config.amplify_iters = min(config.amplify_iters, 4)


def _translator_for(backend: str):
    from generator_ir.emit import emitter_for
    return emitter_for(backend)()


# ---------------------------------------------------------------------------
# kx_generate
# ---------------------------------------------------------------------------



def cmd_build(args: argparse.Namespace) -> int:
    from generator_ir.gen.assembler import build_kernel
    generic = bool(getattr(args, "generic_baseline", False))
    if generic and getattr(args, "template", None) is not None:
        raise ValueError("--generic-baseline cannot be combined with --template")
    if generic and getattr(args, "codegen", False):
        raise ValueError("--generic-baseline cannot be combined with --codegen")
    if generic and getattr(args, "enable_feature_gens", False):
        raise ValueError("--generic-baseline cannot be combined with --enable-feature-gens")
    if not generic:
        from generator_ir.gen import weave_blocks as _wb  # noqa: F401  (registration)
        from generator_ir.gen.weave_handlers import apply_feature_gen_weight
        apply_feature_gen_weight(getattr(args, "enable_feature_gens", False))

    os.makedirs(args.output_dir, exist_ok=True)
    kx_translator = _translator_for(args.backend)
    ext = ".hip" if args.backend == "hip" else ".cu"

    generated = failed = 0
    for i in range(args.count):
        seed = args.seed + i
        config = _make_config(args, seed)
        _cap_repeat_for_timing(args, config)
        try:
            program = build_kernel(config, template_name=args.template)
            _apply_timing_mode(args, program, config)
            source = kx_translator.rendersrc(program)
        except Exception as exc:
            log.error("generation failed seed=%d: %s", seed, exc)
            failed += 1
            continue
        out_path = os.path.join(args.output_dir, f"kernel_{seed}{ext}")
        with open(out_path, "w") as fh:
            fh.write(source)
        generated += 1
        print(f"Generated: {out_path} ({len(source)} bytes)")

    print(f"\nDone: {generated} generated, {failed} failed")
    return 0 if failed == 0 else 1


# ---------------------------------------------------------------------------
# fuzz
# ---------------------------------------------------------------------------


def cmd_perturb(args: argparse.Namespace) -> int:
    from generator_ir.gen.assembler import build_kernel
    from generator_ir.mutate.orchestrator import MutationPipeline

    kx_translator = _translator_for(args.backend)
    ext = ".hip" if args.backend == "hip" else ".cu"
    os.makedirs(args.output_dir, exist_ok=True)

    # Force "high" at construction so GenConfig.__post_init__ applies the rich
    # preset (setting it after has no effect: budgets stay at the default).
    args.complexity = "high"
    config = _make_config(args, args.seed)
    _cap_repeat_for_timing(args, config)
    program = build_kernel(config)
    _apply_timing_mode(args, program, config)

    pipeline = MutationPipeline(seed=args.seed)
    try:
        mutated = pipeline.perturb_by_name(program, args.mutation)
    except Exception as exc:
        log.error("mutation %s failed: %s", args.mutation, exc)
        return 1

    orig_path = os.path.join(args.output_dir, f"original_{args.seed}{ext}")
    with open(orig_path, "w") as fh:
        fh.write(kx_translator.rendersrc(program))
    print(f"Original: {orig_path}")

    mut_path = os.path.join(args.output_dir,
                            f"mutated_{args.mutation}_{args.seed}{ext}")
    with open(mut_path, "w") as fh:
        fh.write(kx_translator.rendersrc(mutated))
    print(f"Mutated:  {mut_path}")
    return 0


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------


def cmd_scan(args: argparse.Namespace) -> int:
    from generator_ir.gen.assembler import build_kernel
    from generator_ir.mutate.orchestrator import MutationPipeline
    from generator_ir.run.toolchain import create_compiler
    from generator_ir.run.measurement import Benchmarker
    from generator_ir.run.detector import DifferentialOracle

    kx_translator = _translator_for(args.backend)
    os.makedirs(args.output_dir, exist_ok=True)

    # Force "high" at construction so GenConfig.__post_init__ applies the rich
    # preset (setting it after has no effect: budgets stay at the default).
    args.complexity = "high"
    config = _make_config(args, args.seed)
    _cap_repeat_for_timing(args, config)
    program = build_kernel(config)
    _apply_timing_mode(args, program, config)
    original_source = kx_translator.rendersrc(program)

    pipeline = MutationPipeline(seed=args.seed)
    try:
        sweep_programs = pipeline.scan_one(program, args.mutation)
    except Exception as exc:
        log.error("cannot scan %s: %s", args.mutation, exc)
        return 1

    mutated_sources = {}
    for i, mp in enumerate(sweep_programs):
        try:
            mutated_sources[i] = kx_translator.rendersrc(mp)
        except Exception as exc:
            log.warning("scan variant %d translation failed: %s", i, exc)
    if not mutated_sources:
        log.error("no scan variants produced")
        return 1

    compiler = create_compiler(backend=args.backend, arch=_resolve_arch(args))
    benchmark = Benchmarker(compiler, num_repeats=5)
    oracle = DifferentialOracle(benchmark)
    opt = args.opt_level or "-O3"
    result = oracle.xtest_scan(original_source, mutated_sources,
                               args.mutation, seed=args.seed, opt_level=opt)
    print(result.summary())

    result_path = os.path.join(args.output_dir,
                               f"sweep_{args.mutation}_{args.seed}.json")
    with open(result_path, "w") as fh:
        json.dump({
            "mutation": args.mutation, "seed": args.seed, "opt_level": opt,
            "num_points": len(result.points),
            "num_regressions": len(result.regression_points),
            "points": [{
                "param": str(p.param_value),
                "baseline_ms": p.baseline_time_ms,
                "mutated_ms": p.mutated_time_ms,
                "delta": p.delta, "p_value": p.p_value,
                "is_regression": p.is_regression,
            } for p in result.points],
        }, fh, indent=2)
    print(f"Saved: {result_path}")
    return 0 if not result.kx_has_regression else 1


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


def cmd_bench(args: argparse.Namespace) -> int:
    from generator_ir.run.toolchain import create_compiler
    from generator_ir.run.measurement import Benchmarker

    if not os.path.isfile(args.source):
        log.error("source file not found: %s", args.source)
        return 1
    with open(args.source) as fh:
        source = fh.read()

    if args.source.endswith(".hip"):
        backend = "hip"
    elif args.source.endswith(".cu"):
        backend = "cuda"
    else:
        backend = args.backend or "hip"
    arch = args.arch or DEFAULT_ARCH[backend]

    compiler = create_compiler(backend=backend, arch=arch)
    benchmark = Benchmarker(compiler, num_repeats=args.repeats)
    opt_levels = args.opt_levels or ["-O0", "-O3"]

    print(f"Benchmarking: {args.source}")
    print(f"Backend: {backend}, arch: {arch}, opts: {opt_levels}")
    results = benchmark.bench_opt_levels(source, opt_levels)
    for opt, res in results.items():
        if res.kx_is_valid:
            print(f"  {opt}: avg={res.avg_time_ms:.2f}ms "
                  f"std={res.std_time_ms:.2f}ms "
                  f"min={res.min_time_ms:.2f}ms max={res.max_time_ms:.2f}ms "
                  f"checksum={res.checksum}")
        else:
            status = "compile_fail" if not res.compile_success else "run_fail"
            print(f"  {opt}: {status}")

    if len(opt_levels) >= 2:
        r0 = results.get(opt_levels[0])
        r1 = results.get(opt_levels[-1])
        if r0 and r1 and r0.kx_is_valid and r1.kx_is_valid and r0.avg_time_ms > 0:
            print(f"\n  Speedup ({opt_levels[-1]} vs {opt_levels[0]}): "
                  f"{r0.avg_time_ms / r1.avg_time_ms:.2f}x")
    return 0


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def kx_cmd_list(args: argparse.Namespace) -> int:
    if args.what in ("mutations", "all"):
        print("Available mutations:")
        try:
            from generator_ir.mutate.orchestrator import MutationPipeline
            pipeline = MutationPipeline(seed=42)
            for name in sorted(pipeline.fetch_mutation_names()):
                print(f"  - {name}")
            print(f"  Total: {len(pipeline.fetch_mutation_names())}")
            parked = [m.name for m in pipeline.parked_mutations]
            if parked:
                print(f"  (parked/disabled: {', '.join(parked)})")
        except Exception as exc:
            print(f"  (mutation module unavailable: {exc})")
        if args.what == "all":
            print()
    if args.what in ("templates", "all"):
        print("Available templates:")
        try:
            from generator_ir.templates.catalog import kx_build_default_registry
            registry = kx_build_default_registry()
            for name in sorted(registry.kx_all_names()):
                print(f"  - {name}")
            print(f"  Total: {len(registry.kx_all_names())}")
        except Exception as exc:
            print(f"  (template module unavailable: {exc})")
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
