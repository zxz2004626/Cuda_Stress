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


def _translator_for(backend: str):
    from generator_ir.emit import emitter_for
    return emitter_for(backend)()


def _make_compiler_pair(args):
    """Old/new compiler pair for cross-compiler campaigns."""
    from generator_ir.run.toolchain import create_compiler
    arch = _resolve_arch(args)
    old = create_compiler(backend=args.backend, arch=arch,
                        compiler_path=args.old_compiler)
    new = create_compiler(backend=args.backend, arch=arch,
                        compiler_path=args.new_compiler)
    return old, new


def _kx_requested_observable_mode(args) -> bool:
    return bool(getattr(args, "observable_fold", False)
                or getattr(args, "codegen", False))


def _program_mode_matches(program, args) -> bool:
    """Reject corpus IR only when its observable semantics truly differ."""
    cfg = getattr(program, "launch", None)
    loaded_observable = bool(getattr(cfg, "observable_fold", False))
    if loaded_observable != _kx_requested_observable_mode(args):
        return False
    # Older corpus pickles predate the explicit codegen marker; accept those
    # through the observable compatibility check above.
    if cfg is None or "codegen" not in vars(cfg):
        return not bool(getattr(args, "generic_baseline", False))
    if bool(cfg.codegen) != bool(getattr(args, "codegen", False)):
        return False
    return bool(getattr(cfg, "generic_baseline", False)) == bool(
        getattr(args, "generic_baseline", False))


def _perf_data(result: dict, use_clock64: bool) -> dict:
    """Build the perf_data dict for corpus scoring from a pipeline result."""
    perf_data = {}
    if use_clock64:
        # Bank cycles under the *_time_us key so _rate_sensitivity (which reads
        # old/new_o3_time_us) sees them; its ratio is unit-invariant, matching
        # the opt-deopt path that also banks cycles under the _time_us key.
        if result.get("t_old"):
            perf_data["old_o3_time_us"] = result["t_old"]
        if result.get("t_new"):
            perf_data["new_o3_time_us"] = result["t_new"]
    else:
        if result.get("t_old"):
            perf_data["old_o3_time_us"] = result["t_old"] * 1000
        if result.get("t_new"):
            perf_data["new_o3_time_us"] = result["t_new"] * 1000
    return perf_data


# ---------------------------------------------------------------------------
# kx_generate
# ---------------------------------------------------------------------------



def kx_cmd_fuzz(args: argparse.Namespace) -> int:
    generic = bool(getattr(args, "generic_baseline", False))
    if generic:
        if getattr(args, "templates", None):
            raise ValueError("--generic-baseline cannot be combined with --templates")
        if getattr(args, "codegen", False):
            raise ValueError("--generic-baseline cannot be combined with --codegen")
        if getattr(args, "enable_feature_gens", False):
            raise ValueError(
                "--generic-baseline cannot be combined with --enable-feature-gens")
        if getattr(args, "corpus_dir", None):
            raise ValueError(
                "--generic-baseline accepts fresh cases only; --corpus-dir is not allowed")
        if getattr(args, "old_compiler", None) or getattr(args, "new_compiler", None):
            raise ValueError(
                "generic baseline cross-compiler campaigns belong in the main "
                "Cuda_Stress pipeline; generator_ir fuzz cross modes apply mutations")
        # Mutations encode optimization-aware relationships and would make this
        # arm cease to be a generator-only baseline.
        args.no_mutations = True
    else:
        from generator_ir.gen import weave_blocks as _wb  # noqa: F401  (registration)
        from generator_ir.gen.weave_handlers import apply_feature_gen_weight
        apply_feature_gen_weight(getattr(args, "enable_feature_gens", False))

    old_path = getattr(args, "old_compiler", None)
    new_path = getattr(args, "new_compiler", None)
    if old_path and new_path:
        return _fuzz_cross_compiler(args)
    return _fuzz_single_compiler(args)



def _fuzz_single_compiler(args: argparse.Namespace) -> int:
    from generator_ir.run.campaign import SingleCompilerFuzzer

    fuzzer = SingleCompilerFuzzer(
        output_dir=args.output_dir, backend=args.backend,
        arch=_resolve_arch(args), seed=args.seed,
        num_repeats=args.repeats, run_timeout=args.timeout,
        regression_threshold=args.threshold,
        cleanup_tmp=not args.keep_tmp,
        mutation_mode=getattr(args, "mutation_mode", "full"),
        generic_baseline=getattr(args, "generic_baseline", False),
    )
    stats = fuzzer.kx_fuzz_loop(
        num_seeds=args.seeds, templates=args.templates,
        complexity=args.complexity,
        opt_levels=args.opt_levels or ["-O3"],
        test_mutations=not args.no_mutations,
        corpus_dir=getattr(args, "corpus_dir", None),
        fresh_ratio=getattr(args, "fresh_ratio", 0.3),
    )
    return 0 if stats.regressions_found == 0 else 1



def _fuzz_cross_compiler(args: argparse.Namespace) -> int:
    if getattr(args, "static", False):
        return _fuzz_static(args)
    method = getattr(args, "method", "opt-deopt")
    if method == "simple":
        return _fuzz_simple(args)
    return _fuzz_opt_deopt(args)



def _fuzz_opt_deopt(args: argparse.Namespace) -> int:
    """6-binary opt/deopt ratio-of-ratios cross-compiler fuzzing."""
    from generator_ir.gen.assembler import build_kernel
    from generator_ir.mutate.orchestrator import MutationPipeline
    from generator_ir.run.detector import RatioOracle
    from generator_ir.run.pipeline import execute_opt_deopt_pipeline
    from generator_ir.templates.catalog import kx_build_default_registry

    kx_translator = _translator_for(args.backend)
    old_comp, new_comp = _make_compiler_pair(args)
    oracle = RatioOracle(baseline_threshold=args.baseline_threshold,
                         opt_threshold=args.opt_threshold,
                         deopt_threshold=args.deopt_threshold)
    pipeline = MutationPipeline(seed=args.seed if args.seed is not None else 42,
                                codegen=getattr(args, "codegen", False))
    registry = kx_build_default_registry()
    templates = args.templates or registry.kx_all_names()

    use_loop = getattr(args, "loop", False)
    use_clock64 = getattr(args, "clock64", False)
    use_ci = getattr(args, "ci", False) or use_clock64
    mutation_mode = getattr(args, "mutation_mode", "full")
    stop_on_fail = getattr(args, "stop_on_failure", False)
    corpus_dir = getattr(args, "corpus_dir", None)
    fresh_ratio = getattr(args, "fresh_ratio", 0.3)

    rng_seed = args.seed if args.seed is not None else int(time.time())
    import random as _rng
    rng = _rng.Random(rng_seed)
    regressions = tested = 0

    corpus = None
    if corpus_dir:
        from generator_ir.corpus.bank import CorpusManager
        corpus = CorpusManager(corpus_dir, max_size=500)
        pipeline.kx_set_donor_corpus(corpus)
        log.info("corpus: %s (%d entries)", corpus_dir, corpus.size())

    log.info("cross-compiler fuzzing: old=%s new=%s isa=%s",
             args.old_compiler, args.new_compiler, _resolve_arch(args))

    for i in range(args.seeds):
        from_corpus = (corpus is not None and corpus.size() > 0
                       and rng.random() > fresh_ratio)
        if from_corpus:
            entry = corpus.draw(rng)
            seed, tmpl = entry.seed, entry.template
            config = _make_config(args, seed)
            source_label = f"corpus(id={entry.entry_id})"
        else:
            seed = rng_seed + i
            tmpl = (registry.kx_weighted_random(
                rng, schedule=kx_schedule_for_seed(seed)).name
                if not args.templates else templates[i % len(templates)])
            config = _make_config(args, seed)
            source_label = "fresh"

        if use_clock64:
            config.amplify_iters = 2
        elif use_ci:
            config.amplify_iters = 4

        try:
            program = build_kernel(config, template_name=tmpl)
        except Exception as exc:
            log.warning("seed=%d gen failed: %s", seed, exc)
            continue

        tested += 1
        try:
            result = execute_opt_deopt_pipeline(
                program=program, seed=seed, template=tmpl,
                kx_translator=kx_translator,
                old_compiler=old_comp, new_compiler=new_comp,
                mutation_pipeline=pipeline, oracle=oracle,
                use_ci_mode=use_ci, use_clock64=use_clock64,
                use_loop=use_loop, opt_level="-O3",
                perf_runs=args.perf_runs, output_dir=args.output_dir,
                baseline_threshold=args.baseline_threshold - 1.0,
                gap_threshold=max(args.opt_threshold, args.deopt_threshold) - 1.0,
                mutation_mode=mutation_mode,
                save_ir=getattr(args, "save_ir", False),
            )
        except Exception as exc:
            log.warning("[%d/%d] seed=%d PIPELINE_ERROR: %s",
                        i + 1, args.seeds, seed, exc)
            continue

        status = "REGRESSION" if not result.success else "pass"
        if not result.success:
            regressions += 1
        log.info("[%d/%d] seed=%d tmpl=%s %s baseline=%.3f opt=%.3f deopt=%.3f [%s]",
                 i + 1, args.seeds, seed, tmpl, status,
                 result.baseline_ratio, result.opt_gap_ratio,
                 result.deopt_gap_ratio, source_label)

        if corpus is not None:
            if from_corpus:
                if not result.success:
                    corpus.kx_record_regression(
                        entry.entry_id,
                        mutation=result.opt_mutator or result.deopt_mutator)
            elif result.success:
                chk = result.checksum_old or result.checksum_new or ""
                perf_data = {}
                if result.src_t_old:
                    perf_data["old_o3_time_us"] = result.src_t_old
                if result.src_t_new:
                    perf_data["new_o3_time_us"] = result.src_t_new
                corpus.add(program, seed=seed, template=tmpl,
                           checksum=chk or None, perf_data=perf_data)

        if stop_on_fail and not result.success:
            log.info("stopping on first failure")
            break

    if corpus is not None:
        log.info("corpus: %s", corpus.summary())
    log.info("done: %d tested, %d regressions", tested, regressions)
    return 0 if regressions == 0 else 1



def _fuzz_simple(args: argparse.Namespace) -> int:
    """Simple cross-compiler fuzzing: perturb, assemble both, compare timing."""
    from generator_ir.gen.assembler import build_kernel
    from generator_ir.mutate.orchestrator import MutationPipeline
    from generator_ir.run.pipeline import execute_simple_cross_pipeline
    from generator_ir.templates.catalog import kx_build_default_registry

    for label, path in (("old", args.old_compiler), ("new", args.new_compiler)):
        if not os.path.isfile(path):
            log.error("%s compiler not found: %s", label, path)
            return 1

    kx_translator = _translator_for(args.backend)
    old_comp, new_comp = _make_compiler_pair(args)
    ext = ".hip" if args.backend == "hip" else ".cu"
    pipeline = MutationPipeline(seed=args.seed if args.seed is not None else 42,
                                codegen=getattr(args, "codegen", False))
    registry = kx_build_default_registry()
    templates = args.templates or registry.kx_all_names()

    use_profiler = getattr(args, "profiler_mode", False)
    use_loop = getattr(args, "loop", False)
    use_clock64 = getattr(args, "clock64", False)
    use_ci = getattr(args, "ci", False) or use_clock64
    paper_only = getattr(args, "mutator_set", "all") == "paper"
    mutation_mode = getattr(args, "mutation_mode", "full")
    stop_on_fail = getattr(args, "stop_on_failure", False)
    corpus_dir = getattr(args, "corpus_dir", None)
    fresh_ratio = getattr(args, "fresh_ratio", 0.3)
    min_corpus = getattr(args, "min_corpus_before_mutation", 10)
    threshold = args.baseline_threshold

    rng_seed = args.seed if args.seed is not None else int(time.time())
    import random as _rng
    rng = _rng.Random(rng_seed)
    regressions = tested = 0

    corpus = None
    if corpus_dir:
        from generator_ir.corpus.bank import CorpusManager
        corpus = CorpusManager(corpus_dir, max_size=500)
        pipeline.kx_set_donor_corpus(corpus)
        log.info("corpus: %s (%d entries, fresh %.0f%%)",
                 corpus_dir, corpus.size(), fresh_ratio * 100)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("cross-compiler SIMPLE: old=%s new=%s isa=%s",
             args.old_compiler, args.new_compiler, _resolve_arch(args))

    for i in range(args.seeds):
        use_fresh = True
        corpus_entry = None
        if corpus is not None and corpus.size() > 0:
            if corpus.size() < min_corpus:
                use_fresh = True
            else:
                use_fresh = rng.random() < fresh_ratio
        if not use_fresh:
            corpus_entry = corpus.draw(rng)
            if corpus_entry is None:
                use_fresh = True

        program = None
        if not use_fresh and corpus_entry is not None:
            seed, tmpl = corpus_entry.seed, corpus_entry.template
            config = _make_config(args, seed)
            source_label = f"corpus(id={corpus_entry.entry_id})"
            loaded = corpus.fetch_program(corpus_entry)
            if loaded is not None and _program_mode_matches(loaded, args):
                program = loaded
            elif loaded is not None:
                source_label += ", regenerated(mode_mismatch)"
        else:
            seed = rng_seed + i
            tmpl = (registry.kx_weighted_random(
                rng, schedule=kx_schedule_for_seed(seed)).name
                if not args.templates else templates[i % len(templates)])
            config = _make_config(args, seed)
            source_label = "fresh"
            use_fresh = True

        if use_clock64:
            config.amplify_iters = 2
        elif use_ci:
            config.amplify_iters = 4

        if program is None:
            try:
                program = build_kernel(config, template_name=tmpl)
            except Exception as exc:
                log.warning("seed=%d gen failed: %s", seed, exc)
                continue

        tested += 1
        try:
            result = execute_simple_cross_pipeline(
                program=program, seed=seed, template=tmpl,
                kx_translator=kx_translator,
                old_compiler=old_comp, new_compiler=new_comp,
                mutation_pipeline=pipeline,
                mutation_mode=mutation_mode, paper_only=paper_only,
                use_ci_mode=use_ci, use_clock64=use_clock64,
                use_loop=use_loop, use_profiler=use_profiler,
                profiler_path=getattr(args, "profiler", None),
                backend=args.backend, threshold=threshold,
                perf_runs=args.perf_runs,
                mutations_per_case=args.simple_mutations,
                fresh_mutations=args.simple_fresh_mutations,
                output_dir=str(output_dir), ext=ext,
                skip_mutation=use_fresh,
                save_ir=getattr(args, "save_ir", False),
            )
        except Exception as exc:
            log.warning("[%d/%d] seed=%d PIPELINE_ERROR: %s",
                        i + 1, args.seeds, seed, exc)
            continue

        num_reg = len(result.get("regressions", []))
        errors = result.get("errors", [])
        status = "REGRESSION" if num_reg > 0 else ("ERROR" if errors else "pass")
        regressions += num_reg
        if errors:
            log.warning("[%d/%d] seed=%d tmpl=%s ERROR: %s [%s]",
                        i + 1, args.seeds, seed, tmpl,
                        "; ".join(errors), source_label)
        else:
            log.info("[%d/%d] seed=%d tmpl=%s %s mutations=%d regressions=%d [%s]",
                     i + 1, args.seeds, seed, tmpl, status,
                     result.get("mutations_tested", 0), num_reg, source_label)

        if corpus is not None:
            if not use_fresh and corpus_entry is not None:
                if num_reg > 0:
                    corpus.kx_record_regression(corpus_entry.entry_id,
                                             mutation=result.get("mutator", ""))
                elif not errors:
                    mut_prog = result.get("mutated_program")
                    if mut_prog is not None and result.get("mutator", "none") != "none":
                        corpus.add(
                            program=mut_prog, seed=seed, template=tmpl,
                            source_code=result.get("mutated_source", "")
                            or result.get("original_source", ""),
                            perf_data=_perf_data(result, use_clock64))
            elif not errors and num_reg == 0:
                banked = result.get("mutated_program") or program
                corpus.add(
                    program=banked, seed=seed, template=tmpl,
                    source_code=result.get("mutated_source", "")
                    or result.get("original_source", ""),
                    perf_data=_perf_data(result, use_clock64))

        if stop_on_fail and num_reg > 0:
            log.info("stopping on first failure")
            break

    log.info("done: %d tested, %d regressions (simple mode)", tested, regressions)
    return 0 if regressions == 0 else 1



def _fuzz_static(args: argparse.Namespace) -> int:
    """Static-oracle fuzzing (assemble-time metrics only, no execution)."""
    from generator_ir.gen.assembler import build_kernel
    from generator_ir.mutate.orchestrator import MutationPipeline
    from generator_ir.run.static_analysis import execute_static_pipeline
    from generator_ir.templates.catalog import kx_build_default_registry

    kx_translator = _translator_for(args.backend)
    old_comp, new_comp = _make_compiler_pair(args)
    pipeline = MutationPipeline(seed=args.seed if args.seed is not None else 42,
                                codegen=getattr(args, "codegen", False))
    registry = kx_build_default_registry()
    templates = args.templates or registry.kx_all_names()
    mutation_mode = getattr(args, "mutation_mode", "full")
    stop_on_fail = getattr(args, "stop_on_failure", False)

    rng_seed = args.seed if args.seed is not None else int(time.time())
    import random as _rng
    rng = _rng.Random(rng_seed)
    regressions = tested = 0

    log.info("static oracle fuzzing: old=%s new=%s isa=%s",
             args.old_compiler, args.new_compiler, _resolve_arch(args))

    for i in range(args.seeds):
        seed = rng_seed + i
        tmpl = (registry.kx_weighted_random(
            rng, schedule=kx_schedule_for_seed(seed)).name
            if not args.templates else templates[i % len(templates)])
        config = _make_config(args, seed)
        try:
            program = build_kernel(config, template_name=tmpl)
        except Exception as exc:
            log.warning("seed=%d gen failed: %s", seed, exc)
            continue

        tested += 1
        try:
            result = execute_static_pipeline(
                program=program, seed=seed, template=tmpl,
                kx_translator=kx_translator,
                old_compiler=old_comp, new_compiler=new_comp,
                mutation_pipeline=pipeline,
                output_dir=args.output_dir, opt_level="-O3",
                mutation_mode=mutation_mode)
        except Exception as exc:
            log.warning("[%d/%d] seed=%d PIPELINE_ERROR: %s",
                        i + 1, args.seeds, seed, exc)
            continue

        status = "REGRESSION" if not result.success else "pass"
        if not result.success:
            regressions += 1
        log.info("[%d/%d] seed=%d tmpl=%s %s phase=%s src_old=%s src_new=%s",
                 i + 1, args.seeds, seed, tmpl, status, result.phase,
                 result.src_old, result.src_new)
        if stop_on_fail and not result.success:
            break

    log.info("done: %d tested, %d regressions (static)", tested, regressions)
    return 0 if regressions == 0 else 1


# ---------------------------------------------------------------------------
# perturb
# ---------------------------------------------------------------------------
