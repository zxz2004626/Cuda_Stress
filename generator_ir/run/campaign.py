"""Single-compiler fuzzer loop (kx_generate -> assemble -> run -> perturb-test)."""

from __future__ import annotations

import copy
import json
import logging
import random
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from generator_ir.run.toolchain import create_compiler
from generator_ir.run.measurement import Benchmarker
from generator_ir.run.detector import DifferentialOracle

log = logging.getLogger(__name__)


@dataclass
class FuzzerStats:
    total_generated: int = 0
    compile_success: int = 0
    compile_fail: int = 0
    run_success: int = 0
    run_fail: int = 0
    mutations_tested: int = 0
    regressions_found: int = 0
    runtime_sum_ms: float = 0.0
    start_time: float = field(default_factory=time.monotonic)

    @property
    def assemble_rate(self) -> float:
        total = self.compile_success + self.compile_fail
        return self.compile_success / total * 100.0 if total else 0.0

    @property
    def execute_rate(self) -> float:
        total = self.run_success + self.run_fail
        return self.run_success / total * 100.0 if total else 0.0

    @property
    def kx_avg_runtime_ms(self) -> float:
        return self.runtime_sum_ms / self.run_success if self.run_success else 0.0

    @property
    def kx_elapsed_s(self) -> float:
        return time.monotonic() - self.start_time

    def kx_to_dict(self) -> Dict[str, Any]:
        return {
            "total_generated": self.total_generated,
            "compile_success": self.compile_success,
            "compile_fail": self.compile_fail,
            "compile_rate_pct": round(self.assemble_rate, 1),
            "run_success": self.run_success,
            "run_fail": self.run_fail,
            "run_rate_pct": round(self.execute_rate, 1),
            "mutations_tested": self.mutations_tested,
            "regressions_found": self.regressions_found,
            "avg_runtime_ms": round(self.kx_avg_runtime_ms, 2),
            "elapsed_s": round(self.kx_elapsed_s, 1),
        }


class SingleCompilerFuzzer:
    """Fuzz one compiler across opt levels + mutations."""

    def __init__(self, output_dir: str = "./fuzz_output", backend: str = "hip",
                 arch: Optional[str] = None, seed: Optional[int] = None,
                 compiler_path: Optional[str] = None, num_repeats: int = 5,
                 run_timeout: int = 300, regression_threshold: float = 0.05,
                 alpha: float = 0.01, cleanup_tmp: bool = True,
                 mutation_mode: str = "full", diversity: bool = True) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target = backend
        self.seed = seed if seed is not None else int(time.time()) % (2 ** 31)
        self.cleanup_tmp = cleanup_tmp
        self.mutation_mode = mutation_mode
        self.diversity = diversity
        if arch is None:
            arch = "gfx908" if backend == "hip" else "sm_75"
        self.compiler = create_compiler(backend=backend, arch=arch,
                                      compiler_path=compiler_path)
        self.benchmark = Benchmarker(self.compiler, num_repeats=num_repeats,
                                     run_timeout=run_timeout, alpha=alpha)
        self.oracle = DifferentialOracle(self.benchmark, regression_threshold,
                                         alpha)
        self._pipeline = None
        self.stats = FuzzerStats()

    @property
    def pipeline(self):
        if self._pipeline is None:
            from generator_ir.mutate.orchestrator import MutationPipeline
            self._pipeline = MutationPipeline(seed=self.seed)
        return self._pipeline

    @property
    def kx_translator(self):
        from generator_ir.emit import emitter_for
        return emitter_for(self.target)()

    def _kx_diversity_profile(self, seed: int) -> Dict[str, Any]:
        """Per-seed diversity knobs (CUDA codegen campaigns only)."""
        if not self.diversity or self.target != "cuda":
            return {}
        r = random.Random(seed ^ 0x5EEDD1F0)
        return {
            "prob_float_pool": 1.0 if r.random() < 0.40 else 0.0,
            "prob_raw_div": r.choice([0.0, 0.0, 0.5, 0.8]),
            "prob_wide_consts": r.choice([0.0, 0.4, 0.7]),
            "input_distribution": r.choice(["hash", "hash", "zeros", "small",
                                            "random", "adversarial"]),
            "randomize_launch": r.random() < 0.5,
        }

    # ------------------------------------------------------------------
    # one seed
    # ------------------------------------------------------------------

    def build_and_xtest(self, config=None, test_mutations: bool = True,
                          opt_levels: Optional[List[str]] = None) -> Dict[str, Any]:
        from generator_ir.gen.options import GenConfig
        from generator_ir.gen.assembler import build_kernel

        if config is None:
            seed = self.seed + self.stats.total_generated
            config = GenConfig(seed=seed, target=self.target,
                               isa=self.compiler.isa,
                               **self._kx_diversity_profile(seed))
        opt_levels = opt_levels or ["-O3"]

        result: Dict[str, Any] = {"seed": config.seed, "status": "unknown",
                                  "regressions": [], "errors": []}
        try:
            program = build_kernel(config)
        except Exception as exc:
            result["status"] = "gen_fail"
            result["errors"].append(f"generation failed: {exc}")
            return result
        try:
            source = self.kx_translator.rendersrc(program)
        except Exception as exc:
            result["status"] = "translate_fail"
            result["errors"].append(f"translation failed: {exc}")
            return result
        result["source_len"] = len(source)

        tmp = tempfile.mkdtemp(prefix="gpuperf_")
        cr = self.compiler.assemble(source, opt_levels[0], output_dir=tmp)
        if not cr.success:
            self.stats.compile_fail += 1
            result["status"] = "compile_fail"
            result["errors"].append(cr.stderr[:500])
            if self.cleanup_tmp:
                self.compiler.cleanup(tmp)
            return result
        self.stats.compile_success += 1

        rr = self.compiler.run(cr.output_path, timeout=self.benchmark.run_timeout)
        if not rr.success:
            self.stats.run_fail += 1
            result["status"] = "run_fail"
            result["errors"].append(rr.stderr[:500])
            if self.cleanup_tmp:
                self.compiler.cleanup(tmp)
            return result
        self.stats.run_success += 1
        if rr.avg_time_ms is not None:
            self.stats.runtime_sum_ms += rr.avg_time_ms
        # If the kernel runs too fast, the measurement is unreliable; note it.
        if rr.avg_time_ms is not None and rr.avg_time_ms < 100:
            result["warning"] = f"runtime too fast ({rr.avg_time_ms:.1f}ms)"
        result["status"] = "success"
        result["checksum"] = rr.checksum
        result["avg_time_ms"] = rr.avg_time_ms
        if self.cleanup_tmp:
            self.compiler.cleanup(tmp)

        if test_mutations:
            try:
                regressions = self._xtest_mutations(program, source, config,
                                                   opt_levels)
                result["regressions"] = regressions
                result["num_regressions"] = len(regressions)
            except Exception as exc:
                result["errors"].append(f"mutation testing failed: {exc}")
                log.debug("mutation testing failed: %s", traceback.format_exc())
        return result

    def _xtest_mutations(self, program, original_source: str, config,
                        opt_levels: List[str]) -> List[Dict[str, Any]]:
        regressions: List[Dict[str, Any]] = []
        try:
            names = self.pipeline.fetch_mutations_by_mode(self.mutation_mode)
        except Exception:
            return regressions
        for name in names:
            self.stats.mutations_tested += 1
            try:
                mutated = self.pipeline.perturb_by_name(program, name)
                mutated_source = self.kx_translator.rendersrc(mutated)
            except Exception:
                continue
            for opt in opt_levels:
                for reg in self.oracle.xtest_single(
                        original_source, mutated_source, name,
                        seed=config.seed, opt_levels=[opt]):
                    if reg.is_regression:
                        self.stats.regressions_found += 1
                        regressions.append({
                            "mutation": name, "opt_level": opt,
                            "mutation_mode": self.mutation_mode,
                            "baseline_ms": reg.baseline_time_ms,
                            "mutated_ms": reg.mutated_time_ms,
                            "ratio": reg.ratio, "p_value": reg.p_value,
                            "original_source": original_source,
                            "mutated_source": mutated_source,
                        })
        return regressions

    # ------------------------------------------------------------------
    # benchmark an existing source file
    # ------------------------------------------------------------------

    def bench_source(self, source_path: str,
                         opt_levels: Optional[List[str]] = None):
        """Benchmark a single source file at multiple opt levels."""
        opt_levels = opt_levels or ["-O0", "-O3"]
        with open(source_path, "r") as fh:
            source = fh.read()
        return self.benchmark.bench_opt_levels(
            source, opt_levels, seed=0, template="user")

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------

    def kx_fuzz_loop(self, num_seeds: int = 100,
                  templates: Optional[List[str]] = None,
                  complexity: str = "high",
                  opt_levels: Optional[List[str]] = None,
                  test_mutations: bool = True,
                  corpus_dir: Optional[str] = None,
                  fresh_ratio: float = 0.3) -> FuzzerStats:
        from generator_ir.gen.options import GenConfig

        opt_levels = opt_levels or ["-O3"]
        corpus = None
        if corpus_dir:
            from generator_ir.corpus.bank import CorpusManager
            corpus = CorpusManager(corpus_dir, max_size=500)
            self.pipeline.kx_set_donor_corpus(corpus)
            log.info("corpus enabled: %s (%d entries)", corpus_dir, corpus.size())

        rng = random.Random(self.seed)
        self.stats = FuzzerStats()

        for i in range(num_seeds):
            use_corpus = (corpus is not None and corpus.size() > 0
                          and rng.random() > fresh_ratio)
            if use_corpus:
                result = self._kx_corpus_iteration(corpus, complexity, opt_levels,
                                                test_mutations, rng=rng)
            else:
                config = GenConfig(seed=self.seed + i, complexity=complexity,
                                   amplify_iters=2048,
                                   target=self.target,
                                   isa=self.compiler.isa)
                if templates:
                    config.template_name = templates[i % len(templates)]
                try:
                    result = self.build_and_xtest(
                        config, test_mutations=test_mutations,
                        opt_levels=opt_levels)
                except Exception as exc:
                    result = {"status": "error", "errors": [str(exc)]}
                if corpus and result.get("status") == "success":
                    self._kx_add_to_corpus(corpus, config, result)

            self.stats.total_generated += 1
            if result.get("regressions"):
                self._kx_save_regression(result.get("seed", self.seed + i), result)
            if (i + 1) % 10 == 0 or i == 0:
                self._progress(i + 1, num_seeds)

        self._final_stats()
        (self.output_dir / "stats.json").write_text(
            json.dumps(self.stats.kx_to_dict(), indent=2))
        if corpus:
            log.info("corpus summary:\n%s", corpus.summary())
        return self.stats

    def _kx_corpus_iteration(self, corpus, complexity: str,
                          opt_levels: List[str],
                          test_mutations: bool, rng=None) -> Dict[str, Any]:
        from generator_ir.gen.options import GenConfig
        from generator_ir.gen.assembler import build_kernel

        entry = corpus.draw(rng=rng)
        if entry is None:
            return {"status": "corpus_empty", "errors": ["empty corpus"]}
        config = GenConfig(seed=entry.seed, complexity=complexity,
                           amplify_iters=2048, target=self.target,
                           isa=self.compiler.isa)
        config.template_name = entry.template
        try:
            program = build_kernel(config)
        except Exception as exc:
            return {"status": "gen_fail", "seed": entry.seed,
                    "errors": [str(exc)]}
        try:
            mutated = self.pipeline.perturb_random(program, num_mutations=1,
                                                  paper_only=True)
            original_source = self.kx_translator.rendersrc(program)
            mutated_source = self.kx_translator.rendersrc(mutated)
        except Exception as exc:
            return {"status": "mutation_fail", "seed": entry.seed,
                    "errors": [str(exc)]}

        result: Dict[str, Any] = {"seed": entry.seed, "status": "unknown",
                                  "regressions": [], "errors": [],
                                  "corpus_entry_id": entry.entry_id}
        tmp = tempfile.mkdtemp(prefix="gpuperf_corpus_")
        cr = self.compiler.assemble(original_source, opt_levels[0], output_dir=tmp)
        if not cr.success:
            self.stats.compile_fail += 1
            result["status"] = "compile_fail"
            if self.cleanup_tmp:
                self.compiler.cleanup(tmp)
            return result
        self.stats.compile_success += 1
        rr = self.compiler.run(cr.output_path, timeout=self.benchmark.run_timeout)
        if self.cleanup_tmp:
            self.compiler.cleanup(tmp)
        if not rr.success:
            self.stats.run_fail += 1
            result["status"] = "run_fail"
            return result
        self.stats.run_success += 1
        if rr.avg_time_ms is not None:
            self.stats.runtime_sum_ms += rr.avg_time_ms
        result["status"] = "success"

        if test_mutations:
            regressions = self._xtest_mutations(program, original_source,
                                               config, opt_levels)
            result["regressions"] = regressions
            for reg in regressions:
                corpus.kx_record_regression(entry.entry_id,
                                         mutation=reg.get("mutation", ""))
        return result

    def _kx_add_to_corpus(self, corpus, config, result) -> None:
        try:
            from generator_ir.gen.assembler import build_kernel
            program = build_kernel(config)
            source = self.kx_translator.rendersrc(program)
            perf_data = {}
            if result.get("avg_time_ms") is not None:
                perf_data["o3_time_us"] = result["avg_time_ms"] * 1000.0
            corpus.add(program, seed=config.seed,
                       template=getattr(config, "template_name", "unknown"),
                       checksum=(str(result["checksum"])
                                  if result.get("checksum") is not None else None),
                       perf_data=perf_data or None, source_code=source)
        except Exception as exc:
            log.debug("corpus add failed: %s", exc)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def _kx_save_regression(self, seed: int, result: Dict[str, Any]) -> None:
        case_dir = self.output_dir / f"regression_{seed}"
        case_dir.mkdir(parents=True, exist_ok=True)
        ext = ".hip" if self.target == "hip" else ".cu"
        snapshot = copy.deepcopy(result)
        for i, reg in enumerate(snapshot.get("regressions", [])):
            name = reg.get("mutation", f"unknown_{i}").replace("/", "_").replace(" ", "_")
            orig = reg.pop("original_source", None)
            mut = reg.pop("mutated_source", None)
            if orig:
                (case_dir / f"original_{name}{ext}").write_text(orig)
            if mut:
                (case_dir / f"mutated_{name}{ext}").write_text(mut)
        (case_dir / "result.json").write_text(
            json.dumps(snapshot, indent=2, default=str))
        log.info("regression saved: %s", case_dir)

    def _progress(self, current: int, total: int) -> None:
        s = self.stats
        log.info("[%3.0f%%] %d/%d | assemble %.0f%% | run %.0f%% | "
                 "avg %.1fms | regressions %d | %.0fs",
                 current / total * 100, current, total,
                 s.assemble_rate, s.execute_rate, s.kx_avg_runtime_ms,
                 s.regressions_found, s.kx_elapsed_s)

    def _final_stats(self) -> None:
        s = self.stats
        log.info("done: %d generated, assemble %.1f%%, run %.1f%%, "
                 "%d regressions, %.1fs",
                 s.total_generated, s.assemble_rate, s.execute_rate,
                 s.regressions_found, s.kx_elapsed_s)
