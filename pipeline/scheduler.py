"""Continuous producer-consumer loop.

    generator -> compile (parallel) -> GPU run (serial per device) -> triage/compare
              -> classify -> archive / retain / GC -> mutate & generate next batch

The loop is cooperative: it checks a stop event between ticks, never starts new work after
stop, waits (bounded) for in-flight compile/run futures, then persists queue state, seeds
and statistics atomically.  `resume` rehydrates from queue/state.json and queue/run_id.txt.
"""

from __future__ import annotations

import json
import random
import signal
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import Config, Toolchain
from .models import CaseResult, Category, Status
from .queue import TaskQueue
from . import archive, comparator, compiler, executor, oracle as oracle_mod, stress_metrics
from generator.base import GeneratedCase, default_registry
from generator.rng import rng_for


_PTX_LOCKS: Dict[str, threading.Lock] = {}
_PTX_LOCKS_GUARD = threading.Lock()


def _ptx_lock(ptx_dir: Path) -> threading.Lock:
    key = str(ptx_dir.resolve())
    with _PTX_LOCKS_GUARD:
        return _PTX_LOCKS.setdefault(key, threading.Lock())


def detect_gpu_devices() -> List[int]:
    import subprocess
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=15)
        idx = [int(l.strip()) for l in r.stdout.splitlines() if l.strip().isdigit()]
        return idx or [0]
    except Exception:  # noqa: BLE001
        return [0]


class _Stopper:
    def __init__(self) -> None:
        self._event = threading.Event()
        self.install()

    def install(self) -> None:
        def handler(signum, frame):
            self._event.set()
        try:
            signal.signal(signal.SIGINT, handler)
            signal.signal(signal.SIGTERM, handler)
        except ValueError:  # not in main thread
            pass

    def stop(self):
        self._event.set()

    @property
    def is_set(self) -> bool:
        return self._event.is_set()


@dataclass
class LoopStats:
    generated: int = 0
    compiled: int = 0
    ran: int = 0
    triaged: int = 0
    categories: Dict[str, int] = field(default_factory=dict)
    strategy_weights: Dict[str, float] = field(default_factory=dict)
    strategy_novelty: Dict[str, float] = field(default_factory=dict)
    started_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated": self.generated, "compiled": self.compiled, "ran": self.ran,
            "triaged": self.triaged, "categories": self.categories,
            "strategy_weights": self.strategy_weights,
            "strategy_novelty": self.strategy_novelty,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LoopStats":
        s = cls()
        for k, v in d.items():
            if hasattr(s, k):
                setattr(s, k, v)
        return s


class LoopScheduler:
    def __init__(self, config: Config, run_dir: Path, queue: TaskQueue,
                 stop: Optional[_Stopper] = None,
                 max_cases: int = 0, duration: float = 0.0, config_seed: int = 0):
        self.cfg = config
        self.run_dir = run_dir
        self.queue = queue
        self.stop = stop or _Stopper()
        self.max_cases = max_cases
        self.duration = duration
        self.config_seed = int(config_seed or config.generation.get("config_seed", 20260810))
        self.registry = default_registry()

        state = queue.read_state()
        self.stats = LoopStats.from_dict(state.get("stats", {}))
        if not self.stats.strategy_weights:
            self.stats.strategy_weights = dict(config.scheduling.get("strategy_weights", {}))
        self._counter = int(state.get("test_counter", 0))
        self._case_seen: set = set(state.get("seen_case_ids", []))

        self.compile_pool = ThreadPoolExecutor(max_workers=int(config.compilation.get("jobs_compile", 8)),
                                               thread_name_prefix="compile")
        self.devices = config.execution.get("gpu_devices") or detect_gpu_devices()
        self.gpu_workers = int(config.execution.get("gpu_workers_per_device", 1))
        # Enough GPU threads to keep every device-worker busy plus slack for the
        # threads blocked on the per-device gate.
        self.run_pool = ThreadPoolExecutor(
            max_workers=max(4, 2 * len(self.devices) * self.gpu_workers),
            thread_name_prefix="gpu")
        self.device_gate = executor.DeviceGate(self.devices, workers_per_device=self.gpu_workers)

        # in-flight tracking
        self._build_futures: Dict[str, Dict[str, Future]] = {}
        self._run_futures: Dict[str, Dict[str, Future]] = {}
        self._cases_compiled: Dict[str, bool] = {}
        self._cases_ran: Dict[str, bool] = {}
        self._inflight_lock = threading.Lock()
        self._generated_count = self.stats.generated

        # Per-device case occupancy so parallel GPUs each get their share of
        # work (see _pick_device / _dispatch_runs / _collect_runs).
        self._device_load: Dict[int, int] = {d: 0 for d in self.devices}
        self._case_device: Dict[str, int] = {}
        self._last_gc_count = 0

    # ------------------------------------------------------------------ utils
    def _cfg(self) -> Config:
        return self.cfg

    def _test_dir(self, case_id: str) -> Path:
        return archive.test_dir_for(self.run_dir, case_id)

    def _save_state(self) -> None:
        self.stats.generated = self._generated_count
        state = {
            "test_counter": self._counter,
            "stats": self.stats.to_dict(),
            "seen_case_ids": sorted(self._case_seen)[-5000:],
            "generation_mode": self.cfg.generation.get("generator", "builtin"),
        }
        self.queue.write_state(state)

    # ------------------------------------------------------------------ generation
    def _generation_strategy_names(self) -> List[str]:
        mode = str(self.cfg.generation.get("generator", "builtin"))
        enabled = [
            name for name in self.registry.names()
            if bool(self.cfg.strategies.get(name, {}).get("enabled", True))
        ]
        if mode == "generator_ir":
            return ["generator_ir"]
        if mode == "builtin":
            return [name for name in enabled if name != "generator_ir"]
        if mode == "mixed":
            return list(dict.fromkeys(enabled + ["generator_ir"]))
        raise ValueError(f"unknown generation.generator mode {mode!r}")

    def _target_sm(self) -> str:
        configured = self.cfg.generation.get("target_sm")
        if configured:
            return str(configured)
        if self.cfg.toolchains:
            return str(self.cfg.toolchains[0].arch)
        return "sm_75"

    def _next_strategy(self) -> str:
        w = self.stats.strategy_weights
        available = self._generation_strategy_names()
        names = [n for n in available if n in w and w.get(n, 0) > 0]
        if not names:
            names = available
        if not names:
            raise RuntimeError("no enabled generation strategies")
        total = sum(w.get(n, 1.0) for n in names)
        r = random.uniform(0, total)
        upto = 0.0
        for n in names:
            upto += w.get(n, 1.0)
            if r <= upto:
                return n
        return names[-1]

    def _generate_one(self) -> Optional[GeneratedCase]:
        strategy_name = self._next_strategy()
        strategy_cfg = self.cfg.strategies.get(strategy_name, {})
        variants = strategy_cfg.get("variants", [{}])
        if not variants:
            return None
        params = dict(variants[self._counter % len(variants)])

        # structured mutation: occasionally perturb a variant's pressure knobs
        r = random.Random(self._counter)
        if r.random() < float(self.cfg.generation.get("mutation_ratio", 0.35)):
            params = self._mutate_params(strategy_name, params, r)

        test_seed = self._counter + 0xC0FFEE
        self._counter += 1
        try:
            case = self.registry.get(strategy_name).generate(
                config_seed=self.config_seed, test_seed=test_seed, params=params,
                gen_version=str(self.cfg.generation.get("generator_version", "1.0.0")),
                target_sm=self._target_sm())
        except Exception:  # noqa: BLE001
            return None
        if not case.valid:
            return None
        return case

    def _mutate_params(self, strategy_name: str, params: Dict[str, Any],
                       r: random.Random) -> Dict[str, Any]:
        p = dict(params)
        if strategy_name == "generator_ir":
            # generator_ir has no fixed ABI; diversify by bumping stress knobs.
            # Bump the repeat count / scalar fan-in, and occasionally enable
            # the FP-pool / wide-constant paths (well-defined; observable_fold
            # keeps the checksum integer-exact so no cross-compiler FP noise).
            for key in ("amplify_iters", "scalar_param_count", "codegen_repeat"):
                if r.random() < 0.5:
                    try:
                        p[key] = max(1, int(p[key] * r.choice([1.5, 2.0, 3.0])))
                    except (KeyError, TypeError):
                        pass
            for key in ("prob_float_pool", "prob_wide_consts"):
                if r.random() < 0.4:
                    p[key] = r.choice([0.25, 0.5, 0.75])
            return p
        numeric_keys = {
            "predicate_pressure": ["num_comparisons", "chain_len", "num_diamonds", "dup_groups"],
            "integer_chain": ["chain_len"],
            "memory_address": ["reads"],
            "warp_mixed": ["chain_len"],
            "register_pressure": ["live_vars", "chains", "merge_len"],
            "inline_ptx": ["n_instr", "n_regs32", "n_regs64", "n_preds"],
        }.get(strategy_name, [])
        if numeric_keys:
            key = r.choice(numeric_keys)
            factor = r.choice([0.6, 0.8, 1.2, 1.5])
            try:
                p[key] = max(1, int(p[key] * factor))
            except (KeyError, TypeError):
                pass
        return p

    def _generate_batch(self) -> None:
        cfg = self.cfg
        backlog = int(cfg.generation.get("backlog_target", 64))
        while (self.queue.count("generated") + self.queue.count("compiling")) < backlog:
            if self.stop.is_set:
                return
            if self.max_cases and self._generated_count >= self.max_cases:
                return
            case = self._generate_one()
            if case is None:
                continue
            cid = case.case_id()
            if cid in self._case_seen:
                continue
            self._case_seen.add(cid)
            case_dir = self.queue.root / "generated" / cid
            archive.write_testcase_artifacts(case_dir, case)
            self._generated_count += 1

    # ------------------------------------------------------------------ compile
    def _dispatch_compiles(self) -> None:
        if self.stop.is_set:
            return
        # Backpressure: if the GPU run stage is already saturated, stop pushing
        # newly-compiled cases into ready_to_run.  Without this, compilation far
        # outpaces execution, the ready_to_run queue (and runs/<id>/tests build
        # artifacts) grows without bound, and listing the directory on every loop
        # tick drags the whole scheduler down.
        if self.queue.count("ready_to_run") >= int(self.cfg.execution.get("run_backlog_target", 64)):
            return
        inflight = sum(len(v) for v in self._build_futures.values())
        limit = int(self.cfg.compilation.get("jobs_compile", 8)) * 2
        if inflight >= limit:
            return
        for case_id in self.queue.list("generated")[: max(0, limit - inflight)]:
            case_dir = self.queue.to_compiling(case_id)
            case = _case_from_dir(case_dir)
            test_dir = self._test_dir(case_id)
            archive.write_testcase_artifacts(test_dir, case)
            extra_flags = case.params.get("ptxas_extra", []) if case.params else []
            self._build_futures[case_id] = {}
            for tc in self.cfg.toolchains:
                ptx_dir = test_dir / "_ptx" / tc.id
                f = self.compile_pool.submit(worker_ptx, self.cfg, tc, ptx_dir, test_dir / "test.cu")
                self._build_futures[case_id][f"ptx:{tc.id}"] = f
                for prof in self.cfg.profiles:
                    key = compiler.build_key(tc.id, prof.name)
                    f = self.compile_pool.submit(worker_build, self.cfg, tc, prof, ptx_dir,
                                                 self._test_dir(case_id), case_id, extra_flags)
                    self._build_futures[case_id][key] = f

    def _collect_compiles(self) -> None:
        done_cases = []
        for case_id, futures in list(self._build_futures.items()):
            if not futures:
                continue
            if all(f.done() for f in futures.values()):
                done_cases.append(case_id)
        for case_id in done_cases:
            self._build_futures.pop(case_id, None)
            try:
                self.queue.to_ready(case_id)
            except FileNotFoundError:
                pass
            self._cases_compiled[case_id] = True
            self.stats.compiled += 1

    # ------------------------------------------------------------------ run
    def _pick_device(self) -> Optional[int]:
        """Least-loaded GPU still under its per-device case cap.

        Each device runs ``gpu_workers_per_device`` cases concurrently (default 1);
        within a case, builds serialize on that device via ``DeviceGate``.  Picking
        the least-loaded device spreads independent cases across all GPUs so they
        execute in parallel.
        """
        best: Optional[int] = None
        for d in self.devices:
            load = self._device_load.get(d, 0)
            if load >= self.gpu_workers:
                continue
            if best is None or load < self._device_load.get(best, 0):
                best = d
        return best

    def _dispatch_runs(self) -> None:
        if self.stop.is_set:
            return
        for case_id in self.queue.list("ready_to_run"):
            if case_id in self._run_futures:
                continue
            device = self._pick_device()
            if device is None:
                break
            try:
                self.queue.to_running(case_id)
            except FileNotFoundError:
                continue
            self._device_load[device] += 1
            self._case_device[case_id] = device
            test_dir = self._test_dir(case_id)
            case = _case_from_dir(test_dir)
            self._run_futures[case_id] = {}
            runs = int(self.cfg.execution.get("runs_per_build", 3))
            if self.cfg.execution.get("adaptive_rerun", False):
                runs = 1  # fast pass; a disagreement triggers a full re-run at triage
            for tc in self.cfg.runnable_toolchains():
                for prof in self.cfg.profiles:
                    key = compiler.build_key(tc.id, prof.name)
                    build_dir = test_dir / "builds" / key
                    f = self.run_pool.submit(worker_run, self.cfg, tc, case, build_dir,
                                             runs, self.device_gate, device, self.stop)
                    self._run_futures[case_id][key] = f

    def _collect_runs(self) -> None:
        done = [cid for cid, fs in self._run_futures.items()
                if fs and all(f.done() for f in fs.values())]
        for case_id in done:
            self._run_futures.pop(case_id, None)
            device = self._case_device.pop(case_id, self.devices[0])
            self._device_load[device] = max(0, self._device_load.get(device, 1) - 1)
            try:
                self.queue.to_triage(case_id)
            except FileNotFoundError:
                pass
            self.stats.ran += 1

    # ------------------------------------------------------------------ triage
    def _rerun_case(self, test_dir: Path, case: GeneratedCase, runs: int) -> None:
        """Synchronously re-run every runnable build `runs` times (confirmation pass).

        Only entered when a single-run fast pass saw builds disagree, so the cost is
        paid exactly when it matters (a candidate differential) rather than on the
        common all-agree path.  Serialized on the device gate so it cannot contend
        with the next case already dispatched to the GPU.
        """
        with self.device_gate.acquire(self.devices[0]):
            for tc in self.cfg.runnable_toolchains():
                for prof in self.cfg.profiles:
                    build_dir = test_dir / "builds" / compiler.build_key(tc.id, prof.name)
                    if not build_dir.exists():
                        continue
                    for i in range(1, runs + 1):
                        executor.run_build(self.cfg, tc, case, build_dir, i,
                                           device=self.devices[0])

    def _triage_one(self, case_id: str) -> None:
        test_dir = self._test_dir(case_id)
        case = _case_from_dir(test_dir)
        runs = int(self.cfg.execution.get("runs_per_build", 3))
        if self.cfg.execution.get("adaptive_rerun", False):
            result = comparator.compare_matrix(self.cfg, case, test_dir, runs=1)
            if result.status == Status.POTENTIAL_COMPILER_BUG:
                # A single flaky run must never masquerade as a wrong-code regression:
                # re-run the full matrix `runs` times and re-compare before deciding.
                self._rerun_case(test_dir, case, runs)
                result = comparator.compare_matrix(self.cfg, case, test_dir, runs=runs)
        else:
            result = comparator.compare_matrix(self.cfg, case, test_dir)
        metrics_list = stress_metrics.collect_build_metrics(test_dir)
        gate = stress_metrics.evaluate_stress_gate(self.cfg, case, test_dir, metrics_list)
        result.stress_gate = gate

        corpus_hashes = set(self.queue.read_state().get("seen_case_ids", []))
        result.novel_score = stress_metrics.novelty_score(
            self.cfg, case, metrics_list, corpus_hashes, set())

        # upgrade PASS to interesting if it clears the stress gate
        if result.status == Status.PASS:
            result.category = Category.INTERESTING if gate["passed"] else Category.REDUNDANT

        # input-sensitivity check for retained cases
        if result.category in (Category.INTERESTING, Category.POTENTIAL_BUG) and \
           result.status != Status.HANG_DIVERGENCE and \
           self.cfg.oracle.get("verify_input_sensitivity", True):
            try:
                if case.oracle_mode == "cross_build":
                    sens = oracle_mod.verify_runtime_input_sensitivity(self.cfg, test_dir)
                else:
                    sens = oracle_mod.verify_input_sensitivity(test_dir, self.config_seed)
                result.stress_gate["input_sensitive"] = sens["input_sensitive"]
                if not sens["input_sensitive"]:
                    result.category = Category.REDUNDANT
            except Exception:  # noqa: BLE001
                pass

        archive.write_case_result(self.run_dir, case_id, result)

        if result.category == Category.POTENTIAL_BUG:
            diff_dir = archive.write_differential(self.cfg, self.run_dir, case, result)
            self.queue.move_to_done_with_reason(case_id, result.category, result.note)
        elif result.category == Category.INVALID:
            self.queue.to_rejected(case_id, reason=result.note or "compile/invalid")
        else:
            self.queue.move_to_done_with_reason(case_id, result.category, result.note)

        self.stats.categories[result.category] = self.stats.categories.get(result.category, 0) + 1
        self._update_strategy_weights(result)
        self.stats.triaged += 1

    def _update_strategy_weights(self, result: CaseResult) -> None:
        if not self.cfg.scheduling.get("adaptive_weights", True):
            return
        w = self.stats.strategy_weights
        base = w.get(result.strategy, 1.0)
        gain = 1.15 if result.category == Category.INTERESTING else \
               1.4 if result.category == Category.POTENTIAL_BUG else \
               0.9 if result.category in (Category.REDUNDANT, Category.INVALID) else 1.0
        w[result.strategy] = round(min(3.0, max(0.2, base * gain)), 3)

    def _triage_pending(self) -> None:
        for case_id in list(self.queue.list("triage")):
            if self.stop.is_set and not self.queue.list("triage"):
                break
            try:
                self._triage_one(case_id)
            except Exception as e:  # noqa: BLE001
                self.queue.move_to_done_with_reason(case_id, "invalid", f"triage error: {e}")
            if self.max_cases and self.stats.triaged >= self.max_cases:
                return

    # ------------------------------------------------------------------ main
    def run(self) -> None:
        t0 = time.monotonic()
        self.stats.started_at = t0
        archive.init_run(self.run_dir, self.cfg)
        self.queue.write_run_id(self.run_dir.name)
        print(f"[loop] run_id={self.run_dir.name}  seed={self.config_seed}  "
              f"generator={self.cfg.generation.get('generator', 'builtin')}  "
              f"devices={self.devices}  profiles={[p.name for p in self.cfg.profiles]}  "
              f"toolchains={[t.id for t in self.cfg.toolchains]}")

        while not self.stop.is_set:
            if self.max_cases and self.stats.triaged >= self.max_cases:
                print(f"[loop] reached max_cases={self.max_cases}")
                break
            if self.duration and (time.monotonic() - t0) > self.duration:
                print(f"[loop] reached duration={self.duration:.0f}s")
                break

            self._generate_batch()
            self._dispatch_compiles()
            self._collect_compiles()
            self._dispatch_runs()
            self._collect_runs()
            self._triage_pending()

            # periodic GC
            if self.stats.triaged - self._last_gc_count >= int(self.cfg.storage.get("gc_interval_cases", 25)):
                self._last_gc_count = self.stats.triaged
                try:
                    gc_report = archive.run_gc(self.cfg, self.run_dir, self.queue.read_state())
                    self.queue.write_state({**self.queue.read_state(), "last_gc": gc_report})
                except Exception as e:  # noqa: BLE001
                    print(f"[loop] gc error: {e}")

            self._save_state()
            time.sleep(0.05)

        # graceful shutdown: drain in-flight (bounded), then persist.
        self._drain(timeout=20)
        self._save_state()
        print(f"[loop] stopped. generated={self._generated_count} triaged={self.stats.triaged} "
              f"categories={self.stats.categories}")

    def _drain(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._collect_compiles()
            self._collect_runs()
            remaining_b = sum(len(v) for v in self._build_futures.values())
            remaining_r = sum(len(v) for v in self._run_futures.values())
            if remaining_b == 0 and remaining_r == 0:
                break
            time.sleep(0.05)
        # any still-running tasks keep running (their futures persist in state implicitly
        # via the queue stage directories); on next resume they are recovered.


def _case_from_dir(dir_path: Path) -> GeneratedCase:
    from .manifest import case_from_dir
    return case_from_dir(dir_path)


def run_single_case(config: Config, test_dir: Path, runs: int = 3,
                    toolchain_ids: Optional[List[str]] = None,
                    profile_names: Optional[List[str]] = None,
                    device: int = 0) -> Dict[str, Any]:
    """Synchronous full matrix for one materialized testcase dir (CLI `run`)."""
    case = _case_from_dir(test_dir)
    tc_list = [t for t in config.toolchains
               if (not toolchain_ids) or t.id in toolchain_ids]
    prof_list = [p for p in config.profiles
                 if (not profile_names) or p.name in profile_names]
    known_tc = {t.id for t in config.toolchains}
    known_profiles = {p.name for p in config.profiles}
    missing_tc = sorted(set(toolchain_ids or []) - known_tc)
    missing_profiles = sorted(set(profile_names or []) - known_profiles)
    if missing_tc or missing_profiles:
        detail = []
        if missing_tc:
            detail.append(f"unknown toolchains: {missing_tc}")
        if missing_profiles:
            detail.append(f"unknown profiles: {missing_profiles}")
        raise ValueError("; ".join(detail))
    if not tc_list or not prof_list:
        raise ValueError("the selected compile matrix is empty")
    extra_flags = case.params.get("ptxas_extra", []) if case.params else []
    (test_dir / "execution_plan.json").write_text(json.dumps({
        "toolchains": [tc.id for tc in tc_list],
        "profiles": [p.name for p in prof_list],
        "runs_per_build": max(1, int(runs)),
        "device": int(device),
    }, indent=2, sort_keys=True))

    summary = {"case_id": case.case_id(), "builds": {}}
    for tc in tc_list:
        ptx_dir = test_dir / "_ptx" / tc.id
        ptx_dir.mkdir(parents=True, exist_ok=True)
        worker_ptx(config, tc, ptx_dir, test_dir / "test.cu")
        for prof in prof_list:
            key = compiler.build_key(tc.id, prof.name)
            build_dir = test_dir / "builds" / key
            build_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            src_ptx = ptx_dir / "kernel.ptx"
            if src_ptx.exists():
                shutil.copy2(src_ptx, build_dir / "kernel.ptx")
            result = compiler.compile_build(config, tc, prof, test_dir, build_dir,
                                            extra_flags,
                                            timeout=int(config.compilation.get("compile_timeout", 120)))
            (build_dir / "compile_result.json").write_text(json.dumps(result, indent=2))
            if result["ok"]:
                from generator.base import default_registry
                try:
                    tgt = default_registry().get(case.strategy).target_opcodes
                except Exception:  # noqa: BLE001
                    tgt = []
                metrics = compiler.compute_metrics(build_dir, tgt)
                metrics["elapsed_s"] = result.get("elapsed_s", 0)
                (build_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
            run_results = []
            for i in range(1, runs + 1):
                rr = executor.run_build(config, tc, case, build_dir, i, device=device)
                run_results.append({"run": i, "status": rr.status, "error": rr.error,
                                    "elapsed_us": rr.elapsed_us})
            summary["builds"][key] = {
                "compile_ok": result["ok"], "compile_error": result.get("error"),
                "runs": run_results,
            }
    return summary


def worker_ptx(config: Config, tc: Toolchain, ptx_dir: Path, src: Path) -> Dict[str, Any]:
    """Compile test.cu -> kernel.ptx for one toolchain (once, shared by all profiles)."""
    ptx_dir.mkdir(parents=True, exist_ok=True)
    ptx_path = ptx_dir / "kernel.ptx"
    with _ptx_lock(ptx_dir):
        if ptx_path.exists() and ptx_path.stat().st_size > 0:
            rec = compiler.CmdRecord(
                argv=[], returncode=0, stdout="", stderr="",
                elapsed=0.0, ok=True)
        else:
            try:
                manifest = json.loads((src.parent / "manifest.json").read_text())
            except Exception:  # noqa: BLE001
                manifest = {}
            mode = manifest.get("execution_mode", "driver_kernel")
            ptx_fn = (compiler.nvcc_standalone_ptx
                      if compiler.uses_full_source_ptx(mode)
                      else compiler.nvcc_ptx)
            rec = ptx_fn(
                tc, src, ptx_path,
                timeout=int(config.compilation.get("ptx_timeout", 60)))
    return {"toolchain": tc.id, "ok": rec.ok, "returncode": rec.returncode,
            "stderr": rec.stderr[-500:]}


def worker_build(config: Config, tc: Toolchain, prof, ptx_dir: Path, test_dir: Path,
                 case_id: str, extra_flags: Optional[List[str]]) -> Dict[str, Any]:
    """Compile one (toolchain x profile) build: cubin + sass + metrics."""
    build_dir = test_dir / "builds" / compiler.build_key(tc.id, prof.name)
    build_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    ptx_status = worker_ptx(config, tc, ptx_dir, test_dir / "test.cu")
    ptx_src = ptx_dir / "kernel.ptx"
    if ptx_status.get("ok") and ptx_src.exists():
        shutil.copy2(ptx_src, build_dir / "kernel.ptx")
    else:
        result = {
            "toolchain": tc.id, "profile": prof.name, "ok": False,
            "error": (f"shared PTX generation failed rc={ptx_status.get('returncode')}: "
                      f"{ptx_status.get('stderr', '')[-300:]}"),
            "commands": [],
        }
        (build_dir / "compile_result.json").write_text(json.dumps(result, indent=2))
        return {"build": compiler.build_key(tc.id, prof.name), "ok": False,
                "error": result["error"]}
    result = compiler.compile_build(config, tc, prof, test_dir, build_dir, extra_flags,
                                    timeout=int(config.compilation.get("compile_timeout", 120)))
    (build_dir / "compile_result.json").write_text(json.dumps(result, indent=2))
    if result["ok"]:
        from generator.base import default_registry
        try:
            tgt = default_registry().get(_manifest_strategy(test_dir)).target_opcodes
        except Exception:  # noqa: BLE001
            tgt = []
        metrics = compiler.compute_metrics(build_dir, tgt)
        metrics["elapsed_s"] = result.get("elapsed_s", 0)
        (build_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return {"build": compiler.build_key(tc.id, prof.name), "ok": result["ok"],
            "error": result.get("error")}


def _manifest_strategy(test_dir: Path) -> str:
    import json
    m = json.loads((test_dir / "manifest.json").read_text())
    return m.get("strategy", "")


def worker_run(config: Config, tc: Toolchain, case: GeneratedCase, build_dir: Path,
               runs: int, gate: executor.DeviceGate, device: int, stop: _Stopper) -> Dict[str, Any]:
    """Execute one build `runs` times, serialized on the device gate. Fresh process per run."""
    with gate.acquire(device):
        results = []
        for i in range(1, runs + 1):
            if stop.is_set:
                break
            r = executor.run_build(config, tc, case, build_dir, i, device=device)
            results.append({"run": i, "status": r.status, "error": r.error,
                            "elapsed_us": r.elapsed_us})
        return {"build": build_dir.name, "toolchain": tc.id, "runs": results}
