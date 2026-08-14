"""Command-line interface.

    python3 pipeline.py loop --config config/pipeline.yaml [--max-cases N] [--duration 24h] [--seed S]
    python3 pipeline.py resume
    python3 pipeline.py status
    python3 -m cuda_stress doctor
    python3 -m cuda_stress generate   ...
    python3 -m cuda_stress reproduce  --id <case_id>
    python3 -m cuda_stress run        --test <dir>
    python3 -m cuda_stress compare    --test <dir>
    python3 -m cuda_stress report     [--run <run_id>] [--json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .config import Config, load_config
from . import archive, comparator, probe, scheduler
from .queue import TaskQueue
from generator.base import default_registry, GeneratedCase

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _make_config(args) -> Config:
    cfg = load_config(config_path=args.config,
                      toolchains_path=getattr(args, "toolchains", None),
                      strategies_path=getattr(args, "strategies", None))
    if getattr(args, "seed", None) is not None:
        cfg.generation["config_seed"] = int(args.seed)
    return cfg


def cmd_doctor(args) -> int:
    cfg = _make_config(args)
    probes = probe.probe_all(cfg) if args.force else probe.cached_probes(cfg, force=args.force)
    gpu = probe.get_gpu_info()
    print(probe.doctor_report(cfg, probes, gpu))
    return 0


def _materialize_generated(config: Config, case: GeneratedCase, out_dir: Path) -> Path:
    cid = case.case_id()
    d = out_dir / cid
    d.mkdir(parents=True, exist_ok=True)
    archive.write_testcase_artifacts(d, case)
    (d / "generation.json").write_text(json.dumps({
        "generator_version": case.generator_version, "config_seed": case.config_seed,
        "test_seed": case.test_seed, "strategy": case.strategy, "strategy_params": case.params,
        "execution_mode": case.execution_mode, "oracle_mode": case.oracle_mode,
    }, indent=2))
    return d


def _selected_strategies(config: Config, reg, explicit: List[str],
                         generator_mode: Optional[str]) -> List[str]:
    """Resolve the optional generator backend without changing legacy defaults."""
    if explicit:
        return list(dict.fromkeys(explicit))
    mode = generator_mode or str(config.generation.get("generator", "builtin"))
    enabled = [
        name for name in reg.names()
        if bool(config.strategies.get(name, {}).get("enabled", True))
    ]
    if mode == "generator_ir":
        return ["generator_ir"]
    if mode == "builtin":
        return [name for name in enabled if name != "generator_ir"]
    if mode == "mixed":
        # An explicit mixed selection opts into the IR backend even when its
        # default config entry is disabled for backwards compatibility.
        return list(dict.fromkeys(enabled + ["generator_ir"]))
    raise ValueError(f"unknown generator mode {mode!r}")


def _target_sm(config: Config) -> str:
    configured = config.generation.get("target_sm")
    if configured:
        return str(configured)
    if config.toolchains:
        return str(config.toolchains[0].arch)
    return "sm_86"


def _regenerate(case: GeneratedCase) -> GeneratedCase:
    return default_registry().get(case.strategy).generate(
        config_seed=case.config_seed,
        test_seed=case.test_seed,
        params=case.params,
        gen_version=case.generator_version,
        target_sm=case.target_sm,
    )


def _same_generated_artifacts(a: GeneratedCase, b: GeneratedCase) -> bool:
    return (
        a.case_id() == b.case_id()
        and a.source_sha256 == b.source_sha256
        and a.input_sha256 == b.input_sha256
        and a.expected_sha256 == b.expected_sha256
        and a.execution_mode == b.execution_mode
        and a.oracle_mode == b.oracle_mode
        and a.execution_spec == b.execution_spec
    )


def cmd_generate(args) -> int:
    config = _make_config(args)
    reg = default_registry()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.reproduce:
        # regenerate one existing case byte-for-byte and verify
        src = find_case_dir(config, args.reproduce)
        if src is None:
            print(f"case {args.reproduce} not found", file=sys.stderr)
            return 1
        stored = _case_from_dir(src)
        case = _regenerate(stored)
        d = _materialize_generated(config, case, out_dir)
        ok = check_case_dir(d) and _same_generated_artifacts(stored, case)
        print(f"reproduced {stored.case_id()} -> {d}  hashes={'OK' if ok else 'MISMATCH'}")
        return 0 if ok else 1

    try:
        strategies = _selected_strategies(
            config, reg, args.strategy, getattr(args, "generator", None))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not strategies:
        print("no enabled generation strategies selected", file=sys.stderr)
        return 2
    count = int(args.count)
    config_seed = int(args.config_seed)
    start_seed = int(args.test_seed)
    params = json.loads(args.params) if args.params else None

    made = 0
    for sname in strategies:
        strat = reg.get(sname)
        for i in range(count):
            test_seed = start_seed + i
            p = params if params is not None else _default_params(config, sname)
            case = strat.generate(
                config_seed, test_seed, p,
                str(config.generation.get("generator_version", "1.0.0")),
                _target_sm(config))
            d = _materialize_generated(config, case, out_dir)
            try:
                rel = d.relative_to(PROJECT_ROOT)
            except ValueError:
                rel = d
            print(f"generated {case.case_id()}  strategy={sname}  test_seed={test_seed}  -> {rel}")
            made += 1
    print(f"generated {made} testcases into {out_dir}")
    return 0


def _default_params(config: Config, sname: str) -> Dict[str, Any]:
    scfg = config.strategies.get(sname, {})
    variants = scfg.get("variants", [{}])
    return dict(variants[0]) if variants else {}


def find_case_dir(config: Config, case_id: str) -> Optional[Path]:
    """Search corpus/, runs/, queue/ for a materialized case dir."""
    roots = [config.project_root / "corpus" / "accepted",
             config.project_root / "corpus" / "rejected"]
    q = TaskQueue(config.project_root / "queue")
    for s in ("done", "triage", "ready_to_run", "running", "compiling", "generated"):
        roots.append(config.project_root / "queue" / s)
    rid = q.read_run_id()
    if rid:
        roots.append(config.project_root / "runs" / rid / "tests")
    for root in roots:
        d = root / case_id
        if (d / "manifest.json").exists():
            return d
    return None


def cmd_reproduce(args) -> int:
    config = _make_config(args)
    src = find_case_dir(config, args.id)
    if src is None:
        print(f"case {args.id} not found in corpus/runs/queue", file=sys.stderr)
        return 1
    stored = _case_from_dir(src)
    case = _regenerate(stored)
    out_dir = Path(args.out).resolve()
    d = _materialize_generated(config, case, out_dir)
    ok = check_case_dir(d) and _same_generated_artifacts(stored, case)
    (d / "reproduce.sh").write_text(archive.make_reproduce_sh(config, case))
    print(f"case_id     : {case.case_id()}")
    print(f"strategy    : {case.strategy}")
    print(f"config_seed : {case.config_seed}  test_seed: {case.test_seed}  kernel_seed: {case.kernel_seed}")
    print(f"source hash : {case.source_sha256}")
    print(f"input hash  : {case.input_sha256}")
    print(f"expected    : {case.expected_sha256}")
    print(f"reproduced  : {d}  hashes={'OK' if ok else 'MISMATCH'}")
    return 0 if ok else 1


def check_case_dir(d: Path) -> bool:
    import json
    m = json.loads((d / "manifest.json").read_text())
    src = (d / "test.cu").read_text()
    inp = (d / "input.bin").read_bytes()
    exp = (d / "expected.bin").read_bytes()
    return (hashlib.sha256(src.encode()).hexdigest() == m["hashes"]["source"].split(":", 1)[1] and
            hashlib.sha256(inp).hexdigest() == m["hashes"]["input"].split(":", 1)[1] and
            hashlib.sha256(exp).hexdigest() == m["hashes"]["expected"].split(":", 1)[1])


def cmd_run(args) -> int:
    config = _make_config(args)
    test_dir = Path(args.test).resolve()
    if not (test_dir / "test.cu").exists():
        print(f"no test.cu in {test_dir}", file=sys.stderr)
        return 1
    try:
        res = scheduler.run_single_case(config, test_dir,
                                        runs=int(args.runs),
                                        toolchain_ids=args.toolchain,
                                        profile_names=args.profile,
                                        device=int(args.device))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(res, indent=2, default=str))
    return 0


def cmd_compare(args) -> int:
    config = _make_config(args)
    test_dir = Path(args.test).resolve()
    case = _case_from_dir(test_dir)
    result = comparator.compare_matrix(config, case, test_dir)
    print(f"case       : {case.case_id()}")
    print(f"status     : {result.status}")
    print(f"category   : {result.category}")
    print(f"note       : {result.note}")
    for key, br in sorted(result.builds.items()):
        val = br.run_output_values[0] if br.run_output_values and br.run_output_values[0] is not None else None
        if val is not None:
            out = str(val)
        elif br.run_output_hashes and br.run_output_hashes[0]:
            out = br.run_output_hashes[0][:8] + " (hash)"
        else:
            out = "-"
        print(f"  {key:<28} {br.status:<22} out={out} compile={br.compile_ok} stable={br.stable} "
              f"regs={br.ptxas_registers} runs={br.run_statuses}")
    return 0


def cmd_report(args) -> int:
    config = _make_config(args)
    run_dir = _resolve_run_dir(config, args.run)
    if run_dir is None:
        print("no run specified and no active run", file=sys.stderr)
        return 1
    ledger = run_dir / "results.jsonl"
    rows = []
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    pass
    summary = {
        "run": run_dir.name,
        "cases": len(rows),
        "by_status": _count(rows, "status"),
        "by_category": _count(rows, "category"),
        "interesting": [r.get("case_id") for r in rows if r.get("category") == "interesting"][:20],
        "bugs": [r for r in rows if r.get("category") == "potential_bug"],
        "flaky": [r.get("case_id") for r in rows if r.get("category") == "flaky"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"run        : {run_dir.name}")
        print(f"cases      : {summary['cases']}")
        print(f"by_status  : {summary['by_status']}")
        print(f"categories : {summary['by_category']}")
        for r in summary["bugs"]:
            print(f"  BUG {r.get('case_id')} {r.get('status')} {r.get('note','')[:80]}")
    return 0


def _count(rows: List[Dict], key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        out[r.get(key, "?")] = out.get(r.get(key, "?"), 0) + 1
    return out


def _resolve_run_dir(config: Config, run_id: Optional[str]) -> Optional[Path]:
    if run_id:
        return config.project_root / "runs" / run_id
    q = TaskQueue(config.project_root / "queue")
    rid = q.read_run_id()
    if rid:
        return config.project_root / "runs" / rid
    runs = sorted((config.project_root / "runs").glob("run-*")) if (config.project_root / "runs").exists() else []
    return runs[-1] if runs else None


def cmd_status(args) -> int:
    config = _make_config(args)
    q = TaskQueue(config.project_root / "queue")
    run_id = q.read_run_id()
    print(f"run_id      : {run_id or '-'}")
    for s in ("generated", "compiling", "ready_to_run", "running", "triage", "done", "rejected"):
        print(f"queue/{s:<14}: {q.count(s)}")
    st = q.read_state()
    print(f"generated   : {st.get('stats', {}).get('generated', 0)}")
    print(f"triaged     : {st.get('stats', {}).get('triaged', 0)}")
    print(f"categories  : {st.get('stats', {}).get('categories', {})}")
    return 0


def cmd_loop(args) -> int:
    config = _make_config(args)
    queue = TaskQueue(config.project_root / "queue")
    if args.resume:
        queue.recover()
        # A plain `resume` continues the backend selected by the interrupted
        # loop.  An explicit flag still overrides the saved mode.
        saved_mode = queue.read_state().get("generation_mode")
        if not getattr(args, "generator", None) and saved_mode:
            config.generation["generator"] = saved_mode
        run_id = queue.read_run_id()
        if not run_id:
            run_id = archive.new_run_id(config)
        run_dir = config.project_root / "runs" / run_id
    else:
        run_dir = config.project_root / "runs" / archive.new_run_id(config)
    if getattr(args, "generator", None):
        config.generation["generator"] = args.generator
    stop = scheduler._Stopper()
    sched = scheduler.LoopScheduler(config, run_dir, queue, stop=stop,
                                    max_cases=int(args.max_cases or 0),
                                    duration=_parse_duration(args.duration),
                                    config_seed=config.generation.get("config_seed", 20260810))
    try:
        sched.run()
    except KeyboardInterrupt:
        stop.stop()
        sched._save_state()
    return 0


def cmd_gc(args) -> int:
    config = _make_config(args)
    run_dir = _resolve_run_dir(config, args.run)
    if run_dir is None:
        run_dir = config.project_root / "runs" / archive.new_run_id(config)
    report = archive.run_gc(config, run_dir, {})
    print(json.dumps(report, indent=2, default=str))
    return 0


def _parse_duration(s: Optional[str]) -> float:
    if not s:
        return 0.0
    s = s.strip().lower()
    if s.endswith("h"):
        return float(s[:-1]) * 3600
    if s.endswith("m"):
        return float(s[:-1]) * 60
    if s.endswith("s"):
        return float(s[:-1])
    return float(s)


def _case_from_dir(d: Path) -> GeneratedCase:
    from .manifest import case_from_dir
    return case_from_dir(d)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cuda_stress", description="CUDA compiler differential-testing pipeline")
    p.add_argument("--config", default=None)
    p.add_argument("--toolchains", default=None)
    p.add_argument("--strategies", default=None)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="probe toolchains/GPU/ptxas -O0..-O4")
    d.add_argument("--force", action="store_true")
    d.set_defaults(func=cmd_doctor)

    g = sub.add_parser("generate", help="generate testcases")
    g.add_argument("--strategy", action="append", default=[])
    g.add_argument("--generator", choices=["builtin", "generator_ir", "mixed"],
                   default=None, help="select a generation backend (strategies remain available)")
    g.add_argument("--count", default=1)
    g.add_argument("--config-seed", default=20260810)
    g.add_argument("--test-seed", default=0)
    g.add_argument("--params", default=None)
    g.add_argument("--out", default=str(PROJECT_ROOT / "corpus" / "accepted"))
    g.add_argument("--reproduce", default=None, help="regenerate case with this id and verify hashes")
    g.set_defaults(func=cmd_generate)

    r = sub.add_parser("reproduce", help="rebuild a case byte-for-byte from its manifest")
    r.add_argument("--id", required=True)
    r.add_argument("--out", default="/tmp/cuda_stress_reproduce")
    r.set_defaults(func=cmd_reproduce)

    rn = sub.add_parser("run", help="compile + run a single testcase matrix")
    rn.add_argument("--test", required=True)
    rn.add_argument("--runs", default=3)
    rn.add_argument("--toolchain", action="append", default=[])
    rn.add_argument("--profile", action="append", default=[])
    rn.add_argument("--device", default=0)
    rn.set_defaults(func=cmd_run)

    c = sub.add_parser("compare", help="compare a testcase's build outputs")
    c.add_argument("--test", required=True)
    c.set_defaults(func=cmd_compare)

    rep = sub.add_parser("report", help="summarize a run")
    rep.add_argument("--run", default=None)
    rep.add_argument("--json", action="store_true")
    rep.set_defaults(func=cmd_report)

    st = sub.add_parser("status", help="show queue/run state")
    st.set_defaults(func=cmd_status)

    lp = sub.add_parser("loop", help="continuous differential-testing loop")
    lp.add_argument("--max-cases", default=0)
    lp.add_argument("--duration", default=None)
    lp.add_argument("--seed", default=None)
    lp.add_argument("--resume", action="store_true")
    lp.add_argument("--generator", choices=["builtin", "generator_ir", "mixed"], default=None)
    lp.set_defaults(func=cmd_loop)

    rs = sub.add_parser("resume", help="resume the interrupted loop")
    rs.add_argument("--max-cases", default=0)
    rs.add_argument("--duration", default=None)
    rs.add_argument("--generator", choices=["builtin", "generator_ir", "mixed"], default=None)
    rs.set_defaults(func=lambda a: cmd_loop(argparse.Namespace(
        config=a.config, toolchains=a.toolchains, strategies=a.strategies,
        max_cases=a.max_cases, duration=a.duration, seed=None, resume=True,
        generator=a.generator)))

    gc = sub.add_parser("gc", help="run garbage collection on the disk budget")
    gc.add_argument("--run", default=None)
    gc.set_defaults(func=cmd_gc)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
