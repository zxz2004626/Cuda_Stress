"""Command-line entry for the IR reducer.

    python3 -m generator_ir.reduce \\
        --case <testcase_dir> --failing cuda-13.3:O1 \\
        --reference cuda-13.3:O0 --out <out_dir>
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from generator.base import GeneratedCase
from generator_ir.ir.program import Program
from generator_ir.ir.traversal import clone_program, blocks
from generator_ir.reduce.ir_io import load_program_from_dir, save_reduced_program
from generator_ir.reduce.interesting import InterestingOracle
from generator_ir.reduce.structure import program_counts, structural_digest
from generator_ir.reduce.transforms import (
    Candidate, unused_decl_candidates, pure_dce_candidates,
    cf_delete_candidates, hdd_candidates_for_block, expr_lit_candidates,
    repeat_candidates, branch_simplify_candidates,
    unreachable_delete_candidates, merge_adjacent_loop_candidates,
    cse_candidates, constant_fold_candidates, inline_single_use_candidates,
    clear_path_cache,
)
from pipeline.config import Config, load_config


def _parse_pair(text: str) -> tuple:
    if ":" not in text:
        raise ValueError(f"expected toolchain:profile, got {text!r}")
    tc_id, profile = text.rsplit(":", 1)
    return tc_id, profile


def _load_original_meta(case_dir: Path) -> Dict[str, Any]:
    m = json.loads((case_dir / "manifest.json").read_text())
    return m


def _make_manifest(case_dir: Path, program: Program, src: str) -> Dict[str, Any]:
    from generator_ir.emit.nvidia import CudaEmitter
    from generator_ir.emit.host_harness import cuda_driver_spec
    meta = _load_original_meta(case_dir)
    kernel = program.fetch_kernel()
    spec = cuda_driver_spec(program, kernel)
    params = meta.get("strategy_params") or {}
    template = params.get("template", "unknown")
    case = GeneratedCase(
        test_id=f"{meta.get('test_id', 'generator_ir_reduced')}__reduced",
        config_seed=int(meta.get("config_seed", 0)),
        test_seed=int(meta.get("test_seed", 0)),
        strategy="generator_ir",
        params=params,
        generator_version=str(meta.get("generator_version", "ir_reduce")),
        kernel_seed=int(meta.get("kernel_seed", 0)),
        source=src,
        input_bytes=b"",
        expected_bytes=b"",
        grid=list(program.launch.grid),
        block=list(program.launch.block),
        input_words=0,
        output_words=1,
        output_words_per_thread=0,
        target_sm=str(program.launch.arch),
        cpu_oracle="cross_build_consensus",
        comparison_policy="cross_build_checksum_u64",
        cuda_features=[program.launch.arch, "generator_ir", template,
                       "driver_checksum"],
        kernel_entry=kernel.name,
        sanitizer_policy="driver_checksum_harness",
        execution_mode="driver_checksum",
        oracle_mode="cross_build",
        execution_spec=spec,
        ir_blob=pickle.dumps(program, protocol=pickle.HIGHEST_PROTOCOL),
    )
    return case.manifest()


def _write_log(log_path: Path, lines: List[Dict[str, Any]]) -> None:
    with open(log_path, "w") as f:
        for line in lines:
            f.write(json.dumps(line, sort_keys=True) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="python3 -m generator_ir.reduce",
        description="UB-preserving generator_ir differential-case minimizer")
    p.add_argument("--case", required=True, help="input generator_ir testcase dir")
    p.add_argument("--failing", required=True, help="toolchain:profile pinning the buggy build")
    p.add_argument("--reference", required=True, help="toolchain:profile pinning the correct build")
    p.add_argument("--out", required=True, help="output directory (not the input case dir)")
    p.add_argument("--timeout", type=int, default=180,
                   help="fallback timeout in seconds (compile and run)")
    p.add_argument("--compile-timeout", type=int, default=None,
                   help="nvcc/ptxas timeout in seconds (default 120)")
    p.add_argument("--run-timeout", type=int, default=None,
                   help="driver_checksum run timeout in seconds (default 15)")
    p.add_argument("--max-tests", type=int, default=500, help="candidate budget")
    p.add_argument("--resume", action="store_true",
                   help="resume from the latest checkpoint in --out if present")
    p.add_argument("--keep-work", action="store_true", help="keep intermediate .cu/cubin")
    p.add_argument("--dry-run-checks", action="store_true",
                   help="only run the static UB gate, no GPU calls (for unit tests)")
    p.add_argument("--no-input-salt-check", action="store_true",
                   help="skip the input-sensitivity oracle run")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--config", default=None)
    p.add_argument("--toolchains", default=None)
    p.add_argument("--strategies", default=None)
    args = p.parse_args(argv)

    case_dir = Path(args.case).resolve()
    out_dir = Path(args.out).resolve()
    if out_dir == case_dir:
        print("--out must differ from --case (refusing to overwrite input)", file=sys.stderr)
        return 2
    if not (case_dir / "manifest.json").exists():
        print(f"no manifest.json in {case_dir}", file=sys.stderr)
        return 2

    try:
        config: Config = load_config(config_path=args.config,
                                     toolchains_path=args.toolchains,
                                     strategies_path=args.strategies)
        failing_tc_id, failing_profile = _parse_pair(args.failing)
        reference_tc_id, reference_profile = _parse_pair(args.reference)
        failing_tc = config.find_toolchain(failing_tc_id)
        reference_tc = config.find_toolchain(reference_tc_id)
        failing_prof = config.profile_by_name(failing_profile)
        reference_prof = config.profile_by_name(reference_profile)
        if not failing_tc.runnable or not reference_tc.runnable:
            raise ValueError("pinned toolchains must be runnable")
    except Exception as exc:  # noqa: BLE001
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    print(f"loading IR from {case_dir}")
    try:
        original = load_program_from_dir(case_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"IR load failed: {exc}", file=sys.stderr)
        return 1

    meta = _load_original_meta(case_dir)
    if meta.get("execution_mode") != "driver_checksum":
        print(f"execution_mode is {meta.get('execution_mode')!r}; only "
              "driver_checksum generator_ir cases are supported", file=sys.stderr)
        return 2
    if meta.get("strategy") != "generator_ir":
        print(f"strategy is {meta.get('strategy')!r}; only generator_ir "
              "cases are supported", file=sys.stderr)
        return 2

    work_root = out_dir / "_work" if args.keep_work else None
    compile_timeout = args.compile_timeout or args.timeout
    run_timeout = args.run_timeout or args.timeout
    oracle = InterestingOracle(
        config, failing_tc, failing_prof, reference_tc, reference_prof,
        original=original, timeout=args.timeout,
        compile_timeout=compile_timeout, run_timeout=run_timeout,
        keep_work=args.keep_work,
        work_root=work_root, dry_run=args.dry_run_checks,
        check_input_salt=not args.no_input_salt_check, device=args.device)

    out_dir.mkdir(parents=True, exist_ok=True)
    log_lines: List[Dict[str, Any]] = []
    stats = {"candidates": 0, "accepted": 0, "rejected_invalid": 0,
             "rejected_no_diverge": 0, "rejected_other": 0,
             "quick_ok": 0, "confirm_ok": 0}
    current = clone_program(original)
    stopped = [False]
    rounds = 0
    original_checksums: Dict[str, Any] = {}
    original_source_lines = 0

    def save_checkpoint() -> None:
        from generator_ir.emit.nvidia import CudaEmitter
        try:
            (out_dir / "checkpoint_program.pkl").write_bytes(
                pickle.dumps(current, protocol=pickle.HIGHEST_PROTOCOL))
            (out_dir / "textreduced.cu").write_text(CudaEmitter().rendersrc(current))
            _write_log(out_dir / "reduce_log.jsonl", log_lines)
            (out_dir / "checkpoint.json").write_text(json.dumps({
                "rounds": rounds,
                "stats": stats,
                "original_checksums": original_checksums,
                "original_source_lines": original_source_lines,
            }, indent=2, sort_keys=True))
        except Exception as exc:  # noqa: BLE001
            print(f"checkpoint write failed: {exc}", file=sys.stderr)

    if args.resume and (out_dir / "checkpoint_program.pkl").exists():
        print(f"resuming from checkpoint in {out_dir}")
        try:
            with open(out_dir / "checkpoint_program.pkl", "rb") as f:
                current = pickle.load(f)
            if not isinstance(current, Program):
                raise ValueError("checkpoint_program.pkl is not a Program")
            if (out_dir / "reduce_log.jsonl").exists():
                log_lines = [json.loads(line) for line in
                             (out_dir / "reduce_log.jsonl").read_text().splitlines()
                             if line.strip()]
            if (out_dir / "checkpoint.json").exists():
                cp = json.loads((out_dir / "checkpoint.json").read_text())
                rounds = int(cp.get("rounds", 0))
                stats = cp.get("stats", stats)
                original_checksums = cp.get("original_checksums", {})
                original_source_lines = int(cp.get("original_source_lines", 0))
        except Exception as exc:  # noqa: BLE001
            print(f"resume failed ({exc}); starting from original", file=sys.stderr)
            current = clone_program(original)
            log_lines = []
            stats = {"candidates": 0, "accepted": 0, "rejected_invalid": 0,
                     "rejected_no_diverge": 0, "rejected_other": 0,
                     "quick_ok": 0, "confirm_ok": 0}
            rounds = 0
            original_checksums = {}
            original_source_lines = 0

    if not original_checksums:
        print(f"checking original IR against {args.failing} vs {args.reference}")
        original_check = oracle.check(original, runs=1)
        if not original_check.ok:
            print(f"original case is not interesting with pinned builds: "
                  f"{original_check.reason}", file=sys.stderr)
            return 1
        original_checksums = original_check.checksums
        original_source_lines = original_check.source_lines
        print(f"original: stmt={program_counts(original)['stmt_count']} "
              f"checksums={original_check.checksums}")
    else:
        original_check = None
        print(f"resumed: original_checksums={original_checksums} "
              f"current stmt={program_counts(current)['stmt_count']}")

    def attempts() -> int:
        return stats["candidates"]

    def attempt(desc: str, cand: Program) -> Optional[Program]:
        if stopped[0] or stats["candidates"] >= args.max_tests:
            stopped[0] = True
            return None
        stats["candidates"] += 1
        # Fast screen: one run per build, no input-salt pass.
        quick = oracle.check(cand, runs=1, input_salt_check=False)
        if quick.ok and not oracle.dry_run:
            stats["quick_ok"] += 1
            # Re-verify with three runs per build and input-sensitivity
            # enabled.  A nondeterministic-but-interesting candidate can pass
            # runs=2 and then fail the final runs=3 confirmation, so the
            # acceptance gate must be at least as strict as the final gate.
            res = oracle.check(cand, runs=3, input_salt_check=True)
            if res.ok:
                stats["confirm_ok"] += 1
        else:
            res = quick
        counts = program_counts(cand)
        entry = {
            "step": stats["candidates"],
            "transform": desc,
            "accepted": res.ok,
            "reason": res.reason,
            "stmt_count": counts["stmt_count"],
            "expr_count": counts["expr_count"],
            "rawcode_count": counts["rawcode_count"],
            "cu_lines": res.source_lines,
            "checksums": res.checksums,
        }
        log_lines.append(entry)
        if stats["candidates"] % 10 == 0 or res.ok or res.invalid:
            print(f"  [{stats['candidates']}/{args.max_tests}] {desc}: "
                  f"{'OK' if res.ok else res.reason} "
                  f"stmt={counts['stmt_count']} expr={counts['expr_count']}",
                  file=sys.stderr, flush=True)
        if res.ok:
            stats["accepted"] += 1
            save_checkpoint()
            return cand
        if res.invalid:
            stats["rejected_invalid"] += 1
        elif "diverge" in res.reason or "converged" in res.reason or "input-sensitive" in res.reason:
            stats["rejected_no_diverge"] += 1
        else:
            stats["rejected_other"] += 1
        return None

    def run_fixedpoint(name: str, make_cands, current: Program,
                        budget: Optional[int] = None) -> tuple:
        """Run one pass until no candidate is accepted or *budget* attempts
        have been spent in this pass.

        The budget prevents one pass (in particular block-hdd) from starving
        the later passes inside an outer round; the outer loop brings every
        pass back after the others have had a turn.
        """
        changed = False
        pass_attempts = 0
        while not stopped[0]:
            clear_path_cache()
            accepted_any = False
            cands = make_cands(current)
            for cand in cands:
                if stopped[0]:
                    break
                if budget is not None and pass_attempts >= budget:
                    break
                accepted = attempt(cand.desc, cand.program)
                pass_attempts += 1
                if accepted is not None:
                    current = accepted
                    changed = True
                    accepted_any = True
                    break  # candidate list is stale after acceptance
            if not accepted_any:
                break
            if budget is not None and pass_attempts >= budget:
                break
        return current, changed

    def all_hdd_candidates(program: Program):
        from generator_ir.reduce.transforms import _loop_control_vars
        loop_vars = _loop_control_vars(program)
        for block in blocks(program):
            yield from hdd_candidates_for_block(
                program, block, max_candidates=max(8, min(64, args.max_tests)),
                loop_vars=loop_vars)

    def run_simple_pass(name: str, make_cands, current: Program,
                         budget: Optional[int] = None):
        return run_fixedpoint(name, make_cands, current, budget=budget)

    def run_hdd_pass(current: Program):
        if stopped[0]:
            return current, False
        return run_fixedpoint("block-hdd", all_hdd_candidates, current)

    def run_expr_lit_pass(current: Program):
        if stopped[0]:
            return current, False
        return run_fixedpoint(
            "expr-lit",
            lambda p: expr_lit_candidates(p, max_candidates=min(192, args.max_tests)),
            current)

    def run_repeat_pass(current: Program):
        if stopped[0]:
            return current, False
        return run_fixedpoint("reduce-repeat", repeat_candidates, current)

    pass_order = [
        ("unused-decl", lambda c: run_simple_pass("unused-decl", unused_decl_candidates, c, budget=40)),
        ("cf-delete", lambda c: run_simple_pass("cf-delete", cf_delete_candidates, c, budget=120)),
        ("block-hdd", lambda c: run_fixedpoint(
            "block-hdd", all_hdd_candidates, c, budget=120)),
        ("pure-dce", lambda c: run_fixedpoint(
            "pure-dce",
            lambda p: pure_dce_candidates(p, max_candidates=min(256, args.max_tests)),
            c, budget=120)),
        ("branch-simplify", lambda c: run_simple_pass("branch-simplify", branch_simplify_candidates, c, budget=20)),
        ("unreachable-delete", lambda c: run_simple_pass("unreachable-delete", unreachable_delete_candidates, c, budget=20)),
        ("merge-adjacent-loops", lambda c: run_simple_pass("merge-adjacent-loops", merge_adjacent_loop_candidates, c, budget=20)),
        ("expr-lit", lambda c: run_fixedpoint(
            "expr-lit",
            lambda p: expr_lit_candidates(p, max_candidates=min(192, args.max_tests)),
            c, budget=40)),
        ("cse", lambda c: run_fixedpoint(
            "cse",
            lambda p: cse_candidates(p, max_candidates=min(48, args.max_tests)),
            c, budget=20)),
        ("const-fold", lambda c: run_fixedpoint(
            "const-fold",
            lambda p: constant_fold_candidates(p, max_candidates=min(64, args.max_tests)),
            c, budget=30)),
        ("inline-single-use", lambda c: run_fixedpoint(
            "inline-single-use",
            lambda p: inline_single_use_candidates(p, max_candidates=min(64, args.max_tests)),
            c, budget=30)),
        ("reduce-repeat", lambda c: run_fixedpoint("reduce-repeat", repeat_candidates, c, budget=20)),
    ]

    while not stopped[0]:
        progressed = False
        rounds += 1
        for _name, pass_fn in pass_order:
            if stopped[0]:
                break
            new_current, changed = pass_fn(current)
            if changed:
                current = new_current
                progressed = True
        if not progressed:
            break
        if rounds > 5000:
            print("reducer pass budget exhausted", file=sys.stderr)
            break

    print(f"reduction finished: {stats['candidates']} candidates, "
          f"{stats['accepted']} accepted")

    # Final artifact emission + stable-divergence confirmation.
    from generator_ir.emit.nvidia import CudaEmitter
    from generator_ir.emit.host_harness import cuda_driver_spec

    reduced_src = CudaEmitter().rendersrc(current)
    kernel = current.fetch_kernel()
    spec = cuda_driver_spec(current, kernel)

    final_result = None
    if not args.dry_run_checks:
        print("final confirmation (3 runs per build + input-salt check)")
        final_result = oracle.check(current, runs=3, input_salt_check=True)
        if not final_result.ok:
            print(f"final confirmation failed: {final_result.reason}", file=sys.stderr)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "textreduced.cu").write_text(reduced_src)
    save_reduced_program(out_dir, current)
    manifest = _make_manifest(case_dir, current, reduced_src)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True))
    _write_log(out_dir / "reduce_log.jsonl", log_lines)

    orig_counts = program_counts(original)
    final_counts = program_counts(current)
    report = {
        "case_dir": str(case_dir),
        "out_dir": str(out_dir),
        "failing": args.failing,
        "reference": args.reference,
        "original": {
            "stmt_count": orig_counts["stmt_count"],
            "expr_count": orig_counts["expr_count"],
            "rawcode_count": orig_counts["rawcode_count"],
            "cu_lines": original_source_lines,
        },
        "final": {
            "stmt_count": final_counts["stmt_count"],
            "expr_count": final_counts["expr_count"],
            "rawcode_count": final_counts["rawcode_count"],
            "cu_lines": reduced_src.count("\n") + 1,
        },
        "tests": {
            "candidates": stats["candidates"],
            "accepted": stats["accepted"],
            "rejected_invalid": stats["rejected_invalid"],
            "rejected_no_diverge": stats["rejected_no_diverge"],
            "rejected_other": stats["rejected_other"],
        },
        "original_checksums": original_checksums,
        "final_checksums": final_result.checksums if final_result is not None else {},
        "still_diverges": bool(final_result and final_result.ok),
        "final_note": final_result.reason if final_result is not None else "dry-run",
        "sanitizer": "not_run",
        "digest": structural_digest(current),
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True))

    print(f"wrote {out_dir}")
    if args.dry_run_checks:
        print("dry-run complete (static UB gate only)")
        return 0
    if final_result is None or not final_result.ok:
        print("final confirmation did not pass; artifacts written but "
              "not claiming a stable reduction", file=sys.stderr)
        return 1
    print(f"stable divergence confirmed: {final_result.checksums}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
