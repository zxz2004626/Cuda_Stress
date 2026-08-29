"""Re-run the full ptxas matrix for a reduced generator_ir case.

Usage:
  python3 -m generator_ir.reduce.matrix_check --out <reduced_out_dir> \
      --case <original_case_dir> [--device 0]

The pinned reducer only ever checks one failing/reference pair.  This script
recompiles the reduced Program for every runnable toolchain x O0/O1/O2 profile
and prints the checksum partition, so a regression in the trigger matrix is
caught immediately.
"""

from __future__ import annotations

import argparse
import json
import pickle
import struct
import tempfile
from pathlib import Path
from typing import Dict, Optional

from generator.base import GeneratedCase
from generator_ir.emit.host_harness import cuda_driver_spec
from generator_ir.emit.nvidia import CudaEmitter
from generator_ir.ir.program import Program
from pipeline import compiler
from pipeline.config import Config, load_config
from pipeline.executor import run_build


def load_program_from_out(out_dir: Path) -> Program:
    for name in ("checkpoint_program.pkl", "program.pkl"):
        p = out_dir / name
        if p.exists():
            with open(p, "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(f"no program.pkl in {out_dir}")


def run_matrix(out_dir: Path, config: Config, device: int = 0) -> Dict:
    program = load_program_from_out(out_dir)
    src = CudaEmitter().rendersrc(program)
    kernel = program.fetch_kernel()
    spec = cuda_driver_spec(program, kernel)
    case = GeneratedCase(
        test_id="matrix_check", config_seed=0, test_seed=0,
        strategy="generator_ir", params={}, generator_version="matrix_check",
        kernel_seed=0, source=src, input_bytes=b"", expected_bytes=b"",
        grid=list(program.launch.grid), block=list(program.launch.block),
        input_words=0, output_words=1, output_words_per_thread=0,
        target_sm=program.launch.arch, cpu_oracle="cross_build_consensus",
        comparison_policy="cross_build_checksum_u64",
        cuda_features=[program.launch.arch, "generator_ir", "driver_checksum"],
        kernel_entry=kernel.name, sanitizer_policy="driver_checksum_harness",
        execution_mode="driver_checksum", oracle_mode="cross_build",
        execution_spec=spec,
    )
    results: Dict[str, Dict[str, Optional[int]]] = {}
    with tempfile.TemporaryDirectory(prefix="matrix_check_") as td:
        work = Path(td)
        src_path = work / "test.cu"
        src_path.write_text(src)
        for tc in config.runnable_toolchains():
            results[tc.id] = {}
            ptx_path = work / f"kernel_{tc.id}.ptx"
            rec = compiler.nvcc_ptx(tc, src_path, ptx_path, 120)
            if not rec.ok:
                results[tc.id]["error"] = f"nvcc: {rec.stderr[-200:]}"
                continue
            for prof in config.profiles:
                build_dir = work / f"{tc.id}_{prof.name}"
                build_dir.mkdir(parents=True, exist_ok=True)
                cubin_path = build_dir / "kernel.cubin"
                rec = compiler.ptxas_cubin(tc, prof, ptx_path, cubin_path,
                                           timeout=120)
                if not rec.ok:
                    results[tc.id][prof.name] = f"ptxas: {rec.stderr[-200:]}"
                    continue
                run = run_build(config, tc, case, build_dir, 0,
                                device=device, timeout=15)
                if run.status != "OK" or not run.output_words:
                    results[tc.id][prof.name] = f"run: {run.status} {run.error[-120:]}"
                else:
                    results[tc.id][prof.name] = struct.unpack("<Q", run.output_words)[0]
    return results


def partition(results: Dict) -> Dict[str, Dict[str, Optional[int]]]:
    groups: Dict[str, list] = {}
    for tc, profiles in results.items():
        values = tuple(sorted(v for v in profiles.values() if isinstance(v, int)))
        if not values:
            continue
        groups.setdefault(values, []).append(tc)
    return groups


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--toolchains", default=None)
    ap.add_argument("--strategies", default=None)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args(argv)
    config = load_config(config_path=args.config,
                         toolchains_path=args.toolchains,
                         strategies_path=args.strategies)
    results = run_matrix(Path(args.out), config, device=args.device)
    print(json.dumps(results, indent=2, sort_keys=True))
    groups = partition(results)
    print("partition:")
    for vals, tcs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        print(f"  {tcs}: {vals}")
    return 0


if __name__ == "__main__":
    main()
