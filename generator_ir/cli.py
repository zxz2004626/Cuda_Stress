"""gpuperf command-line interface.

Subcommands: generate / fuzz / mutate / sweep / benchmark / list.

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

log = logging.getLogger("gpuperf")


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


from generator_ir.cli_fuzz import (kx_cmd_fuzz, _fuzz_single_compiler,
    _fuzz_cross_compiler, _fuzz_opt_deopt, _fuzz_simple, _fuzz_static)
from generator_ir.cli_tools import (cmd_build, cmd_perturb, cmd_scan,
    cmd_bench, kx_cmd_list)

def kx_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpuperf",
        description="gpuperf -- GPU compiler performance-regression fuzzer")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("generate", help="generate kernels")
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--count", type=int, default=1)
    gen.add_argument("--template", type=str, default=None)
    gen.add_argument("--backend", choices=["cuda", "hip"], default="hip")
    gen.add_argument("--output-dir", type=str, default="./output")
    gen.add_argument("--complexity", choices=["low", "medium", "high", "stress"],
                     default="medium")
    gen.add_argument("--arch", type=str, default=None,
                     help="target arch (default: sm_75 for cuda, gfx908 for hip)")
    gen.add_argument("--ci", action="store_true")
    gen.add_argument("--clock64", action="store_true")
    gen.add_argument("--loop", action="store_true")
    gen.add_argument("--observable-fold", action="store_true")
    gen.add_argument("--allow-volatile", action="store_true")
    gen.add_argument("--codegen", action="store_true")
    gen.add_argument("--generic-baseline", action="store_true",
                     help="template-free generic grammar (isolated evaluation baseline)")
    gen.add_argument("--enable-feature-gens", action="store_true")
    gen.add_argument("--vary-launch", action="store_true")

    fuzz = sub.add_parser("fuzz", help="run the fuzzer")
    fuzz.add_argument("--seeds", type=int, default=100)
    fuzz.add_argument("--backend", choices=["cuda", "hip"], default="hip")
    fuzz.add_argument("--output-dir", type=str, default="./fuzz_output")
    fuzz.add_argument("--save-ir", action="store_true")
    fuzz.add_argument("--arch", type=str, default=None)
    fuzz.add_argument("--old-compiler", type=str, default=None)
    fuzz.add_argument("--new-compiler", type=str, default=None)
    fuzz.add_argument("--baseline-threshold", type=float, default=1.25)
    fuzz.add_argument("--opt-threshold", type=float, default=1.20)
    fuzz.add_argument("--deopt-threshold", type=float, default=1.20)
    fuzz.add_argument("--perf-runs", type=int, default=3)
    fuzz.add_argument("--seed", type=int, default=None)
    fuzz.add_argument("--templates", type=str, nargs="+", default=None)
    fuzz.add_argument("--complexity", choices=["low", "medium", "high", "stress"],
                      default="high")
    fuzz.add_argument("--opt-levels", type=str, nargs="+", default=None)
    fuzz.add_argument("--repeats", type=int, default=5)
    fuzz.add_argument("--timeout", type=int, default=300)
    fuzz.add_argument("--threshold", type=float, default=0.05)
    fuzz.add_argument("--no-mutations", action="store_true")
    fuzz.add_argument("--keep-tmp", action="store_true")
    fuzz.add_argument("--ci", action="store_true")
    fuzz.add_argument("--clock64", action="store_true")
    fuzz.add_argument("--loop", action="store_true")
    fuzz.add_argument("--profiler-mode", action="store_true")
    fuzz.add_argument("--corpus-dir", type=str, default=None)
    fuzz.add_argument("--fresh-ratio", type=float, default=0.3)
    fuzz.add_argument("--min-corpus-before-mutation", type=int, default=10)
    fuzz.add_argument("--profiler", type=str, default=None)
    fuzz.add_argument("--mutator-set", type=str, default="all",
                      choices=["paper", "ext", "all"])
    fuzz.add_argument("--mutation-mode", type=str, default="full",
                      choices=["mem", "loc", "reg", "full", "combo"])
    fuzz.add_argument("--simple-mutations", type=int, default=1)
    fuzz.add_argument("--simple-fresh-mutations", type=int, default=0)
    fuzz.add_argument("--observable-fold", action="store_true")
    fuzz.add_argument("--allow-volatile", action="store_true")
    fuzz.add_argument("--codegen", action="store_true")
    fuzz.add_argument("--generic-baseline", action="store_true",
                      help="fresh template-free cases; mutations/corpus are disabled")
    fuzz.add_argument("--enable-feature-gens", action="store_true")
    fuzz.add_argument("--vary-launch", action="store_true")
    fuzz.add_argument("--stop-on-failure", action="store_true")
    fuzz.add_argument("--keep-passing", action="store_true")
    fuzz.add_argument("--static", action="store_true")
    fuzz.add_argument("--method", type=str, default="opt-deopt",
                      choices=["opt-deopt", "simple"])

    mut = sub.add_parser("mutate", help="apply a named mutation")
    mut.add_argument("--seed", type=int, required=True)
    mut.add_argument("--mutation", type=str, required=True)
    mut.add_argument("--param", type=str, default=None)
    mut.add_argument("--backend", choices=["cuda", "hip"], default="hip")
    mut.add_argument("--output-dir", type=str, default="./mutated_output")
    mut.add_argument("--arch", type=str, default=None)
    mut.add_argument("--ci", action="store_true")
    mut.add_argument("--clock64", action="store_true")
    mut.add_argument("--loop", action="store_true")

    scan = sub.add_parser("sweep", help="sweep a mutation parameter")
    scan.add_argument("--seed", type=int, required=True)
    scan.add_argument("--mutation", type=str, required=True)
    scan.add_argument("--backend", choices=["cuda", "hip"], default="hip")
    scan.add_argument("--arch", type=str, default=None)
    scan.add_argument("--output-dir", type=str, default="./sweep_output")
    scan.add_argument("--opt-level", type=str, default="-O3")
    scan.add_argument("--ci", action="store_true")
    scan.add_argument("--clock64", action="store_true")

    bench = sub.add_parser("benchmark", help="benchmark a source file")
    bench.add_argument("--source", type=str, required=True)
    bench.add_argument("--opt-levels", type=str, nargs="+",
                       default=["-O0", "-O3"])
    bench.add_argument("--arch", type=str, default=None)
    bench.add_argument("--backend", choices=["cuda", "hip"], default=None)
    bench.add_argument("--repeats", type=int, default=5)

    lst = sub.add_parser("list", help="list mutations/templates")
    lst.add_argument("what", choices=["mutations", "templates", "all"])
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = kx_build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.WARNING
    else:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    if args.command is None:
        parser.print_help()
        return 0

    dispatch = {
        "generate": cmd_build,
        "fuzz": kx_cmd_fuzz,
        "mutate": cmd_perturb,
        "sweep": cmd_scan,
        "benchmark": cmd_bench,
        "list": kx_cmd_list,
    }
    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    try:
        return handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:
        log.error("fatal error: %s", exc)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
