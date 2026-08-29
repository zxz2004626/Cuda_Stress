"""Tests for generator_ir/reduce: validity gate and dry-run CLI."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from generator.base import default_registry
from generator_ir.ir.nodes import BinOp, Cast, Lit, RelOp, Subscript, ThreadBuiltin, Var
from generator_ir.ir.operators import Arith, Cmp, SafeOp
from generator_ir.ir.program import Func, LaunchConfig, Program
from generator_ir.ir.statements import Assign, Barrier, Block, Declare, If
from generator_ir.ir.typedecl import Qual, T, TPointer
from generator_ir.reduce.validity import valid_ub, validity_errors
from generator_ir.reduce.transforms import unused_decl_candidates
from generator_ir.ir.traversal import clone_program
from pipeline.manifest import write_testcase_dir


def _kernel_with_body(body):
    prog = Program(LaunchConfig(arch="sm_86"))
    prog.add(Func(name="kernel_main", params=[], ret=T.void,
                  body=body, quals=[Qual.KERNEL]))
    return prog


def test_valid_ub_rejects_free_variable():
    p = _kernel_with_body(Block([
        Assign(Var(T.u32, name="x"), Lit(T.u32, value=1)),
    ]))
    assert not valid_ub(p, emit_check=False)
    assert any("free variable" in e for e in validity_errors(p, emit_check=False))


def test_valid_ub_rejects_uninitialized_read():
    p = _kernel_with_body(Block([
        Declare(name="x", gtype=T.u32, init=None),
        Assign(Var(T.u32, name="x"), Var(T.u32, name="x")),
    ]))
    assert not valid_ub(p, emit_check=False)
    assert any("uninitialized" in e for e in validity_errors(p, emit_check=False))


def test_valid_ub_rejects_unguarded_div_and_shift():
    # Division by an unknown kernel parameter must be guarded.
    p = _kernel_with_body(Block([
        Declare(name="x", gtype=T.u32, init=Lit(T.u32, value=10)),
        Assign(Var(T.u32, name="x"),
               BinOp(T.u32, op=Arith.DIV, lhs=Var(T.u32, name="x"),
                     rhs=Var(T.i32, name="n"))),
    ]))
    errs = validity_errors(p, emit_check=False)
    assert any("DIV" in e for e in errs)

    p = _kernel_with_body(Block([
        Declare(name="x", gtype=T.u32, init=Lit(T.u32, value=1)),
        Assign(Var(T.u32, name="x"),
               BinOp(T.u32, op=Arith.SHL, lhs=Var(T.u32, name="x"),
                     rhs=Var(T.i32, name="n"))),
    ]))
    errs = validity_errors(p, emit_check=False)
    assert any("SHL" in e for e in errs)


def test_valid_ub_rejects_divergent_barrier():
    p = _kernel_with_body(Block([
        If(cond=RelOp(T.boolean, op=Cmp.LT,
                      lhs=ThreadBuiltin(T.u32, axis="x", source="threadIdx"),
                      rhs=Lit(T.u32, value=1)),
           then=Block([Barrier()])),
    ]))
    assert not valid_ub(p, emit_check=False)
    assert any("divergent barrier" in e for e in validity_errors(p, emit_check=False))


def test_valid_ub_rejects_unsafe_subscript_index():
    p = Program(LaunchConfig(arch="sm_86"))
    p.add(Func(name="kernel_main", params=[
        Declare(name="output", gtype=TPointer(T.u32)),
        Declare(name="n", gtype=T.i32),
    ], ret=T.void, body=Block([
        Assign(Subscript(T.u32, array=Var(TPointer(T.u32), name="output"),
                         index=Var(T.u32, name="n")),
               Lit(T.u32, value=1)),
    ]), quals=[Qual.KERNEL]))
    errs = validity_errors(p, emit_check=False)
    assert any("unsafe subscript" in e for e in errs)


def test_whitelist_unused_decl_removal_keeps_ub_free():
    p = _kernel_with_body(Block([
        Declare(name="used", gtype=T.u32, init=Lit(T.u32, value=1)),
        Declare(name="unused", gtype=T.u32, init=Lit(T.u32, value=2)),
        Assign(Var(T.u32, name="used"), Var(T.u32, name="used")),
    ]))
    cands = unused_decl_candidates(p)
    assert any("unused" in c.desc for c in cands)
    reduced = next(c.program for c in cands if "unused" in c.desc)
    assert valid_ub(reduced, emit_check=False)


def test_deleting_live_decl_is_rejected_not_repaired():
    p = _kernel_with_body(Block([
        Declare(name="live", gtype=T.u32, init=Lit(T.u32, value=7)),
        Assign(Var(T.u32, name="live"), Var(T.u32, name="live")),
    ]))
    # Simulate the forbidden repair path: deleting the live Declare leaves a
    # dangling Var.  The candidate must be rejected and the name must not be
    # rebound to another variable.
    from generator_ir.reduce.transforms import _path_to, _get_at_path
    bad = clone_program(p)
    from generator_ir.ir.traversal import blocks
    block = next(iter(blocks(bad)))
    block.stmts.pop(0)
    assert not valid_ub(bad, emit_check=False)
    assert any("free variable" in e for e in validity_errors(bad, emit_check=False))


def test_cli_dry_run_loads_and_reports_sizes(tmp_path):
    strategy = default_registry().get("generator_ir")
    case = strategy.generate(
        101, 202, {
            "template": "map_transform",
            "complexity": "low",
            "amplify_iters": 1,
            "codegen": False,
        }, "test", "sm_86")
    case_dir = tmp_path / "case"
    out_dir = tmp_path / "out"
    write_testcase_dir(case_dir, case, "test", 101)

    r = subprocess.run(
        [sys.executable, "-m", "generator_ir.reduce",
         "--case", str(case_dir),
         "--failing", "cuda-12.5:O1", "--reference", "cuda-12.5:O0",
         "--out", str(out_dir),
         "--dry-run-checks", "--max-tests", "10"],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-1000:]
    assert (out_dir / "textreduced.cu").exists()
    assert (out_dir / "program.pkl").exists()
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "report.json").exists()
    report = json.loads((out_dir / "report.json").read_text())
    assert report["original"]["stmt_count"] > 0
    assert report["final"]["stmt_count"] > 0


@pytest.mark.skipif(
    not (ROOT / "poc" / "miscompile-fe535f18-family-full").exists(),
    reason="no reproducible generator_ir GPU miscompile case available")
def test_gpu_reducer_preserves_pinned_divergence():
    pytest.skip("GPU-dependent reproduction requires a confirmed generator_ir case; "
                "do not fake green without a GPU")
