"""Regression tests for generator_ir.reduce.

These tests are GPU-free: they load the pinned generator_ir case, run the
static validity gate, and exercise the candidate generators that guard loop
control variables.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from generator_ir.ir.nodes import Lit, Var
from generator_ir.ir.operators import Arith, Cmp, IncDec
from generator_ir.ir.program import Func, Program
from generator_ir.ir.statements import (
    Assign, Block, Declare, ExprStmt, For, IncDecStmt, While,
)
from generator_ir.ir.traversal import collect, traverse
from generator_ir.ir.typedecl import T, MemSpace, Qual
from generator_ir.reduce.ir_io import load_program_from_dir
from generator_ir.reduce.structure import program_counts
from generator_ir.reduce.transforms import (
    branch_simplify_candidates,
    constant_fold_candidates,
    cse_candidates,
    expr_lit_candidates,
    hdd_candidates_for_block,
    inline_single_use_candidates,
    merge_adjacent_loop_candidates,
    pure_dce_candidates,
    unreachable_delete_candidates,
)
from generator_ir.reduce.validity import validity_errors

REPO_ROOT = Path(__file__).resolve().parent.parent
PINNED_CASE = REPO_ROOT / "queue" / "done" / "generator_ir__350e41ea3b05302b"


def _make_loop_program() -> Program:
    p = Program()
    kernel = Func(
        name="kernel_main",
        params=[],
        ret=T.void,
        quals=[Qual.KERNEL],
        body=Block([
            Declare(name="w_0", gtype=T.i32, space=MemSpace.GENERIC,
                    init=Lit(T.i32, value=0)),
            Declare(name="acc", gtype=T.u32, space=MemSpace.GENERIC,
                    init=Lit(T.u32, value=0)),
            While(
                cond=Var(T.i32, name="w_0"),
                body=Block([
                    IncDecStmt(target=Var(T.i32, name="w_0"),
                               op=IncDec.INC, prefix=False),
                    ExprStmt(Var(T.u32, name="acc")),
                ]),
            ),
        ]),
    )
    p.add(kernel)
    return p


class TestOriginalCaseLoads(unittest.TestCase):
    def test_pinned_case_loads_and_is_valid_with_original_fallback(self):
        if not PINNED_CASE.exists():
            self.skipTest("pinned generator_ir case not present")
        program = load_program_from_dir(PINNED_CASE)
        self.assertGreater(program_counts(program)["stmt_count"], 0)
        errs = validity_errors(program, original=program)
        self.assertEqual(errs, [])


class TestLoopControlVarProtection(unittest.TestCase):
    def test_pure_dce_keeps_loop_control_write(self):
        program = _make_loop_program()
        candidates = list(pure_dce_candidates(program, max_candidates=32))
        # Every candidate must still contain exactly one IncDecStmt writing w_0.
        for cand in candidates:
            incs = [s for s in traverse(cand.program)
                    if isinstance(s, IncDecStmt)]
            self.assertEqual(len(incs), 1)
            self.assertEqual(incs[0].target.name, "w_0")

    def test_hdd_keeps_loop_control_write(self):
        program = _make_loop_program()
        kernel = program.fetch_kernel()
        body_block = kernel.body
        while_block = body_block.stmts[2].body
        candidates = list(hdd_candidates_for_block(program, while_block, max_candidates=64))
        self.assertGreater(len(candidates), 0)
        for cand in candidates:
            incs = [s for s in traverse(cand.program)
                    if isinstance(s, IncDecStmt)]
            self.assertEqual(len(incs), 1)
            self.assertEqual(incs[0].target.name, "w_0")


class TestCandidateGeneratorsRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not PINNED_CASE.exists():
            raise unittest.SkipTest("pinned generator_ir case not present")
        cls.program = load_program_from_dir(PINNED_CASE)

    def test_generators_are_lazy_and_do_not_crash(self):
        generators = [
            branch_simplify_candidates,
            unreachable_delete_candidates,
            merge_adjacent_loop_candidates,
            lambda p: expr_lit_candidates(p, max_candidates=32),
            lambda p: cse_candidates(p, max_candidates=16),
            lambda p: constant_fold_candidates(p, max_candidates=16),
            lambda p: inline_single_use_candidates(p, max_candidates=16),
        ]
        for gen in generators:
            # Some passes legitimately have no candidates on a given program;
            # the point is that iterating one step must not precompute all
            # candidates and must not crash.
            _ = next(gen(self.program), None)

    def test_validity_of_first_generated_candidates(self):
        # Static validity (with original fallback) should accept deletion-only
        # candidates.  Some CSE/literalization candidates may be invalid; the
        # oracle gate exists for exactly that reason.
        for gen in [
            branch_simplify_candidates,
            unreachable_delete_candidates,
            merge_adjacent_loop_candidates,
        ]:
            cand = next(gen(self.program), None)
            if cand is None:
                continue
            errs = validity_errors(cand.program, original=self.program)
            self.assertEqual(errs, [])


if __name__ == "__main__":
    unittest.main()
