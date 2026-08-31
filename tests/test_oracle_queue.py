"""CPU oracle verification and the directory task queue."""

import json
from pathlib import Path

import pytest

from pipeline.oracle import recompute_expected, oracle_audit, verify_input_sensitivity
from pipeline.manifest import write_testcase_dir
from pipeline.queue import TaskQueue
from generator.base import default_registry
from generator.strategies import CPU_ORACLE_ALL

PARAMS = {
    "predicate_pressure": {"num_comparisons": 32, "chain_len": 48, "num_diamonds": 12,
                           "num_folds": 24, "dup_groups": 4, "use_launch_bounds": False},
    "integer_chain": {"chain_len": 64, "use_mulhi": True, "use_funnelshift": True,
                      "use_bfe_bfi": True, "use_prmt": True, "use_lop3": True,
                      "boundary_words": 8},
    "memory_address": {"reads": 32, "widths": [8, 16, 32, 64], "vectors": True,
                       "cache_policy": "cg", "structs": True},
    "warp_mixed": {"warp_ops": ["ballot", "shfl_idx"], "chain_len": 48,
                   "partial_masks": False, "boundary_words": 8},
    "register_pressure": {"live_vars": 32, "chains": 4, "merge_len": 32,
                          "call_style": "noinline", "variant": "none"},
    "inline_ptx": {"n_instr": 40, "n_regs32": 6, "n_regs64": 4, "n_preds": 3,
                  "use_rare_ops": True, "use_immediates": True},
}


@pytest.mark.parametrize("cls", CPU_ORACLE_ALL, ids=lambda c: c.name)
def test_oracle_recompute_matches(tmp_path, cls):
    reg = default_registry()
    case = reg.get(cls.name).generate(20260810, 21, PARAMS[cls.name], "1.0.0", "sm_86")
    write_testcase_dir(tmp_path, case, "1.0.0", 20260810)
    audit = oracle_audit(tmp_path)
    assert audit["status"] == "OK"
    assert audit["oracle_recompute"]["match"] is True


@pytest.mark.parametrize("cls", CPU_ORACLE_ALL, ids=lambda c: c.name)
def test_input_sensitivity(tmp_path, cls):
    """A tweaked input must change the output for every strategy (real runtime dependence)."""
    reg = default_registry()
    case = reg.get(cls.name).generate(20260810, 22, PARAMS[cls.name], "1.0.0", "sm_86")
    write_testcase_dir(tmp_path, case, "1.0.0", 20260810)
    sens = verify_input_sensitivity(tmp_path, 20260810)
    assert sens["input_sensitive"] is True, f"{cls.name} output does not depend on input"


def test_queue_transitions(tmp_path):
    q = TaskQueue(tmp_path / "queue")
    # fabricate a case dir
    case_dir = tmp_path / "src_case"
    case_dir.mkdir()
    (case_dir / "manifest.json").write_text("{}")
    q.enter(case_dir)
    assert q.list("generated") == [case_dir.name]
    q.to_compiling(case_dir.name)
    assert q.stage_of(case_dir.name) == "compiling"
    q.to_ready(case_dir.name)
    assert q.stage_of(case_dir.name) == "ready_to_run"
    q.to_running(case_dir.name)
    q.to_triage(case_dir.name)
    assert q.stage_of(case_dir.name) == "triage"
    q.to_done(case_dir.name)
    assert q.stage_of(case_dir.name) == "done"


def test_queue_recover(tmp_path):
    q = TaskQueue(tmp_path / "queue")
    d = tmp_path / "c1"
    d.mkdir()
    (d / "manifest.json").write_text("{}")
    q.enter(d)
    q.to_compiling("c1")               # simulate crash while compiling
    moved = q.recover()
    assert moved.get("compiling->ready", 0) == 1
    assert q.stage_of("c1") == "ready_to_run"


def test_queue_state_roundtrip(tmp_path):
    q = TaskQueue(tmp_path / "queue")
    q.write_state({"test_counter": 42, "stats": {"triaged": 7}})
    st = q.read_state()
    assert st["test_counter"] == 42
    assert st["stats"]["triaged"] == 7
