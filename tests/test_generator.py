"""Generator strategies: deterministic, valid, byte-reproducible, oracle-mirrored."""

import hashlib
import json

import pytest

from generator.base import default_registry
from generator.strategies import ALL

PARAMS = {
    "predicate_pressure": {"num_comparisons": 32, "chain_len": 48, "num_diamonds": 12,
                           "num_folds": 24, "dup_groups": 4, "use_launch_bounds": False},
    "integer_chain": {"chain_len": 64, "use_mulhi": True, "use_funnelshift": True,
                      "use_bfe_bfi": True, "use_prmt": True, "use_lop3": True,
                      "boundary_words": 8},
    "memory_address": {"reads": 32, "widths": [8, 16, 32, 64], "vectors": True,
                       "cache_policy": "cg", "structs": True},
    "warp_mixed": {"warp_ops": ["ballot", "shfl_idx", "match_any"], "chain_len": 48,
                   "partial_masks": False, "boundary_words": 8},
    "register_pressure": {"live_vars": 32, "chains": 4, "merge_len": 32,
                          "call_style": "noinline", "variant": "none"},
    "inline_ptx": {"n_instr": 40, "n_regs32": 6, "n_regs64": 4, "n_preds": 3,
                  "use_rare_ops": True, "use_immediates": True},
    "generator_ir": {"template": "map_transform", "complexity": "low",
                     "amplify_iters": 1, "codegen": False},
}


@pytest.mark.parametrize("cls", ALL, ids=lambda c: c.name)
def test_strategy_generates_deterministically(cls):
    reg = default_registry()
    strat = reg.get(cls.name)
    params = PARAMS[cls.name]
    c1 = strat.generate(20260810, 7, params, "1.0.0", "sm_86")
    c2 = strat.generate(20260810, 7, params, "1.0.0", "sm_86")
    assert c1.source == c2.source
    assert c1.input_bytes == c2.input_bytes
    assert c1.expected_bytes == c2.expected_bytes
    assert c1.valid
    # output length must match grid*block*owpt
    if c1.oracle_mode == "cpu_exact":
        assert len(c1.expected_bytes) == c1.total_threads() * c1.output_words_per_thread * 8
    else:
        assert c1.expected_bytes == b""
        assert c1.execution_mode == "standalone_checksum"
        assert c1.source.count("<<<grid, block") == 1


@pytest.mark.parametrize("cls", ALL, ids=lambda c: c.name)
def test_strategy_case_id_stable(cls):
    reg = default_registry()
    strat = reg.get(cls.name)
    c1 = strat.generate(1, 2, PARAMS[cls.name], "1.0.0", "sm_86")
    c2 = strat.generate(1, 2, PARAMS[cls.name], "1.0.0", "sm_86")
    assert c1.case_id() == c2.case_id()


@pytest.mark.parametrize("cls", ALL, ids=lambda c: c.name)
def test_manifest_roundtrip(cls):
    from pipeline.manifest import case_from_dir, write_testcase_dir
    import tempfile, pathlib
    reg = default_registry()
    strat = reg.get(cls.name)
    case = strat.generate(20260810, 9, PARAMS[cls.name], "1.0.0", "sm_86")
    with tempfile.TemporaryDirectory() as td:
        write_testcase_dir(pathlib.Path(td), case, "1.0.0", 20260810)
        back = case_from_dir(pathlib.Path(td))
    assert back.source == case.source
    assert back.input_bytes == case.input_bytes
    assert back.expected_bytes == case.expected_bytes
    assert back.kernel_seed == case.kernel_seed
    assert back.execution_mode == case.execution_mode
    assert back.oracle_mode == case.oracle_mode


def test_register_pressure_maxrregcount_flag():
    reg = default_registry()
    strat = reg.get("register_pressure")
    case = strat.generate(5, 5, {"live_vars": 64, "chains": 4, "merge_len": 32,
                                 "call_style": "noinline", "variant": "maxrregcount"},
                          "1.0.0", "sm_86")
    assert case.params.get("ptxas_extra") == ["--maxrregcount=40"]
