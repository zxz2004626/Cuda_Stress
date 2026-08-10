"""Differential comparison semantics: stability, oracle, cross-build divergence."""

import hashlib
import json
import struct

import pytest

from pipeline import comparator
from pipeline.config import load_config, Toolchain, OptimizationProfile
from pipeline.models import Status, Category
from generator.base import GeneratedCase, default_registry

MAGIC = 0x4355434553545245
CANARY = 0xDEADBEEFCAFEBABE
TAIL = 0xCAFEBABEDEADBEEF


def _blob(words):
    data = struct.pack("<QQIIQQ", MAGIC, 0, 0, 0, len(words), CANARY)
    for w in words:
        data += struct.pack("<Q", w)
    data += struct.pack("<Q", TAIL)
    return data


def _make_case(tmp_path):
    reg = default_registry()
    case = reg.get("integer_chain").generate(
        20260810, 3, {"chain_len": 64, "use_mulhi": True, "use_funnelshift": True,
                      "use_bfe_bfi": True, "use_prmt": True, "use_lop3": True,
                      "boundary_words": 8}, "1.0.0", "sm_86")
    test_dir = tmp_path / "case"
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / "test.cu").write_text(case.source)
    (test_dir / "input.bin").write_bytes(case.input_bytes)
    (test_dir / "expected.bin").write_bytes(case.expected_bytes)
    return case, test_dir


def _add_build(test_dir, key, words, ok=True, runs=(1, 2, 3)):
    b = test_dir / "builds" / key
    b.mkdir(parents=True, exist_ok=True)
    (b / "compile_result.json").write_text(json.dumps({"ok": ok, "error": None,
                                                       "registers": 10}))
    for i in runs:
        (b / f"output_run_{i}.bin").write_bytes(_blob(words))


def _cfg_runnable_any():
    cfg = load_config()
    # force only one runnable toolchain so tests don't depend on the host toolchains
    cfg.toolchains = [Toolchain(id="cuda-12.6", root="/x", nvcc="/x/nvcc", ptxas="/x/ptxas",
                                nvdisasm="/x", cuobjdump="/x", arch="sm_86",
                                ccbin="/usr/bin/gcc", runnable=True)]
    cfg.profiles = [OptimizationProfile("O0", 0)]
    return cfg


def _expected_words(case):
    n = len(case.expected_bytes) // 8
    return list(struct.unpack(f"<{n}Q", case.expected_bytes))


def test_all_pass(tmp_path):
    case, td = _make_case(tmp_path)
    exp = _expected_words(case)
    _add_build(td, "cuda-12.6__O0", exp)
    r = comparator.compare_matrix(_cfg_runnable_any(), case, td)
    assert r.status == Status.PASS


def test_single_wrong_build_is_differential(tmp_path):
    case, td = _make_case(tmp_path)
    exp = _expected_words(case)
    wrong = list(exp)
    wrong[0] ^= 0xFFFFFFFFFFFFFFFF
    _add_build(td, "cuda-12.6__O0", exp)      # correct build
    _add_build(td, "cuda-12.6__O1", wrong)    # stable wrong build
    cfg = _cfg_runnable_any()
    cfg.profiles = [OptimizationProfile("O0", 0), OptimizationProfile("O1", 1)]
    r = comparator.compare_matrix(cfg, case, td)
    assert r.status == Status.POTENTIAL_COMPILER_BUG
    assert r.category == Category.POTENTIAL_BUG


def test_nondeterministic_build(tmp_path):
    case, td = _make_case(tmp_path)
    exp = _expected_words(case)
    _add_build(td, "cuda-12.6__O0", exp, runs=(1, 2))
    # run 3 differs => build unstable
    (td / "builds" / "cuda-12.6__O0" / "output_run_3.bin").write_bytes(
        _blob([w ^ 0x1 for w in exp]))
    r = comparator.compare_matrix(_cfg_runnable_any(), case, td)
    assert r.status == Status.NONDETERMINISTIC
    assert r.category == Category.FLAKY


def test_all_agree_but_wrong_oracle(tmp_path):
    case, td = _make_case(tmp_path)
    wrong = [w ^ 0x5 for w in _expected_words(case)]
    _add_build(td, "cuda-12.6__O0", wrong)
    r = comparator.compare_matrix(_cfg_runnable_any(), case, td)
    assert r.status == Status.ORACLE_MISMATCH
    assert r.category == Category.POTENTIAL_BUG


def test_unsupported_toolchain(tmp_path):
    case, td = _make_case(tmp_path)
    _add_build(td, "cuda-99.0__O0", _expected_words(case))
    cfg = _cfg_runnable_any()  # cuda-99.0 not runnable
    r = comparator.compare_matrix(cfg, case, td)
    assert r.status == Status.INCOMPLETE_MATRIX   # no runnable builds => unvalidated
    assert r.builds["cuda-99.0__O0"].status == Status.UNSUPPORTED


def test_compile_failed(tmp_path):
    case, td = _make_case(tmp_path)
    _add_build(td, "cuda-12.6__O0", [], ok=False)
    r = comparator.compare_matrix(_cfg_runnable_any(), case, td)
    assert r.status == Status.COMPILE_FAILED


def test_first_diff_word():
    assert comparator.first_diff_word(struct.pack("<2Q", 1, 2), struct.pack("<2Q", 1, 3)) == (1, 2, 3)
    assert comparator.first_diff_word(struct.pack("<2Q", 1, 2), struct.pack("<2Q", 1, 2)) is None
