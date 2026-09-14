"""CPU C oracle backend (generator_ir.emit.cpu_oracle) tests.

Covers capability gating, emission/compile/execute round trips, determinism,
strategy integration (oracle_mode=cpu_emitted vs cross_build fallback), the
manifest round trip, and — when a GPU + nvcc are present — end-to-end
agreement between the emitted .cu harness checksum and the oracle checksum.
"""

import os
import re
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from generator.base import default_registry
from generator_ir.gen.assembler import build_kernel
from generator_ir.gen.options import GenConfig
from generator_ir.emit.cpu_oracle import (
    CpuOracleUnsupported, cpu_oracle_checksums, run_cpu_oracle_source,
    scan_program, translate_cpu_oracle,
)
from pipeline.manifest import case_from_dir, write_testcase_dir

GXX = shutil.which("g++")
pytestmark = pytest.mark.skipif(GXX is None, reason="g++ is required")


def _program(seed=7, template=None, **kw):
    cfg = GenConfig(
        seed=seed, complexity=kw.pop("complexity", "medium"), target="cuda",
        isa="sm_75", observable_fold=True, checksum_only=True,
        amplify_iters=kw.pop("amplify_iters", 2), **kw)
    return build_kernel(cfg, template_name=template)


# ---------------------------------------------------------------------------
# capability gating
# ---------------------------------------------------------------------------

def test_scan_rejects_non_checksum_only_modes():
    program = _program()
    program.launch.checksum_only = False
    problems = scan_program(program)
    assert any("checksum_only" in p for p in problems)
    with pytest.raises(CpuOracleUnsupported):
        translate_cpu_oracle(program)


def test_scan_rejects_float_types():
    from generator_ir.ir.typedecl import T
    from generator_ir.ir.statements import Declare
    program = _program()
    kernel = program.fetch_kernel()
    kernel.params.append(Declare(name="fp_arg", gtype=T.f32))
    problems = scan_program(program)
    assert any("integer-only" in p for p in problems)


def test_scan_rejects_cooperative_groups_raw_text():
    from generator_ir.ir.statements import RawCode
    program = _program()
    program.fetch_kernel().body.append(
        RawCode(text="namespace cg = cooperative_groups;"))
    problems = scan_program(program)
    assert any("cooperative_groups" in p for p in problems)


def test_scan_rejects_order_sensitive_atomics():
    from generator_ir.ir.nodes import Atomic
    from generator_ir.ir.statements import ExprStmt
    from generator_ir.ir.typedecl import T
    program = _program()
    kernel = program.fetch_kernel()
    kernel.body.append(ExprStmt(Atomic(T.u32, what="Exch")))
    problems = scan_program(program)
    assert any("order-sensitive" in p for p in problems)


# ---------------------------------------------------------------------------
# emission / compile / execute
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed,template", [
    (7, None), (12, "tree_reduce"), (11, "lane_exchange"),
])
def test_translate_compile_run_deterministic(seed, template, tmp_path):
    program = _program(seed=seed, template=template)
    source = translate_cpu_oracle(program)
    assert "checksum=" in source
    assert "int main(void)" in source
    first = run_cpu_oracle_source(source, workdir=tmp_path)
    second = run_cpu_oracle_source(source, workdir=tmp_path)
    assert first == second


def test_salt_changes_checksum(tmp_path):
    program = _program(seed=100)
    res = cpu_oracle_checksums(program, workdir=tmp_path)
    assert res["checksum"] != res["salted_checksum"]


def test_warp_shuffle_reduction_template_runs(tmp_path):
    """tree_reduce exercises __shfl_down chains + shared-memory reduction."""
    program = _program(seed=12, template="tree_reduce")
    assert run_cpu_oracle_source(
        translate_cpu_oracle(program), workdir=tmp_path) > 0


# ---------------------------------------------------------------------------
# strategy integration
# ---------------------------------------------------------------------------

def test_strategy_cpu_oracle_mode():
    case = default_registry().get("generator_ir").generate(
        11, 22, {"template": "tree_reduce", "complexity": "low",
                 "amplify_iters": 1, "cpu_oracle": True}, "t", "sm_86")
    assert case.oracle_mode == "cpu_emitted"
    assert len(case.expected_bytes) == 8
    assert case.oracle_source and "checksum=" in case.oracle_source
    assert case.oracle_note == "ok"


def test_strategy_defaults_to_cross_build_without_the_flag():
    case = default_registry().get("generator_ir").generate(
        11, 22, {"template": "tree_reduce", "complexity": "low",
                 "amplify_iters": 1}, "t", "sm_86")
    assert case.oracle_mode == "cross_build"
    assert case.expected_bytes == b""
    assert case.oracle_source is None


def test_strategy_unsupported_program_falls_back(monkeypatch):
    """An unmirrorable program must fall back to cross_build, never lie."""
    import generator_ir.emit.cpu_oracle as co
    monkeypatch.setattr(
        co, "translate_cpu_oracle",
        lambda program: (_ for _ in ()).throw(
            CpuOracleUnsupported("test: synthetic unsupported")))
    monkeypatch.setattr(co, "run_cpu_oracle_source", lambda *a, **k: 0)
    strategy = default_registry().get("generator_ir")
    case = strategy.generate(
        5, 6, {"template": "tree_reduce", "complexity": "low",
               "amplify_iters": 1, "cpu_oracle": True}, "t", "sm_86")
    assert case.oracle_mode == "cross_build"
    assert case.expected_bytes == b""
    assert case.oracle_source is None
    assert case.oracle_note.startswith("unsupported:")


def test_strategy_cpu_oracle_recompute_matches_generation():
    strategy = default_registry().get("generator_ir")
    params = {"template": "bin_count", "complexity": "low",
              "amplify_iters": 1, "cpu_oracle": True}
    case = strategy.generate(31, 41, params, "t", "sm_86")
    if case.oracle_mode != "cpu_emitted":        # pragma: no cover
        pytest.skip("seed produced an unsupported program")
    recomputed = strategy.cpu_oracle(
        input_words=0, seed=case.kernel_seed, params=case.params,
        input_bytes=b"", grid=case.grid, block=case.block,
        output_words_per_thread=0, target_sm=case.target_sm)
    assert recomputed == case.expected_bytes


def test_manifest_round_trip_writes_oracle_source(tmp_path):
    strategy = default_registry().get("generator_ir")
    case = strategy.generate(71, 72, {"template": "tree_reduce",
                                      "complexity": "low",
                                      "amplify_iters": 1,
                                      "cpu_oracle": True}, "t", "sm_86")
    if case.oracle_mode != "cpu_emitted":        # pragma: no cover
        pytest.skip("seed produced an unsupported program")
    write_testcase_dir(tmp_path / "case", case, "t", 7)
    assert (tmp_path / "case" / "oracle_cpu.c").exists()
    loaded = case_from_dir(tmp_path / "case")
    assert loaded.oracle_mode == "cpu_emitted"
    assert loaded.oracle_source == case.oracle_source
    assert loaded.expected_bytes == case.expected_bytes


# ---------------------------------------------------------------------------
# GPU end-to-end agreement (needs nvcc + a device)
# ---------------------------------------------------------------------------

def _gpu_available() -> bool:
    if not shutil.which("nvcc"):
        return False
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _gpu_available(), reason="no GPU / nvcc")
@pytest.mark.parametrize("seed,template", [
    (7, None), (12, "tree_reduce"), (11, "bin_count"),
])
def test_oracle_checksum_matches_gpu_harness(seed, template, tmp_path):
    from generator_ir.emit.nvidia import CudaEmitter
    program = _program(seed=seed, template=template)
    cpu = run_cpu_oracle_source(translate_cpu_oracle(program), workdir=tmp_path)
    cu = tmp_path / "case.cu"
    cu.write_text(CudaEmitter().rendersrc(program))
    exe = tmp_path / "case_gpu"
    r = subprocess.run(["nvcc", "-O2", "-arch=sm_86", "-o", str(exe), str(cu)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-500:]
    out = subprocess.run([str(exe)], capture_output=True, text=True,
                         timeout=300, env={**os.environ,
                                           "CUDA_STRESS_INPUT_SALT": ""}).stdout
    m = re.search(r"^checksum=([0-9]+)$", out, re.M)
    assert m, out[-300:]
    assert int(m.group(1)) == cpu
