"""IR generator adapter and Driver-checksum execution tests."""

import json
import re
import struct
from concurrent.futures import ThreadPoolExecutor

import pytest

from generator.base import GeneratedCase, default_registry
from pipeline.config import Toolchain, load_config
from pipeline.cli import _regenerate, _selected_strategies
from pipeline.executor import parse_output_blob, run_build
from pipeline import compiler, scheduler


IR_PARAMS = {
    "template": "map_transform",
    "complexity": "low",
    "amplify_iters": 1,
    "codegen": False,
}


@pytest.mark.parametrize("template", ["prefix_scan", "tile_swap", "tiled_gemm"])
def test_structured_observable_templates_have_one_output_writer(template):
    """Natural result layouts must not race the final output[tid] checksum."""
    case = default_registry().get("generator_ir").generate(
        101, 202, {
            "template": template,
            "complexity": "low",
            "amplify_iters": 2,
            "codegen": False,
        }, "test", "sm_86")
    kernel_source = case.source.split("int main(void)", 1)[0]
    writes = re.findall(r"\boutput\s*\[[^;\n]+\]\s*(?:\^?=)", kernel_source)
    assert writes == ["output[tid] ="]
    assert "_fv =" in kernel_source
    assert "__syncthreads()" in kernel_source


def test_generator_ir_emits_reproducible_checksum_only_program():
    strategy = default_registry().get("generator_ir")
    first = strategy.generate(101, 202, IR_PARAMS, "test", "sm_86")
    second = strategy.generate(101, 202, IR_PARAMS, "test", "sm_86")
    assert first.source == second.source
    assert first.params == second.params
    assert first.execution_mode == "driver_checksum"
    assert first.oracle_mode == "cross_build"
    assert first.execution_spec["kind"] == "generator_ir_checksum"
    assert first.execution_spec["kernel"] == first.kernel_entry == "kernel_main"
    assert [a["kind"] for a in first.execution_spec["args"][:3]] == [
        "output", "input", "n"]
    assert first.input_bytes == first.expected_bytes == b""
    assert first.source.count("<<<grid, block") == 1
    assert "checksum=%llu" in first.source
    assert "avg_time_ms" not in first.source
    assert "cudaEventCreate" not in first.source
    reproduced = _regenerate(first)
    assert reproduced.case_id() == first.case_id()
    assert reproduced.source == first.source


def test_generator_backend_selection_preserves_builtin_default():
    cfg = load_config()
    registry = default_registry()
    builtin = _selected_strategies(cfg, registry, [], "builtin")
    assert "generator_ir" not in builtin
    assert _selected_strategies(cfg, registry, [], "generator_ir") == ["generator_ir"]
    mixed = _selected_strategies(cfg, registry, [], "mixed")
    assert "generator_ir" in mixed
    assert set(builtin).issubset(mixed)


def test_generator_ir_codegen_repeat_is_effective_and_canonical():
    strategy = default_registry().get("generator_ir")
    one = strategy.generate(3, 4, {
        "template": "map_transform", "complexity": "low",
        "codegen": True, "amplify_iters": 1,
    }, "test", "sm_86")
    two = strategy.generate(3, 4, {
        "template": "map_transform", "complexity": "low",
        "codegen": True, "amplify_iters": 2,
    }, "test", "sm_86")
    assert one.params["amplify_iters"] == one.params["codegen_repeat"] == 1
    assert two.params["amplify_iters"] == two.params["codegen_repeat"] == 2
    assert one.source != two.source


@pytest.mark.parametrize("params", [
    {"template": "map_transform", "mispelled_knob": True},
    {"template": "map_transform", "block": [2048]},
    {"template": "tree_reduce", "block": [64]},
    {"template": "map_transform", "block": [64], "randomize_launch": True},
    {"template": "map_transform", "codegen": True, "grid": [4]},
    {"template": "map_transform", "type_allowlist": ["definitely_not_a_type"]},
    {"template": "map_transform", "type_allowlist": ["uint4"]},
    {"template": "map_transform", "amplify_iters": 1, "codegen_repeat": 3},
])
def test_generator_ir_rejects_ambiguous_or_unsafe_params(params):
    with pytest.raises(ValueError):
        default_registry().get("generator_ir").generate(
            1, 2, params, "test", "sm_86")


def _checksum_case(mode="standalone_checksum") -> GeneratedCase:
    return GeneratedCase(
        test_id="fake_ir", config_seed=1, test_seed=2,
        strategy="generator_ir", params={}, generator_version="test",
        kernel_seed=0, source="", input_bytes=b"", expected_bytes=b"",
        grid=[1], block=[1], input_words=0, output_words=1,
        output_words_per_thread=0, target_sm="sm_86",
        cpu_oracle="cross_build_consensus",
        comparison_policy="cross_build_checksum_u64",
        execution_mode=mode, oracle_mode="cross_build",
    )


def test_standalone_executor_normalizes_checksum_and_input_salt(tmp_path):
    cfg = load_config()
    tc = Toolchain(
        id="fake", root="/tmp/fake-cuda", nvcc="/bin/false",
        ptxas="/bin/false", nvdisasm="/bin/false", cuobjdump="/bin/false",
        arch="sm_86", ccbin="/bin/false", runnable=True)
    build = tmp_path / "build"
    build.mkdir()
    exe = build / "kernel.exe"
    exe.write_text(
        "#!/bin/sh\n"
        "salt=${CUDA_STRESS_INPUT_SALT:-0}\n"
        "printf 'checksum=%s\\n' $((100 + salt))\n")
    exe.chmod(0o755)

    case = _checksum_case()
    base = run_build(cfg, tc, case, build, 1, input_salt=0)
    salted = run_build(cfg, tc, case, build, "input_sensitivity", input_salt=7)
    assert base.status == salted.status == "OK"
    assert base.output_words == struct.pack("<Q", 100)
    assert salted.output_words == struct.pack("<Q", 107)
    parsed = parse_output_blob((build / "output_run_1.bin").read_bytes())
    assert parsed.status == "OK"
    assert parsed.output_words == struct.pack("<Q", 100)


def test_standalone_failed_rerun_removes_previous_output(tmp_path):
    cfg = load_config()
    tc = Toolchain(
        id="fake", root="/tmp/fake-cuda", nvcc="/bin/false",
        ptxas="/bin/false", nvdisasm="/bin/false", cuobjdump="/bin/false",
        arch="sm_86", ccbin="/bin/false", runnable=True)
    build = tmp_path / "build"
    build.mkdir()
    exe = build / "kernel.exe"
    exe.write_text("#!/bin/sh\nprintf 'checksum=9\\n'\n")
    exe.chmod(0o755)
    case = _checksum_case()

    assert run_build(cfg, tc, case, build, 1).status == "OK"
    assert (build / "output_run_1.bin").exists()
    exe.write_text("#!/bin/sh\necho failed >&2\nexit 1\n")
    failed = run_build(cfg, tc, case, build, 1)
    assert failed.status == "CUDA_ERROR"
    assert not (build / "output_run_1.bin").exists()


def test_driver_checksum_executor_writes_dynamic_abi(monkeypatch, tmp_path):
    cfg = load_config()
    cfg.execution["runner_restart_after"] = 0
    tc = Toolchain(
        id="fake", root="/tmp/fake-cuda", nvcc="/bin/false",
        ptxas="/bin/false", nvdisasm="/bin/false", cuobjdump="/bin/false",
        arch="sm_86", ccbin="/bin/false", runnable=True)
    case = _checksum_case("driver_checksum")
    case.kernel_entry = "kernel_main"
    case.execution_spec = {
        "version": 1, "kind": "generator_ir_checksum", "kernel": "kernel_main",
        "grid": [2, 1, 1], "block": [64, 1, 1], "shared_bytes": 128,
        "n": 1024, "buf_size": 1024, "in_size": 2048,
        "input_distribution": "hash",
        "args": [
            {"kind": "output", "name": "output"},
            {"kind": "input", "name": "input"},
            {"kind": "n", "name": "n"},
            {"kind": "scalar", "name": "var_0", "data_hex": "2a000000"},
        ],
        "globals": [{"name": "_cmem", "init": "cmem_u32"}],
    }
    build = tmp_path / "build"
    build.mkdir()
    (build / "kernel.cubin").write_bytes(b"cubin")
    fake_runner = tmp_path / "runner"
    fake_runner.write_text("runner")

    def fake_run(argv, **kwargs):
        spec_path = argv[argv.index("--ir-spec") + 1]
        lines = dict(line.split("=", 1) for line in open(spec_path) if "=" in line)
        assert lines["kernel"].strip() == "kernel_main"
        assert lines["grid"].strip() == "2,1,1"
        assert lines["input_salt"].strip() == "7"
        text = open(spec_path).read()
        assert "arg=scalar:2a000000" in text
        assert "global=cmem_u32:_cmem" in text
        output = build / "output_run_1.bin"
        from pipeline.executor import _write_ok_blob
        _write_ok_blob(output, case.case_id(), struct.pack("<Q", 123))
        (build / "execution_run_1.json").write_text(json.dumps({
            "status": "OK", "elapsed_us": 10, "context_reused": False}))
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr("pipeline.executor.runner_path", lambda _cfg: fake_runner)
    monkeypatch.setattr("pipeline.executor.subprocess.run", fake_run)
    result = run_build(cfg, tc, case, build, 1, input_salt=7)
    assert result.status == "OK"
    assert result.output_words == struct.pack("<Q", 123)
    assert result.elapsed_us == 10


def test_driver_checksum_build_reuses_shared_ptx(monkeypatch, tmp_path):
    cfg = load_config()
    tc = Toolchain(
        id="fake", root="/tmp/fake-cuda", nvcc="/bin/false",
        ptxas="/bin/false", nvdisasm="/bin/false", cuobjdump="/bin/false",
        arch="sm_86", ccbin="/bin/false", runnable=True)
    profile = cfg.profiles[0]
    test_dir = tmp_path / "case"
    build = test_dir / "builds" / "fake__O0"
    build.mkdir(parents=True)
    (test_dir / "test.cu").write_text("source")
    (test_dir / "manifest.json").write_text(json.dumps({
        "execution_mode": "driver_checksum"}))
    (build / "kernel.ptx").write_text("shared ptx")

    def forbidden_frontend(*args, **kwargs):
        raise AssertionError("per-profile frontend compile must not run")

    def fake_ptxas(_tc, _profile, _ptx, out, *_args, **_kwargs):
        out.write_bytes(b"cubin")
        return compiler.CmdRecord([], 0, stderr="Used 7 registers", ok=True)

    def fake_cuobjdump(_tc, _cubin, out, *_args, **_kwargs):
        out.write_text("sass")
        return compiler.CmdRecord([], 0, stdout="sass", ok=True)

    monkeypatch.setattr(compiler, "nvcc_ptx", forbidden_frontend)
    monkeypatch.setattr(compiler, "ptxas_cubin", fake_ptxas)
    monkeypatch.setattr(compiler, "cuobjdump_sass", fake_cuobjdump)
    result = compiler.compile_build(cfg, tc, profile, test_dir, build)
    assert result["ok"]
    assert result["commands"][0] == {
        "stage": "ptx", "ok": True, "shared_artifact": True,
        "path": "kernel.ptx"}


def test_worker_ptx_serializes_frontend_once(monkeypatch, tmp_path):
    cfg = load_config()
    tc = Toolchain(
        id="fake", root="/tmp/fake-cuda", nvcc="/bin/false",
        ptxas="/bin/false", nvdisasm="/bin/false", cuobjdump="/bin/false",
        arch="sm_86", ccbin="/bin/false", runnable=True)
    test_dir = tmp_path / "case"
    test_dir.mkdir()
    src = test_dir / "test.cu"
    src.write_text("source")
    (test_dir / "manifest.json").write_text(json.dumps({
        "execution_mode": "driver_checksum"}))
    ptx_dir = test_dir / "_ptx" / tc.id
    calls = []

    def fake_frontend(_tc, _src, out, timeout=None):
        calls.append(out)
        out.write_text("ptx")
        return compiler.CmdRecord([], 0, ok=True)

    monkeypatch.setattr(compiler, "nvcc_standalone_ptx", fake_frontend)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(
            lambda _: scheduler.worker_ptx(cfg, tc, ptx_dir, src), range(4)))
    assert all(result["ok"] for result in results)
    assert len(calls) == 1
