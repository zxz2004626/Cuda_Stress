"""IR generator adapter and standalone-checksum execution tests."""

import struct

import pytest

from generator.base import GeneratedCase, default_registry
from pipeline.config import Toolchain, load_config
from pipeline.cli import _regenerate, _selected_strategies
from pipeline.executor import parse_output_blob, run_build


IR_PARAMS = {
    "template": "map_transform",
    "complexity": "low",
    "amplify_iters": 1,
    "codegen": False,
}


def test_generator_ir_emits_reproducible_checksum_only_program():
    strategy = default_registry().get("generator_ir")
    first = strategy.generate(101, 202, IR_PARAMS, "test", "sm_86")
    second = strategy.generate(101, 202, IR_PARAMS, "test", "sm_86")
    assert first.source == second.source
    assert first.params == second.params
    assert first.execution_mode == "standalone_checksum"
    assert first.oracle_mode == "cross_build"
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


def _standalone_case() -> GeneratedCase:
    return GeneratedCase(
        test_id="fake_ir", config_seed=1, test_seed=2,
        strategy="generator_ir", params={}, generator_version="test",
        kernel_seed=0, source="", input_bytes=b"", expected_bytes=b"",
        grid=[1], block=[1], input_words=0, output_words=1,
        output_words_per_thread=0, target_sm="sm_86",
        cpu_oracle="cross_build_consensus",
        comparison_policy="cross_build_checksum_u64",
        execution_mode="standalone_checksum", oracle_mode="cross_build",
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

    case = _standalone_case()
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
    case = _standalone_case()

    assert run_build(cfg, tc, case, build, 1).status == "OK"
    assert (build / "output_run_1.bin").exists()
    exe.write_text("#!/bin/sh\necho failed >&2\nexit 1\n")
    failed = run_build(cfg, tc, case, build, 1)
    assert failed.status == "CUDA_ERROR"
    assert not (build / "output_run_1.bin").exists()
