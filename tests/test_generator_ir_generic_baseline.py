"""Isolation tests for the generator_ir generic baseline mode."""

from __future__ import annotations

import argparse
import pickle
import random

import pytest

from generator.base import default_registry
from generator_ir.cli_tools import cmd_build
from generator_ir.emit.nvidia import CudaEmitter
from generator_ir.gen.options import GenConfig
from generator_ir.ir.typedecl import T


FORBIDDEN_SOURCE_TOKENS = (
    "asm volatile",
    "__launch_bounds__",
    "__cluster_dims__",
    "GPU_FAKE_DIVERGE",
    "GPU_GROUP_DIVERGE",
    "cooperative_groups",
    "cuda::",
    "__pipeline_",
    "__syncwarp",
    "__match_any_sync",
    "__match_all_sync",
    "__syncthreads_count",
    "__syncthreads_and",
    "__syncthreads_or",
    "__threadfence",
    "#pragma unroll",
    "__ldg(",
    "__ldca",
    "__ldcg",
    "__ldcs",
    "__ldlu",
    "__byte_perm",
    "__umul64hi",
    "__mul64hi",
)

ALLOWED_GENERIC_FEATURES = {
    "generic_baseline",
    "generic_shared",
    "sync",
    "atomics",
}


def _generic_config(seed: int = 123, complexity: str = "medium", **overrides):
    values = {
        "seed": seed,
        "complexity": complexity,
        "target": "cuda",
        "isa": "sm_86",
        "observable_fold": True,
        "checksum_only": True,
        "amplify_iters": 1,
        "generic_baseline": True,
    }
    values.update(overrides)
    return GenConfig(**values)


def _render_generic(seed: int = 123, complexity: str = "medium") -> tuple[str, object]:
    from generator_ir.gen.assembler import build_kernel

    program = build_kernel(_generic_config(seed=seed, complexity=complexity))
    return CudaEmitter().rendersrc(program), program


def _assert_no_targeted_source(source: str) -> None:
    for token in FORBIDDEN_SOURCE_TOKENS:
        assert token not in source


def _assert_generic_features(program) -> None:
    features = program.launch.generation_features
    assert features <= ALLOWED_GENERIC_FEATURES
    assert "generic_baseline" in features
    assert not any(
        f.startswith("template_fragment:")
        or f.startswith("forced_template_fragment:")
        or f.startswith("weave_")
        for f in features
    )


def test_generic_baseline_canonicalizes_targeted_knobs():
    cfg = _generic_config(
        complexity="stress",
        do_bank_stress=True,
        reg_pressure_level="extreme",
        do_vector_stress=True,
        do_switch_stress=True,
        do_barrier_stress=True,
        prob_u64_divmod=1.0,
        prob_divergent_switch=1.0,
        prob_strided_unroll=1.0,
        prob_runtime_loop_bound=1.0,
        prob_wide_consts=1.0,
    )

    assert cfg.stmt_budget == 300
    assert not cfg.do_bank_stress
    assert cfg.reg_pressure_level == "none"
    assert not cfg.do_vector_stress
    assert not cfg.do_switch_stress
    assert not cfg.do_barrier_stress
    assert cfg.prob_u64_divmod == 0.0
    assert cfg.prob_divergent_switch == 0.0
    assert cfg.prob_strided_unroll == 0.0
    assert cfg.prob_runtime_loop_bound == 0.0
    assert cfg.prob_wide_consts == 0.0

    with pytest.raises(ValueError, match="generic_baseline"):
        _generic_config(codegen=True)


def test_generic_baseline_skips_template_registry(monkeypatch):
    import generator_ir.gen.assembler as assembler

    def forbidden_registry():
        raise AssertionError("generic baseline must not build the template registry")

    monkeypatch.setattr(assembler, "kx_build_default_registry", forbidden_registry)
    program = assembler.KernelGenerator(_generic_config()).kx_generate()
    assert program.launch.generic_baseline
    assert program.launch.shared_bytes == 0
    _assert_generic_features(program)

    with pytest.raises(ValueError, match="template"):
        assembler.KernelGenerator(_generic_config()).kx_generate("map_transform")


@pytest.mark.parametrize(
    ("seed", "complexity"),
    [(11, "low"), (12, "medium"), (13, "high"), (14, "stress")],
)
def test_generic_baseline_source_excludes_targeted_constructs(seed, complexity):
    source, program = _render_generic(seed=seed, complexity=complexity)
    _assert_no_targeted_source(source)
    _assert_generic_features(program)
    assert "extern \"C\" __global__ void kernel_main" in source
    assert "<<<grid, block" in source
    assert "checksum=%llu" in source
    assert program.launch.input_size_mult == 1
    assert program.fetch_kernel().launch_bounds is None
    assert program.fetch_kernel().cluster_dims is None


def test_generic_expression_and_statement_tables_are_allow_lists():
    from generator_ir.gen.context import GenContext
    from generator_ir.gen.expr_builder import ExprGen
    from generator_ir.gen.stmt_builder import StmtGen
    from generator_ir.ir.nodes import Var

    ctx = GenContext(_generic_config())
    for name in ("output", "input", "n", "tid"):
        ctx.kx_declare(Var(T.u32, name=name), divergent=(name == "tid"))
    expr_names = {name for name, _ in ExprGen(ctx)._generic_kind_table(T.u32)}
    stmt_names = {name for name, _ in StmtGen(ctx, ExprGen(ctx))._kind_table(0)}

    assert {
        "catalog_intrinsic",
        "cuda_cache_load",
        "cuda_dpx",
        "fake_diverge",
        "group_diverge",
        "ldg_load",
        "mul_wide",
        "u64_divmod",
    }.isdisjoint(expr_names)
    assert {
        "async_copy",
        "coop_groups",
        "cuda_barrier",
        "cuda_cache_store",
        "cuda_match",
        "cuda_pipeline",
        "cuda_syncwarp",
        "divergent_switch",
        "dynamic_shared",
        "emi_guard",
        "fence",
        "gated_barrier",
        "pipeline",
        "pragma",
        "scoped_atomic",
        "strided_unroll",
        "sync_predicate",
        "template",
        "warp_aware",
    }.isdisjoint(stmt_names)


def test_pipeline_adapter_generic_baseline_is_canonical_and_reproducible():
    strategy = default_registry().get("generator_ir")
    params = {
        "generic_baseline": True,
        "template": "random",
        "complexity": "high",
        "amplify_iters": 1,
        "prob_wide_consts": 1.0,
    }
    first = strategy.generate(101, 202, params, "test", "sm_86")
    second = strategy.generate(101, 202, params, "test", "sm_86")

    assert first.source == second.source
    assert first.params == second.params
    assert first.params["template"] == "generic_baseline"
    assert first.params["generic_baseline"] is True
    assert first.params["prob_wide_consts"] == 0.0
    assert first.cuda_features == [
        "sm_86", "generator_ir", "generic_baseline", "driver_checksum"]
    _assert_no_targeted_source(first.source)

    program = pickle.loads(first.ir_blob)
    assert program.launch.generic_baseline
    _assert_generic_features(program)


@pytest.mark.parametrize("params", [
    {"generic_baseline": True, "template": "map_transform"},
    {"generic_baseline": True, "codegen": True},
])
def test_pipeline_adapter_rejects_generic_baseline_combinations(params):
    with pytest.raises(ValueError, match="generic_baseline"):
        default_registry().get("generator_ir").generate(
            1, 2, params, "test", "sm_86")


def test_scheduler_does_not_mutate_generic_baseline_params():
    from pipeline.scheduler import LoopScheduler

    params = {
        "generic_baseline": True,
        "complexity": "high",
        "amplify_iters": 1,
        "prob_wide_consts": 0.0,
    }
    mutated = LoopScheduler._mutate_params(
        object(), "generator_ir", params, random.Random(0))
    assert mutated == params
    assert mutated is not params


def test_standalone_cli_generic_baseline_generation_and_validation(tmp_path):
    args = argparse.Namespace(
        seed=77,
        count=2,
        template=None,
        backend="cuda",
        output_dir=str(tmp_path),
        complexity="low",
        arch="sm_86",
        ci=False,
        clock64=False,
        loop=False,
        observable_fold=True,
        allow_volatile=False,
        codegen=False,
        generic_baseline=True,
        enable_feature_gens=False,
        vary_launch=False,
    )

    assert cmd_build(args) == 0
    emitted = sorted(tmp_path.glob("*.cu"))
    assert len(emitted) == 2
    for path in emitted:
        _assert_no_targeted_source(path.read_text())

    bad = argparse.Namespace(**vars(args))
    bad.template = "map_transform"
    with pytest.raises(ValueError, match="generic-baseline"):
        cmd_build(bad)
