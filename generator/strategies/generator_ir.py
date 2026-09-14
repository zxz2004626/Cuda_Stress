"""Adapter for the IR-based CUDA generator.

Unlike the original strategies, generator_ir kernels have variable parameter
lists and may require initialization of auxiliary/global/constant memory.  The
adapter records that ABI in the manifest so the pipeline can compile one PTX
per toolkit, assemble profile-specific cubins, and reproduce the deterministic
checksum through the Driver API.
"""

from __future__ import annotations

import pickle
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..base import GeneratedCase, Strategy


class GeneratorIR(Strategy):
    """Generate a deterministic, checksum-observable CUDA program from IR."""

    name = "generator_ir"
    target_opcodes: List[str] = []

    _COMPLEXITIES = {"low", "medium", "high", "stress"}
    _INPUT_DISTRIBUTIONS = {
        "hash", "zeros", "small", "random", "adversarial",
    }
    _TYPE_NAMES = {
        "bool", "int", "uint", "unsignedint", "char", "uchar",
        "unsignedchar", "signedchar", "schar", "short", "ushort",
        "unsignedshort", "longlong", "unsignedlonglong", "ulonglong",
        "float", "double", "half", "__half", "bfloat16", "bf16",
    }
    _KNOWN_PARAMS = {
        "ir_seed", "template", "template_catalog", "complexity", "observable_fold",
        "checksum_only", "amplify_iters", "codegen", "codegen_repeat",
        "generic_baseline", "pass_targeted_only",
        "allow_volatile", "input_distribution", "randomize_launch",
        "prob_float_pool", "prob_raw_div", "prob_wide_consts",
        "scalar_param_count", "use_full_int_widths", "enable_smem",
        "enable_atomics", "enable_warp", "enable_barriers", "block",
        "grid", "type_allowlist", "cpu_oracle",
    }

    @staticmethod
    def _dim3(value: Any, name: str) -> Optional[Tuple[int, int, int]]:
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 3:
            raise ValueError(f"{name} must contain one to three positive integers")
        dims = [int(v) for v in value]
        if any(v <= 0 for v in dims):
            raise ValueError(f"{name} dimensions must be positive")
        out = tuple((dims + [1, 1, 1])[:3])
        limits = (1024, 1024, 64) if name == "block" else (2**31 - 1, 65535, 65535)
        if any(dim > limit for dim, limit in zip(out, limits)):
            raise ValueError(f"{name} dimensions exceed CUDA limits {limits}")
        if name == "block" and out[0] * out[1] * out[2] > 1024:
            raise ValueError("block contains more than 1024 threads")
        return out

    def _normalized_params(self, config_seed: int, test_seed: int,
                           params: Dict[str, Any], templates: Iterable[str]) -> Dict[str, Any]:
        raw = dict(params or {})
        unknown = sorted(set(raw) - self._KNOWN_PARAMS)
        if unknown:
            raise ValueError(f"unknown generator_ir parameters: {unknown}")
        if "ir_seed" in raw:
            ir_seed = int(raw["ir_seed"])
        else:
            ir_seed = self._new_rng(config_seed, test_seed).getrandbits(63)

        complexity = str(raw.get("complexity", "medium"))
        if complexity not in self._COMPLEXITIES:
            raise ValueError(f"unsupported generator_ir complexity {complexity!r}")

        generic_baseline = bool(raw.get("generic_baseline", False))
        template_catalog = str(raw.get("template_catalog", "default") or "default")
        catalog_key = template_catalog.strip().lower().replace("-", "_")
        if catalog_key not in {"default", "paper", "passspec", "raw_llm", "rawllm", "arm_b", "armb"}:
            raise ValueError(f"unsupported generator_ir template_catalog {template_catalog!r}")
        if generic_baseline and catalog_key not in {"default", "paper", "passspec"}:
            raise ValueError("generic_baseline cannot be combined with a custom template_catalog")
        names = sorted(str(n) for n in templates)
        requested = raw.get("template")
        if generic_baseline:
            if requested not in (None, "", "random", "generic_baseline"):
                raise ValueError(
                    "generic_baseline cannot be combined with a concrete template")
            template = "generic_baseline"
        elif requested in (None, "", "random"):
            # Weighted pick so PREFERENCE_TABLE (0 = never) is respected; the
            # legacy uniform `choice` ignored the table and kept re-selecting
            # templates that hang the GPU.
            from generator_ir.templates.catalog import PREFERENCE_TABLE
            rng = random.Random(ir_seed ^ 0x49525F54454D504C)
            weights = [max(0.0, float(PREFERENCE_TABLE.get(n, 1.0))) for n in names]
            template = rng.choices(names, weights=weights, k=1)[0]
        else:
            template = str(requested)
            if template not in names:
                raise ValueError(
                    f"unknown generator_ir template {template!r}; available: {names}")

        input_distribution = str(raw.get("input_distribution", "hash"))
        if input_distribution not in self._INPUT_DISTRIBUTIONS:
            raise ValueError(f"unsupported input_distribution {input_distribution!r}")

        block = self._dim3(raw.get("block"), "block")
        grid = self._dim3(raw.get("grid"), "grid")
        type_allowlist = raw.get("type_allowlist")
        if type_allowlist is not None:
            if not isinstance(type_allowlist, (list, tuple)):
                raise ValueError("type_allowlist must be a list of generator_ir type names")
            type_allowlist = [str(v) for v in type_allowlist]
            unknown_types = [v for v in type_allowlist
                             if v.lower().replace(" ", "").replace("_", "")
                             not in {n.replace("_", "") for n in self._TYPE_NAMES}]
            if unknown_types:
                raise ValueError(f"unknown generator_ir type names: {unknown_types}")

        codegen = bool(raw.get("codegen", False))
        if generic_baseline and codegen:
            raise ValueError("generic_baseline cannot be combined with codegen/weave mode")
        randomize_launch = bool(raw.get("randomize_launch", False))
        amplify_iters = max(1, int(raw.get("amplify_iters", 2)))
        codegen_repeat = max(1, int(raw.get("codegen_repeat", amplify_iters)))
        if codegen and "amplify_iters" in raw and "codegen_repeat" in raw \
                and amplify_iters != codegen_repeat:
            raise ValueError(
                "codegen mode received conflicting amplify_iters and codegen_repeat")
        if codegen:
            # The weave path calls this knob codegen_repeat.  Canonicalize the
            # compatibility alias so every stored parameter affects source.
            amplify_iters = codegen_repeat
        elif "codegen_repeat" in raw and codegen_repeat != amplify_iters:
            raise ValueError(
                "codegen_repeat is only valid when codegen=true; canonical manifests "
                "must keep it equal to amplify_iters")
        if codegen and (block is not None or grid is not None):
            raise ValueError("codegen mode chooses its own geometry; block/grid overrides are unsupported")
        if not codegen and randomize_launch and (block is not None or grid is not None):
            raise ValueError("randomize_launch cannot be combined with explicit block/grid")
        if not codegen and (block is not None or grid is not None):
            raise ValueError(
                "explicit block/grid overrides are disabled because several templates require fixed geometry")

        default_full_widths = complexity in {"high", "stress"}
        normalized: Dict[str, Any] = {
            "ir_seed": ir_seed,
            "template": template,
            "template_catalog": "raw_llm" if catalog_key in {"raw_llm", "rawllm", "arm_b", "armb"} else "default",
            "complexity": complexity,
            # observable_fold makes the emitted checksum depend on live IR
            # values and deliberately excludes address/FP-only false deltas.
            "observable_fold": True,
            "checksum_only": True,
            "amplify_iters": amplify_iters,
            "codegen": codegen,
            "codegen_repeat": codegen_repeat,
            "generic_baseline": generic_baseline,
            "pass_targeted_only": bool(raw.get("pass_targeted_only", False)),
            "allow_volatile": bool(raw.get("allow_volatile", False)),
            "input_distribution": input_distribution,
            "randomize_launch": randomize_launch,
            "prob_float_pool": float(raw.get("prob_float_pool", 0.0)),
            "prob_raw_div": float(raw.get("prob_raw_div", 0.0)),
            "prob_wide_consts": (0.0 if generic_baseline else
                                  float(raw.get("prob_wide_consts", 0.0))),
            "scalar_param_count": max(0, int(raw.get("scalar_param_count", 3))),
            "use_full_int_widths": bool(
                raw.get("use_full_int_widths", default_full_widths)),
            "enable_smem": bool(raw.get("enable_smem", True)),
            "enable_atomics": bool(raw.get("enable_atomics", True)),
            "enable_warp": bool(raw.get("enable_warp", True)),
            "enable_barriers": bool(raw.get("enable_barriers", True)),
            # Opt-in: emit + execute a CPU C oracle alongside the GPU program.
            # Off by default so archived cross_build manifests still reproduce
            # byte-for-byte; enable per campaign via strategy params.
            "cpu_oracle": bool(raw.get("cpu_oracle", False)),
        }
        if block is not None:
            normalized["block"] = list(block)
        if grid is not None:
            normalized["grid"] = list(grid)
        if type_allowlist is not None:
            normalized["type_allowlist"] = type_allowlist
        return normalized

    @staticmethod
    def _build_program(p: Dict[str, Any], target_sm: str):
        """Deterministically rebuild the IR Program from normalized params.

        Shared by generate() and cpu_oracle() so a recompute from the
        manifest reproduces the exact program the case was built from.
        """
        from generator_ir.gen.assembler import build_kernel
        from generator_ir.gen.options import GenConfig

        generic_requested = bool(p.get("generic_baseline", False))
        requested_catalog = str(p.get("template_catalog", "default") or "default")
        if generic_requested:
            build_registry = None
        else:
            from generator_ir.templates.catalog import kx_build_named_registry
            catalog_key = requested_catalog.strip().lower().replace("-", "_")
            build_registry = kx_build_named_registry(requested_catalog) \
                if catalog_key in {"raw_llm", "rawllm", "arm_b", "armb"} else None

        cfg = GenConfig(
            seed=p["ir_seed"],
            complexity=p["complexity"],
            target="cuda",
            isa=target_sm,
            observable_fold=True,
            checksum_only=True,
            amplify_iters=p["amplify_iters"],
            codegen=p["codegen"],
            codegen_repeat=p["codegen_repeat"],
            generic_baseline=p["generic_baseline"],
            pass_targeted_only=p["pass_targeted_only"],
            allow_volatile=p["allow_volatile"],
            input_distribution=p["input_distribution"],
            randomize_launch=p["randomize_launch"],
            prob_float_pool=p["prob_float_pool"],
            prob_raw_div=p["prob_raw_div"],
            prob_wide_consts=p["prob_wide_consts"],
            scalar_param_count=p["scalar_param_count"],
            use_full_int_widths=p["use_full_int_widths"],
            enable_smem=p["enable_smem"],
            enable_atomics=p["enable_atomics"],
            enable_warp=p["enable_warp"],
            enable_barriers=p["enable_barriers"],
            type_allowlist=p.get("type_allowlist"),
        )
        # Complexity presets intentionally adjust several knobs.  Explicit
        # adapter values are the effective manifest contract, so apply them
        # after __post_init__ as well.
        cfg.amplify_iters = p["amplify_iters"]
        cfg.scalar_param_count = p["scalar_param_count"]
        cfg.use_full_int_widths = p["use_full_int_widths"]
        if "block" in p:
            cfg.threads_per_block_dim = tuple(p["block"])
        if "grid" in p:
            cfg.blocks_per_grid_dim = tuple(p["grid"])
        return build_kernel(
            cfg,
            template_name=None if p["generic_baseline"] else p["template"],
            registry=build_registry,
        )

    def generate(self, config_seed: int, test_seed: int, params: Dict[str, Any],
                 gen_version: str, target_sm: str) -> GeneratedCase:
        if not str(target_sm).startswith("sm_"):
            raise ValueError(f"generator_ir CUDA target must be an sm_* architecture, got {target_sm!r}")

        from generator_ir import __version__ as ir_version
        from generator_ir.emit.nvidia import CudaEmitter
        from generator_ir.emit.host_harness import cuda_driver_spec
        # The generic baseline must not instantiate the template catalog.  A
        # concrete template request is rejected by normalization below.
        generic_requested = bool((params or {}).get("generic_baseline", False))
        requested_catalog = str((params or {}).get("template_catalog", "default") or "default")
        if generic_requested:
            template_names: Iterable[str] = ()
        else:
            from generator_ir.templates.catalog import kx_build_named_registry
            registry = kx_build_named_registry(requested_catalog)
            template_names = registry.kx_all_names()
        p = self._normalized_params(
            config_seed, test_seed, params, template_names)
        if p["template_catalog"] == "raw_llm" and p["codegen"]:
            raise ValueError("template_catalog=raw_llm is not compatible with codegen/weave mode")

        program = self._build_program(p, target_sm)
        source = CudaEmitter().rendersrc(program)
        ir_blob = pickle.dumps(program, protocol=pickle.HIGHEST_PROTOCOL)
        kernel = program.fetch_kernel()
        if kernel is None:
            raise ValueError("generator_ir produced a program without a CUDA kernel")
        if "checksum=" not in source or "int main(" not in source:
            raise ValueError("generator_ir checksum harness was not emitted")

        # ---- optional CPU C oracle (IR -> gcc-runnable C) -------------------
        # Emits a standalone C program that replays the same deterministic
        # checksum contract and records its u64 checksum as expected.bin.
        # Constructs the runtime cannot mirror exactly fall back to the
        # cross-build oracle — never to a possibly-wrong oracle.
        oracle_source = None
        oracle_checksum: Optional[int] = None
        oracle_note = "disabled"
        if p["cpu_oracle"]:
            from generator_ir.emit.cpu_oracle import (
                translate_cpu_oracle, run_cpu_oracle_source, CpuOracleUnsupported)
            try:
                oracle_source = translate_cpu_oracle(program)
                oracle_checksum = run_cpu_oracle_source(
                    oracle_source, timeout=self.cpu_oracle_timeout_s)
                oracle_note = "ok"
            except CpuOracleUnsupported as exc:
                oracle_source, oracle_checksum = None, None
                oracle_note = f"unsupported: {exc.reason}"
            except Exception as exc:  # noqa: BLE001 - never claim a broken oracle
                oracle_source, oracle_checksum = None, None
                oracle_note = f"failed: {exc}"

        launch = program.launch
        execution_spec = cuda_driver_spec(program, kernel)
        suffix = f"+generator_ir-{ir_version}"
        effective_version = gen_version if "+generator_ir-" in gen_version else f"{gen_version}{suffix}"
        if oracle_checksum is not None:
            expected_bytes = oracle_checksum.to_bytes(8, "little")
            oracle_mode = "cpu_emitted"
            comparison_policy = "cpu_oracle_checksum_u64"
        else:
            # The Driver checksum harness derives its buffers deterministically
            # from execution_spec. Empty artifacts preserve the common testcase
            # directory shape without pretending a CPU oracle exists.
            expected_bytes = b""
            oracle_mode = "cross_build"
            comparison_policy = "cross_build_checksum_u64"
        return GeneratedCase(
            test_id=f"generator_ir__{config_seed:x}__{test_seed:x}",
            config_seed=config_seed,
            test_seed=test_seed,
            strategy=self.name,
            params=p,
            generator_version=effective_version,
            kernel_seed=p["ir_seed"],
            source=source,
            input_bytes=b"",
            expected_bytes=expected_bytes,
            grid=list(launch.grid),
            block=list(launch.block),
            input_words=0,
            output_words=1,
            output_words_per_thread=0,
            target_sm=target_sm,
            cpu_oracle="cpu_emitted_c" if oracle_checksum is not None
            else "cross_build_consensus",
            comparison_policy=comparison_policy,
            cuda_features=[target_sm, "generator_ir", p["template"], "driver_checksum"],
            kernel_entry=kernel.name,
            sanitizer_policy="driver_checksum_harness",
            execution_mode="driver_checksum",
            oracle_mode=oracle_mode,
            execution_spec=execution_spec,
            ir_blob=ir_blob,
            oracle_source=oracle_source,
            oracle_note=oracle_note,
        )

    # gcc compile+run budget for one oracle evaluation during generation.
    cpu_oracle_timeout_s = 300.0

    def cpu_oracle(self, input_words: int, seed: int, params: Dict[str, Any],
                   input_bytes: bytes, grid: List[int], block: List[int],
                   output_words_per_thread: int,
                   target_sm: Optional[str] = None,
                   input_salt: Optional[int] = None) -> bytes:
        """Recompute the CPU-oracle checksum from manifest params.

        Rebuilds the same IR Program the case was generated from (pure
        function of the normalized params + target SM), re-emits and
        re-executes the oracle.  Returns the u64 checksum as 8 LE bytes.
        """
        if not params.get("cpu_oracle"):
            raise NotImplementedError(
                "this generator_ir case has no CPU oracle (cross_build)")
        if "ir_seed" not in params:
            raise ValueError("cpu_oracle() expects already-normalized manifest params")
        sm = str(target_sm or "sm_75")
        program = self._build_program(dict(params), sm)
        from generator_ir.emit.cpu_oracle import (
            run_cpu_oracle_source, translate_cpu_oracle)
        source = translate_cpu_oracle(program)
        checksum = run_cpu_oracle_source(
            source, timeout=self.cpu_oracle_timeout_s, input_salt=input_salt)
        return checksum.to_bytes(8, "little")
