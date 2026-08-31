"""Generator strategy interface and GeneratedCase container.

A testcase is the pure function: (config_seed, test_seed, strategy, params) -> GeneratedCase.
The same inputs must reproduce the source, input.bin and expected.bin byte-for-byte.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .rng import rng_for

KERNEL_ABI = """\
// Fixed kernel ABI shared by every testcase.
//   input:       input_words u64 words (input.bin)
//   output:      output_words u64 words (grid*block*2 by default)
//   input_words: u32 word count
//   seed:        u64 kernel seed from the manifest
extern "C" __global__ void test_kernel(
    const unsigned long long* __restrict__ input,
    unsigned long long* __restrict__ output,
    unsigned int input_words,
    unsigned long long seed);
"""


@dataclass
class GeneratedCase:
    """Fully materialized, validated testcase (source + inputs + expected output)."""

    test_id: str
    config_seed: int
    test_seed: int
    strategy: str
    params: Dict[str, Any]
    generator_version: str
    kernel_seed: int                # u64 kernel seed passed to test_kernel at launch
    source: str
    input_bytes: bytes
    expected_bytes: bytes           # raw u64 LE, grid*block*output_words_per_thread
    grid: List[int]
    block: List[int]
    input_words: int
    output_words: int
    output_words_per_thread: int
    target_sm: str
    cpu_oracle: str                 # oracle identifier (usually the strategy name)
    comparison_policy: str
    cuda_features: List[str] = field(default_factory=list)
    kernel_entry: str = "test_kernel"
    sanitizer_policy: str = "none"
    valid: bool = True
    reject_reason: str = ""
    # How the compiled testcase is executed. Legacy strategies expose a fixed
    # Driver ABI; generator_ir records its variable Driver checksum ABI below.
    # standalone_checksum remains supported for archived cases.
    execution_mode: str = "driver_kernel"
    # cpu_exact compares every stable result with expected.bin.  cross_build
    # has no independent CPU interpreter and treats agreement across the
    # compile matrix as the baseline while still detecting stable divergence.
    oracle_mode: str = "cpu_exact"
    # Optional JSON-serializable launch/setup contract for non-fixed Driver API
    # kernels.  The common driver_kernel mode has an implicit ABI; generator_ir
    # records its variable parameter list and module-global initialization here.
    execution_spec: Dict[str, Any] = field(default_factory=dict)
    # Optional archived IR (generator_ir only).  Kept out of case_id() and
    # manifest hashes so byte-for-byte source/input/expected semantics are
    # unchanged; reducer uses it as a fast exact-load path.
    ir_blob: Optional[bytes] = None

    # -- stable identity ----------------------------------------------------
    @property
    def source_sha256(self) -> str:
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()

    @property
    def input_sha256(self) -> str:
        return hashlib.sha256(self.input_bytes).hexdigest()

    @property
    def expected_sha256(self) -> str:
        return hashlib.sha256(self.expected_bytes).hexdigest()

    def case_id(self) -> str:
        """Stable dedup id: strategy + content-derived hash."""
        h = hashlib.sha256(
            json.dumps([self.strategy, self.params, self.test_seed,
                        self.source_sha256], sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        return f"{self.strategy}__{h}"

    def total_threads(self) -> int:
        g = (list(self.grid) + [1, 1, 1])[:3]
        b = (list(self.block) + [1, 1, 1])[:3]
        return g[0] * g[1] * g[2] * b[0] * b[1] * b[2]

    def manifest(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "case_id": self.case_id(),
            "config_seed": self.config_seed,
            "test_seed": self.test_seed,
            "kernel_seed": self.kernel_seed,
            "strategy": self.strategy,
            "strategy_params": self.params,
            "generator_version": self.generator_version,
            "kernel_entry": self.kernel_entry,
            "grid": self.grid,
            "block": self.block,
            "input_word_count": self.input_words,
            "output_word_count": self.output_words,
            "output_words_per_thread": self.output_words_per_thread,
            "target_sm": self.target_sm,
            "comparison_policy": self.comparison_policy,
            "cpu_oracle": self.cpu_oracle,
            "execution_mode": self.execution_mode,
            "oracle_mode": self.oracle_mode,
            "execution_spec": self.execution_spec,
            "sanitizer_policy": self.sanitizer_policy,
            "cuda_features": self.cuda_features,
            "hashes": {
                "source": f"sha256:{self.source_sha256}",
                "input": f"sha256:{self.input_sha256}",
                "expected": f"sha256:{self.expected_sha256}",
            },
            "valid": self.valid,
            "reject_reason": self.reject_reason,
        }


class Strategy:
    """Base class for generator strategies.

    Subclasses implement `generate` (source + input construction) and `cpu_oracle`
    (exact expected output computed from the SAME params and input bytes).
    """

    name = "base"
    target_opcodes: List[str] = []       # PTX/SASS opcodes this strategy stresses

    def generate(self, config_seed: int, test_seed: int, params: Dict[str, Any],
                 gen_version: str, target_sm: str) -> GeneratedCase:
        raise NotImplementedError

    def cpu_oracle(self, input_words: int, seed: int, params: Dict[str, Any],
                   input_bytes: bytes, grid: List[int], block: List[int],
                   output_words_per_thread: int) -> bytes:
        raise NotImplementedError

    # -- helpers ------------------------------------------------------------
    def _new_rng(self, config_seed: int, test_seed: int) -> "random.Random":
        return rng_for(config_seed, test_seed, self.name)


class StrategyRegistry:
    def __init__(self) -> None:
        self._strategies: Dict[str, Strategy] = {}

    def register(self, s: Strategy) -> None:
        self._strategies[s.name] = s

    def get(self, name: str) -> Strategy:
        if name not in self._strategies:
            raise KeyError(f"unknown strategy {name!r}; available: {sorted(self._strategies)}")
        return self._strategies[name]

    def names(self) -> List[str]:
        return sorted(self._strategies)

    def all(self) -> List[Strategy]:
        return [self._strategies[n] for n in self.names()]


_registry: Optional[StrategyRegistry] = None


def default_registry() -> StrategyRegistry:
    global _registry
    if _registry is None:
        _registry = StrategyRegistry()
        from .strategies import register_all
        register_all(_registry)
    return _registry
