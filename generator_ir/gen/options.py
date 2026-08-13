"""Kernel-build parameters.

``KernelOpts`` is a dataclass bag controlling every generation axis:
tree-depth ceilings, statement budgets, complexity presets, the runtime-
amplification loop count, feature toggles, and opt-in diversity levers.
The internal method names, variable conventions, and code structure
intentionally differ from any reference implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

_LEVELS = ("low", "medium", "high", "stress")


@dataclass
class KernelOpts:
    """All knobs the generator reads to build one kernel."""

    seed: int = 42

    # ---- recursion ceilings ----
    expr_tree_limit: int = 5
    stmt_budget: int = 30
    nesting_loop_cap: int = 3
    nesting_branch_cap: int = 3

    # ---- runtime amplification ----
    amplify_target_ms: int = 800
    amplify_iters: int = 2048

    # ---- feature toggles ----
    enable_smem: bool = True
    enable_atomics: bool = True
    enable_warp: bool = True
    enable_barriers: bool = True

    # ---- launch geometry ----
    threads_per_block_dim: Tuple[int, int, int] = (256, 1, 1)
    blocks_per_grid_dim: Tuple[int, int, int] = (128, 1, 1)
    smem_bytes: int = 0

    # ---- complexity preset ----
    complexity: str = "medium"

    # ---- type palette ----
    type_allowlist: Optional[List[str]] = None
    scalar_param_count: int = 3
    use_full_int_widths: bool = False

    element_count: int = 32768

    target: str = "hip"
    isa: str = "sm_75"

    # ---- differential-testing harness ----
    checksum_only: bool = False

    # ---- stress section toggles ----
    do_bank_stress: bool = False
    reg_pressure_level: str = "none"
    do_vector_stress: bool = False
    do_switch_stress: bool = False
    do_barrier_stress: bool = False

    # ---- observable-fold knobs ----
    observable_fold: bool = False
    prob_u64_divmod: float = 0.5
    prob_divergent_switch: float = 0.5
    prob_strided_unroll: float = 0.5
    prob_runtime_loop_bound: float = 0.5
    strided_unroll_depth: int = 32
    # Host input-buffer kx_multiplier: strided-unroll reads reach
    # input[(tid%n) + (U-1)*n] (< U*n), so the harness sizes the input buffer
    # input_size_mult * n.  Blocks that widen the reach bump this at gen time.
    input_size_mult: int = 1

    # ---- per-seed diversity levers ----
    prob_float_pool: float = 0.0
    prob_raw_div: float = 0.0
    prob_wide_consts: float = 0.0
    input_distribution: str = "hash"
    randomize_launch: bool = False

    # ---- weaving tier ----
    codegen: bool = False
    codegen_repeat: int = 1024

    # ---- volatile policy ----
    allow_volatile: bool = False

    def __post_init__(self) -> None:
        self._configure_preset()

    def _configure_preset(self) -> None:
        """Adjust budgets per complexity level (monotonic: low ≤ med ≤ high)."""
        handler = {
            "low": self._preset_minimal,
            "medium": self._preset_balanced,
            "high": self._preset_rich,
            "stress": self._preset_maximal,
        }.get(self.complexity)
        if handler:
            handler()

    def _preset_minimal(self) -> None:
        """Low complexity: tight ceilings, no stress sections."""
        self.expr_tree_limit = min(self.expr_tree_limit, 3)
        self.stmt_budget = min(self.stmt_budget, 15)
        self.nesting_loop_cap = min(self.nesting_loop_cap, 2)
        self.nesting_branch_cap = min(self.nesting_branch_cap, 2)

    def _preset_balanced(self) -> None:
        """Medium: moderate depth, selective stress."""
        self.expr_tree_limit = min(self.expr_tree_limit, 3)
        self.stmt_budget = min(self.stmt_budget, 30)

    def _preset_rich(self) -> None:
        """High: wide type palette, all stress sections at low intensity."""
        self.expr_tree_limit = 3
        self.stmt_budget = max(self.stmt_budget, 50)
        self.nesting_loop_cap = max(self.nesting_loop_cap, 3)
        self.nesting_branch_cap = max(self.nesting_branch_cap, 3)
        self.scalar_param_count = max(self.scalar_param_count, 12)
        self.use_full_int_widths = True
        self._activate_stress(reg_level="low")

    def _preset_maximal(self) -> None:
        """Stress: everything on, high register pressure."""
        self.expr_tree_limit = 4
        self.stmt_budget = 300
        self.nesting_loop_cap = 3
        self.nesting_branch_cap = 3
        self.scalar_param_count = 18
        self.use_full_int_widths = True
        if not self.observable_fold:
            self.amplify_iters = 2
        self._activate_stress(reg_level="high")

    def _activate_stress(self, reg_level: str) -> None:
        """Turn on all stress toggles at a given register-pressure intensity."""
        self.do_bank_stress = True
        self.reg_pressure_level = reg_level
        self.do_vector_stress = True
        self.do_switch_stress = True
        self.do_barrier_stress = True

    @property
    def kx_total_threads(self) -> int:
        bx, by, bz = self.threads_per_block_dim
        gx, gy, gz = self.blocks_per_grid_dim
        return bx * by * bz * gx * gy * gz

    def compute_amplification(self) -> int:
        """How many times to repeat the kernel body per launch."""
        iters = self.amplify_iters
        if iters < 100:
            return iters
        if self.observable_fold:
            return 64
        if self.use_full_int_widths:
            return 4
        if self.stmt_budget >= 50:
            return 1000
        return 3000

    def repeat_rounds(self) -> int:
        """Compatibility alias for compute_amplification()."""
        return self.compute_amplification()

# Alias for backward compatibility.

    # Backward-compatible property aliases (old → new field names).
    @property
    def max_expr_depth(self):
        return self.expr_tree_limit

    @max_expr_depth.setter
    def max_expr_depth(self, v):
        self.expr_tree_limit = v

    @property
    def max_stmt_count(self):
        return self.stmt_budget

    @max_stmt_count.setter
    def max_stmt_count(self, v):
        self.stmt_budget = v

    @property
    def max_loop_depth(self):
        return self.nesting_loop_cap

    @max_loop_depth.setter
    def max_loop_depth(self, v):
        self.nesting_loop_cap = v

    @property
    def max_if_depth(self):
        return self.nesting_branch_cap

    @max_if_depth.setter
    def max_if_depth(self, v):
        self.nesting_branch_cap = v

    @property
    def body_repeat_target_ms(self):
        return self.amplify_target_ms

    @body_repeat_target_ms.setter
    def body_repeat_target_ms(self, v):
        self.amplify_target_ms = v

    @property
    def body_repeat_rounds(self):
        return self.amplify_iters

    @body_repeat_rounds.setter
    def body_repeat_rounds(self, v):
        self.amplify_iters = v

    @property
    def use_shared_memory(self):
        return self.enable_smem

    @use_shared_memory.setter
    def use_shared_memory(self, v):
        self.enable_smem = v

    @property
    def use_atomics(self):
        return self.enable_atomics

    @use_atomics.setter
    def use_atomics(self, v):
        self.enable_atomics = v

    @property
    def use_warp_ops(self):
        return self.enable_warp

    @use_warp_ops.setter
    def use_warp_ops(self, v):
        self.enable_warp = v

    @property
    def use_sync(self):
        return self.enable_barriers

    @use_sync.setter
    def use_sync(self, v):
        self.enable_barriers = v

    @property
    def block_dim(self):
        return self.threads_per_block_dim

    @block_dim.setter
    def block_dim(self, v):
        self.threads_per_block_dim = v

    @property
    def grid_dim(self):
        return self.blocks_per_grid_dim

    @grid_dim.setter
    def grid_dim(self, v):
        self.blocks_per_grid_dim = v

    @property
    def shared_mem_bytes(self):
        return self.smem_bytes

    @shared_mem_bytes.setter
    def shared_mem_bytes(self, v):
        self.smem_bytes = v

    @property
    def allowed_types(self):
        return self.type_allowlist

    @allowed_types.setter
    def allowed_types(self, v):
        self.type_allowlist = v

    @property
    def num_scalar_params(self):
        return self.scalar_param_count

    @num_scalar_params.setter
    def num_scalar_params(self, v):
        self.scalar_param_count = v

    @property
    def use_diverse_types(self):
        return self.use_full_int_widths

    @use_diverse_types.setter
    def use_diverse_types(self, v):
        self.use_full_int_widths = v

    @property
    def n_elements(self):
        return self.element_count

    @n_elements.setter
    def n_elements(self, v):
        self.element_count = v

    @property
    def backend(self):
        return self.target

    @backend.setter
    def backend(self, v):
        self.target = v

    @property
    def arch(self):
        return self.isa

    @arch.setter
    def arch(self, v):
        self.isa = v

    @property
    def stress_bank_conflict(self):
        return self.do_bank_stress

    @stress_bank_conflict.setter
    def stress_bank_conflict(self, v):
        self.do_bank_stress = v

    @property
    def stress_register_pressure(self):
        return self.reg_pressure_level

    @stress_register_pressure.setter
    def stress_register_pressure(self, v):
        self.reg_pressure_level = v

    @property
    def stress_vectorize_loads(self):
        return self.do_vector_stress

    @stress_vectorize_loads.setter
    def stress_vectorize_loads(self, v):
        self.do_vector_stress = v

    @property
    def stress_switch_divergence(self):
        return self.do_switch_stress

    @stress_switch_divergence.setter
    def stress_switch_divergence(self, v):
        self.do_switch_stress = v

    @property
    def stress_barrier_span(self):
        return self.do_barrier_stress

    @stress_barrier_span.setter
    def stress_barrier_span(self, v):
        self.do_barrier_stress = v

    @property
    def p_u64_divmod(self):
        return self.prob_u64_divmod

    @p_u64_divmod.setter
    def p_u64_divmod(self, v):
        self.prob_u64_divmod = v

    @property
    def p_divergent_switch(self):
        return self.prob_divergent_switch

    @p_divergent_switch.setter
    def p_divergent_switch(self, v):
        self.prob_divergent_switch = v

    @property
    def p_strided_unroll(self):
        return self.prob_strided_unroll

    @p_strided_unroll.setter
    def p_strided_unroll(self, v):
        self.prob_strided_unroll = v

    @property
    def p_runtime_loop_bound(self):
        return self.prob_runtime_loop_bound

    @p_runtime_loop_bound.setter
    def p_runtime_loop_bound(self, v):
        self.prob_runtime_loop_bound = v

    @property
    def strided_unroll_max(self):
        return self.strided_unroll_depth

    @strided_unroll_max.setter
    def strided_unroll_max(self, v):
        self.strided_unroll_depth = v

    @property
    def p_float_types(self):
        return self.prob_float_pool

    @p_float_types.setter
    def p_float_types(self, v):
        self.prob_float_pool = v

    @property
    def p_raw_division(self):
        return self.prob_raw_div

    @p_raw_division.setter
    def p_raw_division(self, v):
        self.prob_raw_div = v

    @property
    def p_wide_constants(self):
        return self.prob_wide_consts

    @p_wide_constants.setter
    def p_wide_constants(self, v):
        self.prob_wide_consts = v

    @property
    def input_dist(self):
        return self.input_distribution

    @input_dist.setter
    def input_dist(self, v):
        self.input_distribution = v

    @property
    def vary_launch(self):
        return self.randomize_launch

    @vary_launch.setter
    def vary_launch(self, v):
        self.randomize_launch = v


GenConfig = KernelOpts
