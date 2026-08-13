"""Mutation pipeline: operator registry, selection modes, retry/safety loop."""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional, Tuple

from generator_ir.ir.program import Program
from generator_ir.ir.traversal import (
    clone_program, kernel_of, collect, fix_free_variables,
    has_divergent_barrier,
)
from generator_ir.gen.weights import kx_schedule_for_seed
from generator_ir.mutate.foundation import Mutation
from generator_ir.mutate.mem_ops import (
    MemRegisterToLocal, MemPromotion, MemStrideAlteration,
    MemVolatileQualifier, MemRestrictQualifier, MemPadding,
    ExtAccessPatternSweep, ExtSharedStaging, ExtBankConflict,
    ExtCacheHintSweep, ExtShuffleSharedSwap, ExtVectorizationWidth,
    ExtPrefetchDistance, ExtKernelFusionFission,
)
from generator_ir.mutate.div_ops import (
    DivDivergenceSource, DivDivergencePropagation, DivBarrierInsertRemove,
    DivAtomicSubstitution, ExtBranchToArithmetic, ExtReconvergenceShift,
    ExtPredicatedBranched, ExtWarpCentric, ExtSwitchDivergence,
    ExtBarrierRelocation, ExtCudaWarpCollectives, ExtAsymmetricDivergence,
)
from generator_ir.mutate.comp_ops import (
    CompWorkCoarsen, CompFmaDecomposition, CompStrengthReductionReversal,
    CompTripCountOpacify, CompLoopFission, CompInlineSwitch,
    ExtUnrollFactor, ExtBlockSizeSweep, ExtRegisterPressureInflation,
    ExtLaunchBoundsSweep, ExtCudaKernelAttrSweep, ExtCudaDpxRewrite,
)
from generator_ir.mutate.emi_ops import (
    EmiDeadCode, EmiTrueGuard, EmiFakeDivergence, ExtSplice,
)
from generator_ir.mutate.weave_mutator import WeaveInsertMutator

log = logging.getLogger(__name__)

MODE_MEM = "mem"
MODE_LOC = "loc"
MODE_REG = "reg"
MODE_FULL = "full"
MODE_COMBO = "combo"

VALID_MODES = frozenset({MODE_MEM, MODE_LOC, MODE_REG, MODE_FULL, MODE_COMBO})
_MODE_TO_CATEGORY = {MODE_MEM: "memory", MODE_LOC: "divergence", MODE_REG: "compute"}

# Category weights for the scan of mutation combinations.


class MutationPipeline:
    """Orchestrates all mutation operators.

    paper operators: MEM1..MEM6 (MEM4 parked), DIV1..DIV4, COMP1..COMP6
    extension operators: EXT_*
    """

    def __init__(self, seed: int = 42, codegen: bool = False) -> None:
        self.rng = random.Random(seed)
        self._codegen = bool(codegen)

        self.paper_mutations: List[Mutation] = [
            MemRegisterToLocal(),
            MemPromotion(),
            MemStrideAlteration(),
            # MemVolatileQualifier is registered but parked (enabled=False).
            MemRestrictQualifier(),
            MemPadding(),
            DivDivergenceSource(),
            DivDivergencePropagation(),
            DivBarrierInsertRemove(),
            DivAtomicSubstitution(),
            CompWorkCoarsen(),
            CompFmaDecomposition(),
            CompStrengthReductionReversal(),
            CompTripCountOpacify(),
            CompLoopFission(),
            CompInlineSwitch(),
        ]
        self.parked_mutations: List[Mutation] = [
            MemVolatileQualifier(),
        ]
        self.extension_mutations: List[Mutation] = [
            ExtSharedStaging(),
            ExtAccessPatternSweep(),
            ExtBankConflict(),
            ExtCacheHintSweep(),
            ExtShuffleSharedSwap(),
            ExtVectorizationWidth(),
            ExtPrefetchDistance(),
            ExtKernelFusionFission(),
            ExtBranchToArithmetic(),
            ExtReconvergenceShift(),
            ExtPredicatedBranched(),
            ExtWarpCentric(),
            ExtSwitchDivergence(),
            ExtBarrierRelocation(),
            ExtCudaWarpCollectives(),
            ExtAsymmetricDivergence(),
            ExtRegisterPressureInflation(),
            ExtLaunchBoundsSweep(),
            ExtCudaKernelAttrSweep(),
            ExtCudaDpxRewrite(),
            ExtUnrollFactor(),
            ExtBlockSizeSweep(),
            EmiDeadCode(),
            EmiTrueGuard(),
            EmiFakeDivergence(),
            ExtSplice(),
        ]
        # Weave/codegen mutator: ADDED ONLY under the --codegen gate.  It
        # ignores the legacy corpus (operates only on weave-shaped kernels)
        # and is absent from the default pipeline (perffuzzer decision 12).
        self.codegen_mutations: List[Mutation] = []
        if self._codegen:
            self.codegen_mutations.append(WeaveInsertMutator())
        self.mutations: List[Mutation] = (
            self.paper_mutations + self.extension_mutations
            + self.codegen_mutations)
        self._map: Dict[str, Mutation] = {m.name: m for m in self.mutations}
        self._fail_counts: Dict[str, int] = {}
        self._last_chain: List[str] = []

    # ------------------------------------------------------------------
    # registry queries
    # ------------------------------------------------------------------

    @property
    def kx_fail_counts(self) -> Dict[str, int]:
        return dict(self._fail_counts)

    @property
    def last_mutation_chain(self) -> List[str]:
        return list(self._last_chain)

    @property
    def last_mutator_name(self) -> str:
        return "+".join(self._last_chain) if self._last_chain else "none"

    def fetch_mutation_names(self) -> List[str]:
        return [m.name for m in self.mutations]

    def fetch_paper_mutation_names(self) -> List[str]:
        return [m.name for m in self.paper_mutations]

    def fetch_extension_mutation_names(self) -> List[str]:
        return [m.name for m in self.extension_mutations]

    def fetch_scan_mutations(self) -> List[str]:
        return [m.name for m in self.mutations if m.sweep_space]

    def fetch_mutations_by_category(self, category: str) -> List[str]:
        return [m.name for m in self.mutations if m.category == category]

    def fetch_mutations_by_mode(self, mode: str) -> List[str]:
        if mode not in VALID_MODES:
            raise ValueError(f"unknown mode {mode!r}")
        if mode in (MODE_FULL, MODE_COMBO):
            return self.fetch_mutation_names()
        return self.fetch_mutations_by_category(_MODE_TO_CATEGORY[mode])

    def fetch_mutation(self, name: str) -> Optional[Mutation]:
        return self._map.get(name)

    def kx_set_donor_corpus(self, corpus) -> None:
        for m in self.mutations:
            if isinstance(m, ExtSplice):
                m.kx_set_donor_corpus(corpus)

    # ------------------------------------------------------------------
    # selection machinery
    # ------------------------------------------------------------------

    def _mode_pool(self, mode: str, paper_only: bool) -> List[Mutation]:
        if mode not in VALID_MODES:
            raise ValueError(f"unknown mode {mode!r}")
        pool = self.paper_mutations if paper_only else self.mutations
        if mode in (MODE_FULL, MODE_COMBO):
            return list(pool)
        cat = _MODE_TO_CATEGORY[mode]
        return [m for m in pool if m.category == cat]

    def _kx_swarm_pool(self, pool: List[Mutation]) -> Tuple[List[Mutation], List[float]]:
        """Per-call swarm subset + jittered weights."""
        sched = kx_schedule_for_seed(self.rng.randrange(1 << 30))
        table = sched.apply("mutator", [(m.name, 1.0) for m in pool])
        wmap = dict(table)
        active = [m for m in pool if m.name in wmap]
        return active, [wmap[m.name] for m in active]

    # ------------------------------------------------------------------
    # safety / fingerprint
    # ------------------------------------------------------------------

    @staticmethod
    def fingerprint(program: Program) -> int:
        """Structural fingerprint for change detection (no-op rejection)."""
        parts = []
        for n in collect(program, lambda _: True):
            op = getattr(n, "op", None)
            val = getattr(n, "value", None)
            parts.append((
                type(n).__name__,
                str(op) if op is not None else None,
                getattr(n, "name", None),
                val if isinstance(val, (int, float, bool, str)) else None,
                getattr(n, "text", None) if isinstance(
                    getattr(n, "text", None), str) else None,
                getattr(n, "stride", None),
                tuple(str(q) for q in getattr(n, "quals", []) or []) or None,
                str(getattr(n, "gtype", None)),
                str(getattr(n, "space", None)),
                # Emitted Func metadata that mutations change but the per-node
                # fields above miss: __forceinline__/__noinline__ (inline_hint)
                # and __cluster_dims__ (cluster_dims).  Without these a hint-flip
                # or cluster-only sweep is indistinguishable from a no-op and the
                # random/combo pipelines silently drop it.
                getattr(n, "inline_hint", None),
                getattr(n, "cluster_dims", None),
            ))
        return hash(tuple(parts))

    def _config_fingerprint(self, program: Program) -> int:
        h = 0
        kernel = kernel_of(program)
        if kernel is not None:
            for i, q in enumerate(kernel.quals):
                h ^= hash(q.name) << ((i * 4) % 28)
            if kernel.launch_bounds:
                h ^= hash(kernel.launch_bounds)
            if kernel.maxnreg:
                h ^= hash(kernel.maxnreg) << 8
        cfg = program.launch
        h ^= (hash(cfg.block) ^ (hash(cfg.grid) << 16))
        return (self.fingerprint(program) ^ (h << 8)) & 0xFFFFFFFFFFFFFFFF

    def _kx_is_safe(self, program: Program) -> bool:
        kernel = kernel_of(program)
        if kernel is None or kernel.body is None:
            return True
        return not has_divergent_barrier(kernel.body)

    # ------------------------------------------------------------------
    # mutation entry points
    # ------------------------------------------------------------------

    def perturb_random(self, program: Program, num_mutations: int = 1,
                      paper_only: bool = False,
                      mode: str = MODE_FULL) -> Program:
        """Apply N random mutations with no-op/unsafe rejection + retry."""
        self._last_chain = []
        if mode == MODE_COMBO:
            result, applied = self.perturb_combo(program, paper_only=paper_only)
            self._last_chain = list(applied)
            return result

        pool = self._mode_pool(mode, paper_only)
        active, weights = self._kx_swarm_pool(pool)
        result = clone_program(program)
        applied = attempts = 0
        budget = num_mutations + 3 * len(pool)
        while applied < num_mutations and attempts < budget:
            attempts += 1
            mut = self.rng.choices(active, weights=weights, k=1)[0]
            before = self._config_fingerprint(result)
            try:
                cand = mut.perturb(result, self.rng)
            except Exception as exc:
                log.debug("mutation %s failed: %s", mut.name, exc)
                self._fail_counts[mut.name] = self._fail_counts.get(mut.name, 0) + 1
                continue
            if cand is not None:
                fix_free_variables(cand)
            if cand is None or self._config_fingerprint(cand) == before:
                continue
            if not self._kx_is_safe(cand):
                continue
            result = cand
            applied += 1
            self._last_chain.append(mut.name)
        return result

    def _mutate_directional(self, program: Program, num_mutations: int,
                            paper_only: bool, direction: str,
                            mode: str = MODE_FULL) -> Tuple[Program, List[str]]:
        pool = self._mode_pool(mode, paper_only)
        if not pool:
            return clone_program(program), []
        active, weights = self._kx_swarm_pool(pool)
        result = clone_program(program)
        applied: List[str] = []
        for _ in range(num_mutations):
            mut = self.rng.choices(active, weights=weights, k=1)[0]
            try:
                result = getattr(mut, direction)(result, self.rng)
                fix_free_variables(result)
                applied.append(mut.name)
            except Exception as exc:
                log.debug("mutation %s.%s failed: %s", mut.name, direction, exc)
                self._fail_counts[mut.name] = self._fail_counts.get(mut.name, 0) + 1
        self._last_chain = list(applied)
        return result, applied

    def perturb_opt_random(self, program: Program, num_mutations: int = 1,
                          paper_only: bool = True) -> Program:
        result, _ = self._mutate_directional(
            program, num_mutations, paper_only, "perturb_opt")
        return result

    def perturb_deopt_random(self, program: Program, num_mutations: int = 1,
                            paper_only: bool = True) -> Program:
        result, _ = self._mutate_directional(
            program, num_mutations, paper_only, "perturb_deopt")
        return result

    def perturb_opt_by_mode(self, program: Program, mode: str,
                           num_mutations: int = 1,
                           paper_only: bool = True) -> Tuple[Program, List[str]]:
        return self._mutate_directional(program, num_mutations, paper_only,
                                        "perturb_opt", mode)

    def perturb_deopt_by_mode(self, program: Program, mode: str,
                             num_mutations: int = 1,
                             paper_only: bool = True) -> Tuple[Program, List[str]]:
        return self._mutate_directional(program, num_mutations, paper_only,
                                        "perturb_deopt", mode)

    def perturb_by_mode(self, program: Program, mode: str,
                       num_mutations: int = 1,
                       paper_only: bool = False) -> Tuple[Program, List[str]]:
        """Apply N random mutations from a mode category, returning applied names."""
        self._last_chain = []
        if mode == MODE_COMBO:
            result, applied = self.perturb_combo(program, paper_only=paper_only)
            self._last_chain = list(applied)
            return result, applied

        pool = self._mode_pool(mode, paper_only)
        if not pool:
            raise ValueError(f"no mutations for mode {mode}")
        active, weights = self._kx_swarm_pool(pool)
        result = clone_program(program)
        applied: List[str] = []
        for _ in range(num_mutations):
            for _retry in range(5):
                mut = self.rng.choices(active, weights=weights, k=1)[0]
                before = self._config_fingerprint(result)
                try:
                    cand = mut.perturb(result, self.rng)
                    fix_free_variables(cand)
                except Exception as exc:
                    log.debug("mutation %s failed: %s", mut.name, exc)
                    self._fail_counts[mut.name] = self._fail_counts.get(mut.name, 0) + 1
                    continue
                if self._config_fingerprint(cand) != before:
                    if not self._kx_is_safe(cand):
                        continue  # reject unsafe candidate
                    result = cand
                    applied.append(mut.name)
                    break
            else:
                log.debug("mutation retries exhausted (no change)")
        self._last_chain = list(applied)
        return result, applied

    def perturb_combo(self, program: Program,
                     paper_only: bool = False) -> Tuple[Program, List[str]]:
        """One mutator from each category (memory, divergence, compute)."""
        result = clone_program(program)
        applied: List[str] = []
        for cat in ("memory", "divergence", "compute"):
            pool = self.paper_mutations if paper_only else self.mutations
            cat_pool = [m for m in pool if m.category == cat]
            if not cat_pool:
                continue
            for _retry in range(5):
                mut = self.rng.choice(cat_pool)
                before = self._config_fingerprint(result)
                try:
                    cand = mut.perturb(result, self.rng)
                except Exception as exc:
                    log.debug("combo %s failed: %s", mut.name, exc)
                    self._fail_counts[mut.name] = self._fail_counts.get(mut.name, 0) + 1
                    continue
                if self._config_fingerprint(cand) != before:
                    fix_free_variables(cand)
                    result = cand
                    applied.append(mut.name)
                    break
        self._last_chain = list(applied)
        return result, applied

    def perturb_by_name(self, program: Program, name: str, param=None) -> Program:
        mut = self._map.get(name)
        if mut is None:
            raise ValueError(f"unknown mutation: {name}")
        return mut.perturb(clone_program(program), self.rng, param)

    def perturb_by_category(self, program: Program, category: str,
                           num_mutations: int = 1) -> Program:
        """Apply random mutations from one category (memory|divergence|compute)."""
        category_mutations = [m for m in self.mutations if m.category == category]
        if not category_mutations:
            raise ValueError(f"unknown category: {category}")
        result = clone_program(program)
        for _ in range(num_mutations):
            mut = self.rng.choice(category_mutations)
            try:
                result = mut.perturb(result, self.rng)
            except Exception:
                continue
        return result

    def scan_one(self, program: Program, name: str) -> List[Program]:
        mut = self._map.get(name)
        if mut is None:
            raise ValueError(f"unknown mutation: {name}")
        if not mut.sweep_space:
            raise ValueError(f"mutation {name} does not support sweeping")
        return mut.scan(program, self.rng)

    def scan_all(self, program: Program) -> Dict[str, List[Program]]:
        out: Dict[str, List[Program]] = {}
        for mut in self.mutations:
            if not mut.sweep_space:
                continue
            try:
                out[mut.name] = mut.scan(program, self.rng)
            except Exception:
                out[mut.name] = []
        return out

    def compose(self, program: Program, names: List[str],
                params: Optional[List] = None) -> Program:
        result = clone_program(program)
        params = params or [None] * len(names)
        for name, param in zip(names, params):
            mut = self._map.get(name)
            if mut is None:
                raise ValueError(f"unknown mutation: {name}")
            try:
                result = mut.perturb(result, self.rng, param)
            except Exception:
                continue
        return result

    def summary(self) -> str:
        lines = [
            f"gpuperf mutation pipeline "
            f"({len(self.paper_mutations)} paper + "
            f"{len(self.extension_mutations)} extension = "
            f"{len(self.mutations)} operators; "
            f"{len(self.parked_mutations)} parked)",
            "",
        ]
        for title, pool in (("PAPER", self.paper_mutations),
                            ("EXTENSION", self.extension_mutations),
                            ("PARKED", self.parked_mutations)):
            lines.append(f"  [{title}]")
            for cat in ("memory", "divergence", "compute"):
                group = [m for m in pool if m.category == cat]
                if not group:
                    continue
                lines.append(f"    {cat}:")
                for m in group:
                    sw = f" scan={m.sweep_space}" if m.sweep_space else ""
                    lines.append(f"      {m.name}{sw}")
            lines.append("")
        return "\n".join(lines)
