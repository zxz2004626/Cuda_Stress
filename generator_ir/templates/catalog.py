"""Template registry with regression-aware default weights."""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from generator_ir.templates.skeleton import KernelTemplate


# Base weights favor templates that historically expose regressions.
PREFERENCE_TABLE = {
    "isa_stress": 2.0,
    "reg_spill_probe": 2.0,
    "smem_bank_stress": 1.5,
    "sync_span_live": 1.5,
    "vec_load_probe": 1.5,
    "tiled_gemm": 1.5,
    "cf_switch_tree": 1.2,
    "bin_count": 1.2,
}


class TemplateRegistry:
    def __init__(self) -> None:
        self._templates: Dict[str, KernelTemplate] = {}

    def register(self, template: KernelTemplate) -> None:
        self._templates[template.name] = template

    def get(self, name: str) -> Optional[KernelTemplate]:
        return self._templates.get(name)

    def random(self, rng: random.Random) -> KernelTemplate:
        return rng.choice(list(self._templates.values()))

    def kx_weighted_random(self, rng: random.Random, schedule=None) -> KernelTemplate:
        templates = list(self._templates.values())
        weights = [(t.name, PREFERENCE_TABLE.get(t.name, 1.0)) for t in templates]
        if schedule is not None:
            weights = schedule.apply("template", weights)
        names = [n for n, _ in weights]
        ws = [w for _, w in weights]
        return self._templates[rng.choices(names, weights=ws, k=1)[0]]

    def kx_all_names(self) -> List[str]:
        return list(self._templates.keys())

    def kx_all_templates(self) -> List[KernelTemplate]:
        return list(self._templates.values())

    def __len__(self) -> int:
        return len(self._templates)


def kx_build_default_registry() -> TemplateRegistry:
    from generator_ir.templates.structural import (
        ElementwiseTemplate, ReductionTemplate, StencilTemplate,
        MatmulTemplate, ScanTemplate, HistogramTemplate, TransposeTemplate,
    )
    from generator_ir.templates.pressure import (
        BankConflictTemplate, BarrierPressureTemplate, SwitchDivergenceTemplate,
        GatherScatterTemplate, WarpCommFallbackTemplate, PermutationCommTemplate,
        RegisterPressureTemplate, VectorizationStressTemplate, PTXStressTemplate,
    )
    reg = TemplateRegistry()
    reg.register(ElementwiseTemplate())
    reg.register(ReductionTemplate())
    reg.register(StencilTemplate())
    reg.register(MatmulTemplate())
    reg.register(ScanTemplate())
    reg.register(HistogramTemplate())
    reg.register(TransposeTemplate())
    reg.register(BankConflictTemplate())
    reg.register(BarrierPressureTemplate())
    reg.register(SwitchDivergenceTemplate())
    reg.register(GatherScatterTemplate())
    reg.register(WarpCommFallbackTemplate())
    reg.register(PermutationCommTemplate())
    reg.register(RegisterPressureTemplate())
    reg.register(VectorizationStressTemplate())
    reg.register(PTXStressTemplate())
    return reg
