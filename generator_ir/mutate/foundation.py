"""Mutation operator base class and shared helpers."""

from __future__ import annotations

import random
from typing import List, Optional

from generator_ir.ir.program import Program
from generator_ir.ir.traversal import clone_program


class Mutation:
    """Base class for mutation operators.

    Subclasses set ``name``/``category`` and implement ``perturb``.
    Directional operators additionally override ``perturb_opt``/``perturb_deopt``.
    Sweepable operators set ``sweep_space``.
    """

    name: str = "mutation"
    category: str = "memory"           # memory | divergence | compute
    sweep_space: Optional[list] = None
    #: Operators that frequently produce uncompilable output can be parked.
    enabled: bool = True

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        raise NotImplementedError

    def perturb_opt(self, program: Program, rng: random.Random, param=None) -> Program:
        return self.perturb(program, rng, param)

    def perturb_deopt(self, program: Program, rng: random.Random, param=None) -> Program:
        return self.perturb(program, rng, param)

    def scan(self, program: Program, rng: random.Random) -> List[Program]:
        if not self.sweep_space:
            raise ValueError(f"{self.name} has no scan space")
        return [self.perturb(clone_program(program), rng, p)
                for p in self.sweep_space]

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


def cuda_sm_version(program: Program) -> int:
    """Numeric SM version for CUDA targets, else 0."""
    arch = getattr(program.launch, "arch", "").lower()
    if not arch.startswith("sm_"):
        return 0
    try:
        return int(arch[3:].rstrip("a"))
    except ValueError:
        return 0


def kx_is_cuda(program: Program, min_sm: int = 0) -> bool:
    return cuda_sm_version(program) >= max(0, min_sm) and \
        getattr(program.launch, "arch", "").lower().startswith("sm_")
