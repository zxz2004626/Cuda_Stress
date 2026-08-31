"""Strategy implementations.  `register_all` populates the shared registry.

Only generator_ir is registered: the original fixed-ABI generator strategies
have been removed from the generation path.
"""

from __future__ import annotations

from ..base import StrategyRegistry
from .generator_ir import GeneratorIR

ALL = [
    GeneratorIR,
]


def register_all(registry: StrategyRegistry) -> None:
    for cls in ALL:
        registry.register(cls())
