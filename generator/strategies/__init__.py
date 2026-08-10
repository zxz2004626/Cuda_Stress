"""Strategy implementations.  `register_all` populates the shared registry."""

from __future__ import annotations

from ..base import StrategyRegistry
from .inline_ptx import InlinePtx
from .integer_chain import IntegerChain
from .memory_address import MemoryAddress
from .predicate_pressure import PredicatePressure
from .register_pressure import RegisterPressure
from .warp_mixed import WarpMixed

ALL = [
    PredicatePressure,
    IntegerChain,
    MemoryAddress,
    WarpMixed,
    RegisterPressure,
    InlinePtx,
]


def register_all(registry: StrategyRegistry) -> None:
    for cls in ALL:
        registry.register(cls())
