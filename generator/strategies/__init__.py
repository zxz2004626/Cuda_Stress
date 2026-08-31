"""Strategy implementations.  `register_all` populates the shared registry."""

from __future__ import annotations

from ..base import StrategyRegistry
from .inline_ptx import InlinePtx
from .generator_ir import GeneratorIR
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
    GeneratorIR,
]

# Strategies with a byte-for-byte independent CPU oracle.  generator_ir is
# intentionally excluded: it uses stable cross-build checksum consensus.
CPU_ORACLE_ALL = [cls for cls in ALL if cls is not GeneratorIR]


def register_all(registry: StrategyRegistry) -> None:
    for cls in ALL:
        registry.register(cls())
