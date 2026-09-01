"""Top-level declarations: functions, structs, launch config, Program."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from generator_ir.ir.statements import Block, Declare
from generator_ir.ir.typedecl import GType, Qual, TStruct


@dataclass
class LaunchConfig:
    """GPU launch + harness configuration carried by a Program."""
    grid: Tuple[int, int, int] = (4, 1, 1)
    block: Tuple[int, int, int] = (32, 1, 1)
    shared_bytes: int = 0
    arch: str = "sm_75"
    extra_headers: Set[str] = field(default_factory=set)
    body_repeat: int = 1          # repeats of the kernel body per launch
    n_elements: int = 0           # buffer size override (0 = derive from threads)
    kx_is_2d: bool = False           # matrix templates: buffers are n*n
    # ---- timing harness selection (mutually exclusive) ----
    time_ci: bool = False         # trimmed-mean CI harness
    time_clock64: bool = False    # device clock64() instrumentation
    time_loop: bool = False       # calibrated N-loop harness
    time_profiler: bool = False   # plain launches; external profiler times
    checksum_only: bool = False   # skip timing; one fresh checksum launch only
    # ---- generation metadata (used when re-entering mutation) ----
    observable_fold: bool = False
    codegen: bool = False
    generic_baseline: bool = False
    generation_features: Set[str] = field(default_factory=set)
    complexity: str = "medium"
    input_size_mult: int = 1
    input_dist: str = "hash"
    allow_volatile: bool = False
    needs_constant_mem: bool = False

    @property
    def threads_per_block(self) -> int:
        return self.block[0] * self.block[1] * self.block[2]

    @property
    def block_count(self) -> int:
        return self.grid[0] * self.grid[1] * self.grid[2]

    @property
    def kx_arch_num(self) -> int:
        """Numeric arch: 'sm_75' -> 75, 'gfx908' -> 908, 'gfx90a' -> 908.

        Handles the single-letter suffix form (gfx90a/gfx942 variants) by
        treating a trailing letter after 3 digits as a minor revision that
        rounds up to the next feature family (gfx90a exposes 908-class
        features).
        """
        s = self.arch.strip().lower()
        for prefix in ("sm_", "gfx"):
            if s.startswith(prefix):
                num = s[len(prefix):]
                if num.isdigit():
                    return int(num)
                # e.g. "90a": keep the numeric prefix and bump the family
                digits = ""
                for ch in num:
                    if ch.isdigit():
                        digits += ch
                    else:
                        break
                if not digits:
                    return 0
                suffix = num[len(digits):]
                base = int(digits)
                if suffix and len(digits) == 2:
                    return base * 10 + 8
                return base
        return 0


@dataclass(eq=False)
class Func:
    """Device or kernel function declaration."""
    name: str
    params: List[Declare]
    ret: GType
    body: Optional[Block] = None
    quals: List[Qual] = field(default_factory=list)
    launch_bounds: Optional[Tuple[int, ...]] = None
    maxnreg: Optional[int] = None
    cluster_dims: Optional[Tuple[int, int, int]] = None
    inline_hint: Optional[str] = None     # "__forceinline__" | "__noinline__"

    def __hash__(self) -> int:
        return id(self)

    def kx_is_kernel(self) -> bool:
        return Qual.KERNEL in self.quals

    def kx_is_device(self) -> bool:
        return Qual.DEVICE in self.quals

    def kx_has_body(self) -> bool:
        return self.body is not None

    def __repr__(self) -> str:
        return f"Func({self.name!r})"


@dataclass(eq=False)
class StructDecl:
    struct: TStruct

    def __hash__(self) -> int:
        return id(self)


class Program:
    """A translation unit: top-level declarations plus launch config."""

    def __init__(self, launch: Optional[LaunchConfig] = None) -> None:
        self.decls: List = []      # Func | StructDecl | Declare (device globals)
        self.launch: LaunchConfig = launch or LaunchConfig()

    def add(self, decl) -> None:
        self.decls.append(decl)

    def fetch_kernel(self) -> Optional[Func]:
        for d in self.decls:
            if isinstance(d, Func) and d.kx_is_kernel():
                return d
        return None

    def fetch_function(self, name: str) -> Optional[Func]:
        for d in self.decls:
            if isinstance(d, Func) and d.name == name:
                return d
        return None

    def functions(self) -> List[Func]:
        return [d for d in self.decls if isinstance(d, Func)]

    def __repr__(self) -> str:
        return f"Program(decls={len(self.decls)}, isa={self.launch.arch!r})"
