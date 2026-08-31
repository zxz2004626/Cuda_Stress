"""GPU architecture descriptors and feature gating.

Covers NVIDIA ``sm_NN`` targets and AMD ``gfxNNN[x]`` targets.  Feature
predicates answer "is intrinsic/type/construct X legal on this target" for
generation and mutation gating.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import List


class Vendor(Enum):
    NVIDIA = auto()
    AMD = auto()


class Arch:
    """A concrete GPU target (compute capability or GFX version)."""

    def __init__(self, name: str = "sm_75") -> None:
        self.name = name.strip().lower()
        self.vendor = Vendor.AMD if self.name.startswith("gfx") else Vendor.NVIDIA
        self.version = self._read_version(self.name)

    @staticmethod
    def _read_version(name: str) -> int:
        for prefix in ("sm_", "gfx"):
            if name.startswith(prefix):
                num = name[len(prefix):]
                if num.isdigit():
                    return int(num)
                digits = ""
                for ch in num:
                    if ch.isdigit():
                        digits += ch
                    else:
                        break
                if not digits:
                    raise ValueError(f"cannot parse arch name: {name!r}")
                base = int(digits)
                # gfx90a / gfx942-style suffix: two digits + letter rounds up
                # into the next feature family (90a -> 908-class features).
                if len(digits) == 2 and len(num) > len(digits):
                    return base * 10 + 8
                return base
        raise ValueError(f"cannot parse arch name: {name!r}")

    @classmethod
    def parse(cls, s: str) -> "Arch":
        """Accept 'sm_75', '75', 'gfx908', ..."""
        s = s.strip().lower()
        if s.isdigit():
            s = f"sm_{s}"
        return cls(s)

    # -- basic properties ------------------------------------------------

    @property
    def kx_is_nvidia(self) -> bool:
        return self.vendor is Vendor.NVIDIA

    @property
    def kx_is_amd(self) -> bool:
        return self.vendor is Vendor.AMD

    @property
    def flag(self) -> str:
        """Value for the compiler arch flag."""
        return self.name

    @property
    def kx_warp_size(self) -> int:
        if self.kx_is_amd:
            return 32 if self.version >= 1000 else 64  # RDNA vs CDNA
        return 32

    # -- feature gates -----------------------------------------------------

    def _sm(self, at_least: int) -> bool:
        return self.kx_is_nvidia and self.version >= at_least

    @property
    def fp16(self) -> bool:
        return True if self.kx_is_amd else self._sm(53)

    @property
    def bf16(self) -> bool:
        return self.version >= 908 if self.kx_is_amd else self._sm(80)

    @property
    def coop_groups(self) -> bool:
        return True if self.kx_is_amd else self._sm(60)

    @property
    def scoped_atomics(self) -> bool:
        return True if self.kx_is_amd else self._sm(60)

    @property
    def double_atomics(self) -> bool:
        return self.version >= 908 if self.kx_is_amd else self._sm(60)

    @property
    def warp_sync(self) -> bool:
        """__shfl_sync / __ballot_sync and friends."""
        return True if self.kx_is_amd else self._sm(70)

    @property
    def warp_match(self) -> bool:
        """__match_any_sync / __match_all_sync (no AMD equivalent)."""
        return False if self.kx_is_amd else self._sm(70)

    @property
    def warp_reduce(self) -> bool:
        """__reduce_*_sync (sm_80+ only)."""
        return False if self.kx_is_amd else self._sm(80)

    @property
    def wmma(self) -> bool:
        return self.version >= 908 if self.kx_is_amd else self._sm(70)

    @property
    def grid_constant(self) -> bool:
        return False if self.kx_is_amd else self._sm(70)

    @property
    def async_barrier(self) -> bool:
        return False if self.kx_is_amd else self._sm(70)

    @property
    def async_copy(self) -> bool:
        return False if self.kx_is_amd else self._sm(80)

    @property
    def l2_persistence(self) -> bool:
        return False if self.kx_is_amd else self._sm(80)

    @property
    def ldlu(self) -> bool:
        return False if self.kx_is_amd else self._sm(80)

    @property
    def pipeline(self) -> bool:
        return False if self.kx_is_amd else self._sm(80)

    @property
    def mbarrier(self) -> bool:
        return False if self.kx_is_amd else self._sm(80)

    @property
    def dpx(self) -> bool:
        return False if self.kx_is_amd else self._sm(90)

    @property
    def cluster(self) -> bool:
        return False if self.kx_is_amd else self._sm(90)

    @property
    def tma(self) -> bool:
        return False if self.kx_is_amd else self._sm(90)

    @property
    def dependent_launch(self) -> bool:
        return False if self.kx_is_amd else self._sm(90)

    @property
    def atomics_128(self) -> bool:
        return False if self.kx_is_amd else self._sm(90)

    @property
    def fp64_wmma(self) -> bool:
        return self.version >= 908 if self.kx_is_amd else self._sm(80)

    @property
    def tf32_wmma(self) -> bool:
        return False if self.kx_is_amd else self._sm(80)

    @property
    def float_atomics(self) -> bool:
        return True

    @property
    def cache_load_hints(self) -> bool:
        """__ldg / __ldcg / __ldca / __ldcs / __ldlu."""
        return self.kx_is_nvidia

    @property
    def cache_store_hints(self) -> bool:
        """__stcg / __stcs / __stwb / __stwt."""
        return self.kx_is_nvidia

    # -- type availability ---------------------------------------------------

    def scalar_types(self) -> List:
        from generator_ir.ir.typedecl import T
        out = [T.boolean, T.uchar, T.schar, T.short, T.ushort,
               T.i32, T.u32, T.i64, T.u64, T.f32, T.f64]
        if self.fp16:
            out.append(T.f16)
        if self.bf16:
            out.append(T.bf16)
        return out

    def kx_supports_type(self, t) -> bool:
        return getattr(t, "kx_min_arch", 0) <= self.version

    def __repr__(self) -> str:
        return f"Arch({self.name!r})"

    def __str__(self) -> str:
        return self.name


DEFAULT_CUDA_ARCH = Arch("sm_75")
DEFAULT_HIP_ARCH = Arch("gfx908")
