"""Deterministic RNG plumbing for testcase generation.

Every random decision in a strategy is driven by a `random.Random` seeded from a
derivation of (config_seed, test_seed, strategy).  Regenerating a testcase byte-for-byte
only needs those two integers plus the strategy params.
"""

from __future__ import annotations

import random


def mix64(x: int) -> int:
    """SplitMix64 finalizer -- stable, dependency-free integer mixing."""
    x &= 0xFFFFFFFFFFFFFFFF
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


def rng_for(config_seed: int, test_seed: int, strategy: str) -> random.Random:
    """Build the per-testcase RNG.  Pure function of its arguments."""
    key = mix64(config_seed * 0x9E3779B97F4A7C15 ^ mix64(test_seed))
    key ^= mix64(0xC0FFEE ^ sum((ord(c) * 31) ** 3 for c in strategy))
    r = random.Random()
    r.seed(key)
    return r


def draw_u64(rng: random.Random) -> int:
    return rng.getrandbits(64)


def draw_u32(rng: random.Random) -> int:
    return rng.getrandbits(32)


def draw_int(rng: random.Random, lo: int, hi: int) -> int:
    """Inclusive integer in [lo, hi]."""
    return rng.randint(lo, hi)
