"""Shared helpers for strategy implementations: input construction and u64 packing."""

from __future__ import annotations

import struct
from typing import List

MASK64 = (1 << 64) - 1
MASK32 = (1 << 32) - 1

# Boundary values injected into inputs so integer edge cases are actually exercised.
BOUNDARIES = [
    0,
    1,
    0x7FFFFFFF,
    0x80000000,
    0xFFFFFFFF,
    0x0000000100000000,
    0xFFFFFFFF00000000,
    0x7FFFFFFFFFFFFFFF,
    0x8000000000000000,
    0xFFFFFFFFFFFFFFFF,
    0xAAAAAAAAAAAAAAAA,
    0x5555555555555555,
]


def make_input_words(rng, input_words: int, boundary_count: int = 8) -> List[int]:
    """Deterministic input words with boundary constants sprinkled at the front."""
    words: List[int] = []
    for i in range(input_words):
        if i < boundary_count:
            words.append(BOUNDARIES[i % len(BOUNDARIES)] if i % 2 == 0 else rng.getrandbits(64))
        else:
            words.append(rng.getrandbits(64))
    return words


def pack_words(words: List[int]) -> bytes:
    return b"".join(struct.pack("<Q", w & MASK64) for w in words)


def unpack_words(data: bytes) -> List[int]:
    n = len(data) // 8
    return list(struct.unpack(f"<{n}Q", data))


def c_fmt_u64(x: int) -> str:
    """Format a u64 constant as a C++ literal."""
    x &= MASK64
    return f"{x}ULL"


def c_fmt_u32(x: int) -> str:
    return f"{x & MASK32}u"


def rotl64(x: int, r: int) -> int:
    x &= MASK64
    r &= 63
    if r == 0:
        return x
    return ((x << r) | (x >> (64 - r))) & MASK64


def rotr64(x: int, r: int) -> int:
    x &= MASK64
    r &= 63
    if r == 0:
        return x
    return ((x >> r) | (x << (64 - r))) & MASK64
