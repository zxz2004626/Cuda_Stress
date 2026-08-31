"""Deterministic RNG: same seeds => same stream; different seeds => different stream."""

from generator.rng import rng_for, mix64, draw_u64


def test_mix64_masks():
    for x in (0, 1, 0xFFFFFFFFFFFFFFFF, 2**200):
        v = mix64(x)
        assert 0 <= v < 2**64


def test_rng_deterministic():
    a1 = [rng_for(100, 200, "integer_chain").getrandbits(64) for _ in range(8)]
    a2 = [rng_for(100, 200, "integer_chain").getrandbits(64) for _ in range(8)]
    assert a1 == a2


def test_rng_differs_on_seed():
    a = [rng_for(100, 200, "integer_chain").getrandbits(64) for _ in range(8)]
    b = [rng_for(101, 200, "integer_chain").getrandbits(64) for _ in range(8)]
    assert a != b


def test_rng_differs_on_strategy():
    a = [rng_for(100, 200, "integer_chain").getrandbits(64) for _ in range(8)]
    b = [rng_for(100, 200, "predicate_pressure").getrandbits(64) for _ in range(8)]
    assert a != b


def test_draw_u64_range():
    r = rng_for(1, 2, "memory_address")
    for _ in range(50):
        assert 0 <= draw_u64(r) < 2**64
