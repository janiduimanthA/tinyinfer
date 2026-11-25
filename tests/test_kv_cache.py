"""KV cache correctness: cached decoding must match full recomputation exactly."""
import numpy as np
import pytest


@pytest.mark.parametrize("i", [0, 1, 2])
def test_prefill_matches_full_forward(model, golden, i):
    ids = golden[f"ids_{i}"]
    cache = model.new_cache()
    got = model.forward_cached(ids, cache)
    expected = model.forward(ids)[-1:]
    np.testing.assert_allclose(got, expected, atol=1e-4)
    assert cache.length == len(ids)


def test_token_by_token_matches_full_forward(model, golden):
    """Feed a sequence ONE token at a time through the cache; every single step must match."""
    ids = golden["greedy_1"]  # prompt + 20 generated tokens
    cache = model.new_cache()
    for t in range(len(ids)):
        got = model.forward_cached(ids[t:t + 1], cache)
        expected = model.forward(ids[:t + 1])[-1:]
        np.testing.assert_allclose(got, expected, atol=1e-3, err_msg=f"mismatch at step {t}")


def test_cached_generation_matches_uncached(model, golden):
    ids = golden["ids_2"].tolist()
    with_cache = model.generate(ids, 15, use_cache=True)
    without_cache = model.generate(ids, 15, use_cache=False)
    assert with_cache == without_cache


def test_cache_overflow_raises(model):
    cache = model.new_cache()
    cache.length = model.block_size  # pretend the cache is already full
    with pytest.raises(ValueError):
        model.forward_cached([0], cache)


def test_cache_memory_size(model):
    """2 (K,V) x 12 layers x 12 heads x 1024 positions x 64 dims x 4 bytes = ~75.5 MB."""
    assert model.new_cache().nbytes == 2 * 12 * 12 * 1024 * 64 * 4