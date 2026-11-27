"""Batching correctness: decoding sequences together must match decoding them one by one."""
import numpy as np
import pytest


def mixed_prompts(golden):
    """Three prompts of different lengths (5, 25 and 2 tokens) to exercise the masks."""
    return [golden["ids_0"].tolist(), golden["greedy_1"].tolist(), golden["ids_2"][:2].tolist()]


def test_batch_decode_logits_match_single(model, golden):
    """One batched decode step vs a separate full forward for each sequence."""
    prompts = mixed_prompts(golden)
    cache = model.new_batch_cache(len(prompts), capacity=40)
    for b, p in enumerate(prompts):
        model.prefill_slot(p, cache, b)

    new = [11, 12, 13]
    got = model.forward_batch_decode(new, cache)
    for b, p in enumerate(prompts):
        expected = model.forward(p + [new[b]])[-1]
        np.testing.assert_allclose(got[b], expected, atol=1e-3, err_msg=f"sequence {b}")
    assert cache.lengths.tolist() == [len(p) + 1 for p in prompts]


def test_batch_generation_matches_one_by_one(model, golden):
    prompts = mixed_prompts(golden)
    batched = model.generate_batch(prompts, 10)
    one_by_one = [model.generate(p, 10) for p in prompts]
    assert batched == one_by_one


def test_batch_matches_golden(model, golden):
    """Batched greedy generation reproduces Hugging Face GPT-2 for all three prompts."""
    prompts = [golden[f"ids_{i}"].tolist() for i in range(3)]
    expected = [golden[f"greedy_{i}"].tolist() for i in range(3)]
    assert model.generate_batch(prompts, 20) == expected


def test_batch_of_one_matches_generate(model, golden):
    p = golden["ids_1"].tolist()
    assert model.generate_batch([p], 10) == [model.generate(p, 10)]


def test_batch_with_int8(model_int8, golden):
    prompts = mixed_prompts(golden)
    assert model_int8.generate_batch(prompts, 8) == [model_int8.generate(p, 8) for p in prompts]


def test_batch_capacity_overflow_raises(model):
    with pytest.raises(ValueError):
        model.generate_batch([[0] * 1000], 100)  # 1000 + 100 > 1024