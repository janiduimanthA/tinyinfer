"""Property tests for sampling: randomness is tested by rules that must always hold."""
import numpy as np

from reference.sampling import sample_next

LOGITS = (np.random.default_rng(0).standard_normal(1000) * 3).astype(np.float32)
TOP = int(np.argmax(LOGITS))


def test_temperature_zero_is_greedy():
    assert sample_next(LOGITS, temperature=0) == TOP


def test_top_k_1_is_greedy():
    for seed in range(20):
        assert sample_next(LOGITS, top_k=1, rng=np.random.default_rng(seed)) == TOP


def test_top_k_only_picks_from_top_k():
    top10 = set(np.argsort(LOGITS)[-10:].tolist())
    rng = np.random.default_rng(1)
    for _ in range(500):
        assert sample_next(LOGITS, top_k=10, rng=rng) in top10


def test_top_p_only_picks_from_nucleus():
    # Probabilities 0.5, 0.3, 0.15, 0.05 -> cumulative 0.5, 0.8, 0.95, 1.0
    logits = np.log(np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32))
    rng = np.random.default_rng(2)

    picks = {sample_next(logits, top_p=0.7, rng=rng) for _ in range(500)}
    assert picks == {0, 1}  # 0.5 < 0.7 <= 0.8 -> keep the top 2

    picks = {sample_next(logits, top_p=0.4, rng=rng) for _ in range(500)}
    assert picks == {0}     # the top token alone already covers 0.4


def test_higher_temperature_is_more_random():
    logits = np.array([3.0, 2.0, 1.0, 0.0], dtype=np.float32)
    rng = np.random.default_rng(3)
    cold = [sample_next(logits, temperature=0.5, rng=rng) for _ in range(1000)]
    hot = [sample_next(logits, temperature=2.0, rng=rng) for _ in range(1000)]
    assert cold.count(0) > hot.count(0)  # cold picks the favourite more often


def test_same_seed_same_sequence():
    def draw(seed):
        rng = np.random.default_rng(seed)
        return [sample_next(LOGITS, rng=rng) for _ in range(20)]

    assert draw(42) == draw(42)  # reproducible
    assert draw(42) != draw(43)  # but different seeds differ