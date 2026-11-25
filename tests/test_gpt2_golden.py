"""The big one: does our NumPy engine reproduce Hugging Face GPT-2?"""
from pathlib import Path

import numpy as np
import pytest

from reference.gpt2_numpy import GPT2

WEIGHTS = Path("weights/gpt2.npz")
GOLDEN = Path("tests/golden/gpt2_golden.npz")

# Weights aren't in Git (too big), so skip instead of failing if they're missing
pytestmark = pytest.mark.skipif(
    not WEIGHTS.exists(), reason="run scripts/export_weights.py first"
)


@pytest.fixture(scope="module")
def model():
    return GPT2(WEIGHTS)  # load once, reuse for every test in this file


@pytest.fixture(scope="module")
def golden():
    return np.load(GOLDEN)


@pytest.mark.parametrize("i", [0, 1, 2])
def test_logits_match_golden(model, golden, i):
    expected = golden[f"logits_{i}"]
    got = model.forward(golden[f"ids_{i}"])

    assert got.shape == expected.shape
    assert got.dtype == np.float32                         # no silent float64 upcasting
    max_diff = np.abs(got - expected).max()
    print(f"\n  prompt {i}: max |diff| = {max_diff:.2e}")
    assert max_diff < 1e-2                                 # every score nearly identical
    assert (got.argmax(-1) == expected.argmax(-1)).all()   # same top pick everywhere


@pytest.mark.parametrize("i", [0, 1, 2])
def test_greedy_generation_matches_golden(model, golden, i):
    ids = golden[f"ids_{i}"].tolist()
    expected = golden[f"greedy_{i}"].tolist()
    got = model.generate(ids, max_new_tokens=len(expected) - len(ids))
    assert got == expected                                 # exact same 20 tokens

@pytest.mark.parametrize("i", [0, 1, 2])
def test_last_only_matches_full(model, golden, i):
    """The last_only optimization must give the same last-position logits as a full forward."""
    ids = golden[f"ids_{i}"]
    full = model.forward(ids)
    last = model.forward(ids, last_only=True)
    assert last.shape == (1, full.shape[1])
    np.testing.assert_allclose(last[0], full[-1], atol=1e-4)