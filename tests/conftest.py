"""Shared pytest fixtures: loaded once per test session, used by every test file."""
from pathlib import Path

import numpy as np
import pytest

from reference.gpt2_numpy import GPT2

WEIGHTS = Path("weights/gpt2.npz")
GOLDEN = Path("tests/golden/gpt2_golden.npz")


@pytest.fixture(scope="session")
def model():
    # Weights aren't in Git (too big): skip instead of failing if they're missing
    if not WEIGHTS.exists():
        pytest.skip("run scripts/export_weights.py first")
    return GPT2(WEIGHTS)


@pytest.fixture(scope="session")
def golden():
    return np.load(GOLDEN)

@pytest.fixture(scope="session")
def model_int8():
    if not WEIGHTS.exists():
        pytest.skip("run scripts/export_weights.py first")
    return GPT2(WEIGHTS, int8=True)