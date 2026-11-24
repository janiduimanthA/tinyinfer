"""Check every NumPy op against PyTorch's version."""
import numpy as np
import torch
import torch.nn.functional as F

from reference.ops import gelu, layer_norm, linear, softmax

rng = np.random.default_rng(0)


def test_linear():
    x = rng.standard_normal((4, 768)).astype(np.float32)
    w = rng.standard_normal((768, 2304)).astype(np.float32)
    b = rng.standard_normal(2304).astype(np.float32)
    expected = (torch.from_numpy(x) @ torch.from_numpy(w) + torch.from_numpy(b)).numpy()
    np.testing.assert_allclose(linear(x, w, b), expected, atol=1e-3)


def test_gelu():
    x = rng.standard_normal((4, 768)).astype(np.float32)
    expected = F.gelu(torch.from_numpy(x), approximate="tanh").numpy()
    np.testing.assert_allclose(gelu(x), expected, atol=1e-5)


def test_softmax():
    x = (rng.standard_normal((4, 10)) * 10).astype(np.float32)
    expected = F.softmax(torch.from_numpy(x), dim=-1).numpy()
    out = softmax(x)
    np.testing.assert_allclose(out, expected, atol=1e-6)
    np.testing.assert_allclose(out.sum(axis=-1), 1.0, atol=1e-6)  # rows sum to 1


def test_softmax_large_values_no_overflow():
    x = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32)
    assert np.isfinite(softmax(x)).all()


def test_layer_norm():
    x = rng.standard_normal((4, 768)).astype(np.float32)
    g = rng.standard_normal(768).astype(np.float32)
    b = rng.standard_normal(768).astype(np.float32)
    expected = F.layer_norm(
        torch.from_numpy(x), (768,), torch.from_numpy(g), torch.from_numpy(b), eps=1e-5
    ).numpy()
    np.testing.assert_allclose(layer_norm(x, g, b), expected, atol=1e-4)

def test_ops_keep_float32():
    """Regression test: a stray np.float64 constant once silently upcast everything."""
    x = rng.standard_normal((4, 8)).astype(np.float32)
    g = np.ones(8, dtype=np.float32)
    b = np.zeros(8, dtype=np.float32)
    assert gelu(x).dtype == np.float32
    assert softmax(x).dtype == np.float32
    assert layer_norm(x, g, b).dtype == np.float32