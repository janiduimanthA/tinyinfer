"""INT8 quantization: unit tests on small arrays, plus basic checks on the real model."""
import numpy as np

from reference.ops import linear
from reference.quant import quantize

rng = np.random.default_rng(0)


def test_quantized_values_are_int8_in_range():
    w = rng.standard_normal((64, 32)).astype(np.float32)
    qt = quantize(w, axis=1)
    assert qt.q.dtype == np.int8
    assert qt.q.min() >= -127 and qt.q.max() <= 127
    assert qt.scale.shape == (32,)  # one scale per column


def test_round_trip_error_is_at_most_half_a_step():
    """Rounding to the nearest INT8 step can't be off by more than half a step (scale / 2)."""
    w = rng.standard_normal((64, 32)).astype(np.float32)
    qt = quantize(w, axis=1)
    err = np.abs(qt.dequantize() - w)
    assert (err <= qt.scale[None, :] / 2 + 1e-6).all()


def test_per_channel_isolates_outliers():
    """An outlier in one column must not wipe out the small weights in other columns."""
    w = (rng.standard_normal((64, 4)) * 0.01).astype(np.float32)
    w[0, 0] = 5.0  # huge outlier in column 0
    deq = quantize(w, axis=1).dequantize()
    np.testing.assert_allclose(deq[:, 1:], w[:, 1:], atol=2e-4)  # other columns: still accurate


def test_row_axis_for_embeddings():
    w = rng.standard_normal((10, 8)).astype(np.float32)
    qt = quantize(w, axis=0)
    assert qt.scale.shape == (10,)  # one scale per row
    err = np.abs(qt.dequantize() - w)
    assert (err <= qt.scale[:, None] / 2 + 1e-6).all()


def test_all_zero_channel_is_safe():
    w = np.zeros((8, 4), dtype=np.float32)
    qt = quantize(w, axis=1)
    assert np.isfinite(qt.dequantize()).all()
    assert (qt.dequantize() == 0).all()


def test_linear_with_qtensor_matches_dequantized():
    """The (x @ q) * scale trick must equal x @ dequantize(q) + b."""
    x = rng.standard_normal((5, 64)).astype(np.float32)
    w = rng.standard_normal((64, 32)).astype(np.float32)
    b = rng.standard_normal(32).astype(np.float32)
    qt = quantize(w, axis=1)
    np.testing.assert_allclose(linear(x, qt, b), x @ qt.dequantize() + b, atol=1e-4)


def test_int8_model_is_about_2x_smaller(model, model_int8):
    """Blocks are INT8; embeddings (tied lm_head) stay float32 for quality: ~2x overall."""
    ratio = model.nbytes / model_int8.nbytes
    print(f"\n  float32: {model.nbytes / 1e6:.0f} MB, int8: {model_int8.nbytes / 1e6:.0f} MB ({ratio:.2f}x)")
    assert ratio > 1.9


def test_int8_model_runs(model_int8, golden):
    logits = model_int8.forward(golden["ids_0"])
    assert logits.shape == (len(golden["ids_0"]), 50257)
    assert logits.dtype == np.float32
    assert np.isfinite(logits).all()