"""
Weight-only INT8 quantization: symmetric, one scale per channel.

    w ≈ q * scale,   q = round(w / scale) in [-127, 127],   scale = max|w| / 127

Each channel (a column of a matmul weight, or a row of the embedding table) gets its
own scale, so one outlier weight only hurts its own channel.
"""
import numpy as np


class QTensor:
    """An INT8 weight with one float32 scale per channel along `axis`."""

    def __init__(self, q, scale, axis):
        self.q = q          # int8, same shape as the original weight
        self.scale = scale  # float32, one value per channel along `axis`
        self.axis = axis

    @property
    def shape(self):
        return self.q.shape

    @property
    def nbytes(self):
        return self.q.nbytes + self.scale.nbytes

    def dequantize(self):
        """Back to float32 (approximately the original weight)."""
        shape = [1] * self.q.ndim
        shape[self.axis] = -1
        return self.q.astype(np.float32) * self.scale.reshape(shape)


def quantize(w, axis):
    """
    Quantize a 2D float32 weight to INT8 with one scale per channel along `axis`.

    axis=1: one scale per column (matmul weights stored as (in, out))
    axis=0: one scale per row    (embedding table, one row per token)
    """
    reduce_axis = 1 - axis
    max_abs = np.max(np.abs(w), axis=reduce_axis, keepdims=True)
    scale = (max_abs / 127.0).astype(np.float32)
    scale = np.where(scale == 0, 1.0, scale).astype(np.float32)  # all-zero channel: avoid 0/0
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return QTensor(q, scale.squeeze(reduce_axis), axis)