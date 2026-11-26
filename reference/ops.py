"""
Basic building blocks for GPT-2, in pure NumPy.

Every function works on the last axis, so it handles any shape (T, C), (B, T, C), etc.
"""
import numpy as np
import math
from reference.quant import QTensor


def linear(x, w, b):
    """
    Fully-connected layer: x @ w + b.

    GPT-2 stores weights as (in_features, out_features) (Conv1D layout),
    so there is NO transpose here, unlike PyTorch's nn.Linear.

    If w is an INT8 QTensor with one scale per output column, then
    x @ (q * scale) == (x @ q) * scale, so we scale the OUTPUT (one multiply
    per output) instead of dequantizing every weight first.
    """
    if isinstance(w, QTensor):
        return (x @ w.q) * w.scale + b
    return x @ w + b


# Plain Python float (not np.float64), so float32 arrays stay float32 under NumPy 2
_GELU_C = math.sqrt(2.0 / math.pi)


def gelu(x):
    """GELU activation, tanh approximation (the version GPT-2 uses)."""
    return 0.5 * x * (1.0 + np.tanh(_GELU_C * (x + 0.044715 * x**3)))


def softmax(x):
    """
    Turn scores into probabilities along the last axis.

    Subtracting the max first prevents overflow (exp(1000) = inf)
    and does not change the result.
    """
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def layer_norm(x, g, b, eps=1e-5):
    """Normalize to mean 0 / variance 1, then scale by g and shift by b."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(var + eps) + b

def log_softmax(x):
    """log(softmax(x)) along the last axis, computed stably: x - max - log(sum(exp(x - max)))."""
    x = x - np.max(x, axis=-1, keepdims=True)
    return x - np.log(np.sum(np.exp(x), axis=-1, keepdims=True))