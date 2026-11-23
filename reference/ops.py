"""
Basic building blocks for GPT-2, in pure NumPy.

Every function works on the last axis, so it handles any shape (T, C), (B, T, C), etc.
"""
import numpy as np


def linear(x, w, b):
    """
    Fully-connected layer: x @ w + b.

    GPT-2 stores weights as (in_features, out_features) (Conv1D layout),
    so there is NO transpose here, unlike PyTorch's nn.Linear.
    """
    return x @ w + b


def gelu(x):
    """GELU activation, tanh approximation (the version GPT-2 uses)."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


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