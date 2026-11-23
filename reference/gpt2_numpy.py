"""
GPT-2 forward pass in pure NumPy.

This is the "golden" Python reference engine: simple, readable, and correct.
Every faster engine we build later is tested against this one.
"""
import numpy as np

from reference.ops import gelu, layer_norm, linear, softmax


def attention(x, p, n_head):
    """
    Causal multi-head self-attention.

    x: (T, C) input, p: weights of one block, returns (T, C).
    """
    T, C = x.shape
    hs = C // n_head  # head size: 768 // 12 = 64

    # 1. Q, K, V in a single matmul (Conv1D layout: no transpose)
    qkv = linear(x, p["attn.c_attn.weight"], p["attn.c_attn.bias"])  # (T, 3C)
    q, k, v = np.split(qkv, 3, axis=-1)                                 # 3 x (T, C)

    # 2. Split into heads: (T, C) -> (T, n_head, hs) -> (n_head, T, hs)
    q = q.reshape(T, n_head, hs).transpose(1, 0, 2)
    k = k.reshape(T, n_head, hs).transpose(1, 0, 2)
    v = v.reshape(T, n_head, hs).transpose(1, 0, 2)

    # 3. Scores: how well each query matches each key, scaled by sqrt(hs)
    att = (q @ k.transpose(0, 2, 1)) / np.sqrt(hs)                      # (n_head, T, T)

    # 4. Causal mask: True above the diagonal = future tokens -> -inf
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    att = np.where(mask, -np.inf, att)

    # 5. Scores -> weights, then take a weighted mix of the values
    att = softmax(att)
    out = att @ v                                                       # (n_head, T, hs)

    # 6. Merge heads back: (n_head, T, hs) -> (T, n_head, hs) -> (T, C)
    out = out.transpose(1, 0, 2).reshape(T, C)

    # 7. Output projection
    return linear(out, p["attn.c_proj.weight"], p["attn.c_proj.bias"])