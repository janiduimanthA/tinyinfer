"""
GPT-2 forward pass in pure NumPy.

This is the "golden" Python reference engine: simple, readable, and correct.
Every faster engine we build later is tested against this one.
"""
import numpy as np
import math

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
    att = (q @ k.transpose(0, 2, 1)) / math.sqrt(hs)                    # (n_head, T, T)

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


def mlp(x, p):
    """Feed-forward: expand 768 -> 3072, GELU, project back 3072 -> 768."""
    h = gelu(linear(x, p["mlp.c_fc.weight"], p["mlp.c_fc.bias"]))
    return linear(h, p["mlp.c_proj.weight"], p["mlp.c_proj.bias"])


def block(x, p, n_head):
    """One transformer block: attention then MLP, each with pre-LayerNorm and a residual."""
    x = x + attention(layer_norm(x, p["ln_1.weight"], p["ln_1.bias"]), p, n_head)
    x = x + mlp(layer_norm(x, p["ln_2.weight"], p["ln_2.bias"]), p)
    return x


class GPT2:
    """GPT-2 (124M) inference engine in pure NumPy."""

    def __init__(self, weights_path="weights/gpt2.npz", n_layer=12, n_head=12):
        w = np.load(weights_path)
        self.n_head = n_head

        self.wte = w["transformer.wte.weight"]   # (vocab, C) token embeddings, also the lm_head
        self.wpe = w["transformer.wpe.weight"]   # (block_size, C) position embeddings
        self.ln_f_g = w["transformer.ln_f.weight"]
        self.ln_f_b = w["transformer.ln_f.bias"]
        self.block_size = self.wpe.shape[0]      # 1024

        # One dict per block, with the "transformer.h.{i}." prefix removed,
        # so block i's weights are named like "attn.c_attn.weight"
        self.blocks = []
        for i in range(n_layer):
            prefix = f"transformer.h.{i}."
            self.blocks.append(
                {k[len(prefix):]: w[k] for k in w.files if k.startswith(prefix)}
            )

    def forward(self, ids):
        """ids: T token IDs -> logits of shape (T, vocab)."""
        ids = np.asarray(ids)
        T = len(ids)
        if T > self.block_size:
            raise ValueError(f"sequence length {T} exceeds block size {self.block_size}")

        x = self.wte[ids] + self.wpe[np.arange(T)]   # token + position embeddings
        for p in self.blocks:                         # 12 transformer blocks
            x = block(x, p, self.n_head)
        x = layer_norm(x, self.ln_f_g, self.ln_f_b)   # final LayerNorm
        return x @ self.wte.T                         # lm_head (tied to wte)

    def generate(self, ids, max_new_tokens):
        """Greedy decoding: always pick the highest-scoring next token."""
        ids = list(ids)
        for _ in range(max_new_tokens):
            logits = self.forward(ids[-self.block_size:])
            ids.append(int(np.argmax(logits[-1])))
        return ids