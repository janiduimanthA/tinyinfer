"""
GPT-2 forward pass in pure NumPy.

This is the "golden" Python reference engine: simple, readable, and correct.
Every faster engine we build later is tested against this one.
"""
import math

import numpy as np

from reference.ops import gelu, layer_norm, linear, softmax
from reference.quant import QTensor, quantize
from reference.sampling import sample_next

# Matmul weights inside each block that get INT8-quantized (biases and LayerNorms stay float32)
QUANT_KEYS = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")


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


class KVCache:
    """
    Stores K and V for every layer, so past tokens are never recomputed.

    Pre-allocated to the full block size (like real engines do), with `length`
    tracking how many positions are filled. GPT-2 at 1024 tokens: ~75 MB.
    """

    def __init__(self, n_layer, n_head, block_size, head_size):
        shape = (n_layer, n_head, block_size, head_size)
        self.k = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)
        self.length = 0  # number of positions already stored

    @property
    def nbytes(self):
        return self.k.nbytes + self.v.nbytes


def attention_cached(x, p, n_head, k_cache, v_cache, start):
    """
    Causal self-attention for NEW tokens only, reading past K/V from the cache.

    x: (T, C) new tokens at absolute positions start .. start+T-1.
    k_cache, v_cache: this layer's cache, shape (n_head, block_size, hs). Updated in place.
    """
    T, C = x.shape
    hs = C // n_head
    end = start + T

    # 1. Q, K, V for the new tokens only
    qkv = linear(x, p["attn.c_attn.weight"], p["attn.c_attn.bias"])
    q, k, v = np.split(qkv, 3, axis=-1)
    q = q.reshape(T, n_head, hs).transpose(1, 0, 2)  # (n_head, T, hs)
    k = k.reshape(T, n_head, hs).transpose(1, 0, 2)
    v = v.reshape(T, n_head, hs).transpose(1, 0, 2)

    # 2. Save the new K, V into the cache, then read back ALL keys/values so far
    k_cache[:, start:end] = k
    v_cache[:, start:end] = v
    K = k_cache[:, :end]  # (n_head, end, hs)
    V = v_cache[:, :end]

    # 3. New queries vs all keys
    att = (q @ K.transpose(0, 2, 1)) / math.sqrt(hs)  # (n_head, T, end)

    # 4. Causal mask: new token i sits at position start+i and may see keys 0 .. start+i
    mask = np.arange(end)[None, :] > (start + np.arange(T))[:, None]  # (T, end)
    att = np.where(mask, -np.inf, att)

    # 5. Weighted mix of values, merge heads, output projection
    att = softmax(att)
    out = (att @ V).transpose(1, 0, 2).reshape(T, C)
    return linear(out, p["attn.c_proj.weight"], p["attn.c_proj.bias"])


def block_cached(x, p, n_head, k_cache, v_cache, start):
    """One transformer block for new tokens, using this layer's KV cache."""
    a = layer_norm(x, p["ln_1.weight"], p["ln_1.bias"])
    x = x + attention_cached(a, p, n_head, k_cache, v_cache, start)
    x = x + mlp(layer_norm(x, p["ln_2.weight"], p["ln_2.bias"]), p)
    return x


class GPT2:
    """GPT-2 (124M) inference engine in pure NumPy."""

    def __init__(self, weights_path="weights/gpt2.npz", n_layer=12, n_head=12, int8=False):
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

        if int8:
            self.wte = quantize(self.wte, axis=0)      # one scale per token row
            for p in self.blocks:
                for key in QUANT_KEYS:
                    p[key] = quantize(p[key], axis=1)  # one scale per output column

    def _embed(self, ids):
        """Token embedding lookup, from float32 or INT8 (per-row scale) weights."""
        if isinstance(self.wte, QTensor):
            return self.wte.q[ids].astype(np.float32) * self.wte.scale[ids][:, None]
        return self.wte[ids]

    def _lm_head(self, x):
        """Logits via the tied embedding table: x @ wte.T."""
        if isinstance(self.wte, QTensor):
            return (x @ self.wte.q.T) * self.wte.scale  # per-row scale = per-vocab-column here
        return x @ self.wte.T

    @property
    def nbytes(self):
        """Bytes used by all weights in memory."""
        total = self.wte.nbytes + self.wpe.nbytes + self.ln_f_g.nbytes + self.ln_f_b.nbytes
        for p in self.blocks:
            total += sum(w.nbytes for w in p.values())
        return total

    def forward(self, ids, last_only=False):
        """
        ids: T token IDs -> logits of shape (T, vocab), or (1, vocab) if last_only.

        During generation only the last position's logits are needed. ln_f and lm_head
        treat each position independently, so with last_only we drop the other positions
        right after the blocks and skip most of the expensive (768 x 50257) lm_head matmul.
        Attention mixes positions, so the blocks must still see the whole sequence.
        """
        ids = np.asarray(ids)
        T = len(ids)
        if T > self.block_size:
            raise ValueError(f"sequence length {T} exceeds block size {self.block_size}")

        x = self._embed(ids) + self.wpe[np.arange(T)]   # token + position embeddings
        for p in self.blocks:                            # 12 transformer blocks
            x = block(x, p, self.n_head)
        if last_only:
            x = x[-1:]                                   # keep only the last position: (1, C)
        x = layer_norm(x, self.ln_f_g, self.ln_f_b)      # final LayerNorm
        return self._lm_head(x)                          # lm_head (tied to wte)

    def new_cache(self):
        """An empty KV cache sized for this model."""
        head_size = self.wte.shape[1] // self.n_head
        return KVCache(len(self.blocks), self.n_head, self.block_size, head_size)

    def forward_cached(self, ids, cache):
        """
        Process only the NEW tokens `ids`, reading and updating the KV cache.

        Prefill: pass the whole prompt once. Decode: pass one token at a time.
        Returns logits for the last position only: (1, vocab).
        """
        ids = np.asarray(ids)
        T = len(ids)
        start = cache.length
        if start + T > self.block_size:
            raise ValueError(f"KV cache full: {start} + {T} tokens exceeds block size {self.block_size}")

        # Positions continue from where the cache left off (NOT from 0!)
        x = self._embed(ids) + self.wpe[np.arange(start, start + T)]
        for layer, p in enumerate(self.blocks):
            x = block_cached(x, p, self.n_head, cache.k[layer], cache.v[layer], start)
        cache.length += T

        x = layer_norm(x[-1:], self.ln_f_g, self.ln_f_b)  # last position only
        return self._lm_head(x)                           # lm_head (tied to wte)

    def stream(self, ids, max_new_tokens, temperature=0.0, top_k=None, top_p=None,
               seed=None, use_cache=True):
        """
        Yield new token IDs one at a time. temperature=0 means greedy.

        With use_cache (default), the prompt is processed once (prefill), then each
        step processes only the single newest token (decode).
        """
        ids = list(ids)
        rng = np.random.default_rng(seed)
        cache = self.new_cache() if use_cache else None
        pending = list(ids)  # tokens the model hasn't processed yet

        for _ in range(max_new_tokens):
            if use_cache:
                logits = self.forward_cached(pending, cache)  # prefill first, then 1 token
            else:
                logits = self.forward(ids[-self.block_size:], last_only=True)
            next_id = sample_next(logits[-1], temperature, top_k, top_p, rng)
            ids.append(next_id)
            pending = [next_id]
            yield next_id

    def generate(self, ids, max_new_tokens, **sampling):
        """Return prompt + generated token IDs (greedy unless sampling options are given)."""
        return list(ids) + list(self.stream(ids, max_new_tokens, **sampling))