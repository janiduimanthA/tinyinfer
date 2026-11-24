"""
Turn logits into the next token.

Greedy (temperature=0) always picks the top score. Sampling picks randomly,
weighted by probability, with three knobs:
  temperature: <1 sharper (safer), >1 flatter (more creative)
  top_k:       only consider the k highest-scoring tokens
  top_p:       only consider the smallest set of tokens whose probabilities sum to >= p
"""
import numpy as np

from reference.ops import softmax


def sample_next(logits, temperature=1.0, top_k=None, top_p=None, rng=None):
    """logits: (vocab,) scores for the next token -> chosen token ID (int)."""
    if temperature == 0:
        return int(np.argmax(logits))  # greedy

    rng = rng if rng is not None else np.random.default_rng()
    logits = logits / temperature

    # Top-k: remove everything below the k-th best score
    if top_k is not None:
        k = min(top_k, logits.shape[-1])
        kth_best = np.partition(logits, -k)[-k]
        logits = np.where(logits < kth_best, -np.inf, logits)

    probs = softmax(logits).astype(np.float64)

    # Top-p: keep the most likely tokens until their probabilities add up to top_p
    if top_p is not None:
        order = np.argsort(probs)[::-1]                # most likely first
        cumulative = np.cumsum(probs[order])
        cutoff = np.searchsorted(cumulative, top_p) + 1  # always keep at least one
        keep = np.zeros(probs.shape, dtype=bool)
        keep[order[:cutoff]] = True
        probs = np.where(keep, probs, 0.0)

    probs = probs / probs.sum()  # renormalize so probabilities sum to exactly 1
    return int(rng.choice(len(probs), p=probs))