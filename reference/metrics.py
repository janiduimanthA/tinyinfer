"""Language-model quality metrics."""
import numpy as np

from reference.ops import log_softmax


def token_nll(logits, ids):
    """
    Negative log-likelihood ("surprise") of each actual next token.

    logits: (T, vocab) from a forward pass over ids (T,). Position t predicts token t+1,
    so we score predictions 0..T-2 against targets 1..T-1. Returns (T-1,).
    """
    logprobs = log_softmax(logits[:-1])
    targets = np.asarray(ids[1:])
    return -logprobs[np.arange(len(targets)), targets]


def perplexity(nll):
    """exp(mean surprise). Lower is better. ~N means 'as unsure as N equally likely choices'."""
    return float(np.exp(np.mean(nll)))