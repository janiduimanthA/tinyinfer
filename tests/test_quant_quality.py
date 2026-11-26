"""
Quality regression tests for INT8. If a future change silently degrades quantized
quality, these fail. Thresholds come from measured results (eval_quant.py) plus a
safety margin, since these tests use a smaller text sample.
"""
from pathlib import Path

import pytest
import tiktoken

from reference.metrics import perplexity, token_nll

DATA_PATH = Path("data/tinyshakespeare.txt")
N_TOKENS = 256

# Measured with eval_quant.py on 2048 tokens: perplexity +X.XX%, top-1 agreement XX.X%
# Measured with eval_quant.py on 2048 tokens (INT8 blocks, float32 embeddings):
# perplexity -1.06%, top-1 agreement 97.46%. Margins allow for the smaller 256-token sample.
MAX_PERPLEXITY_INCREASE_PCT = 2.0
MIN_TOP1_AGREEMENT = 0.92


@pytest.fixture(scope="module")
def eval_ids():
    if not DATA_PATH.exists():
        pytest.skip("run eval_quant.py once to download the eval text")
    text = DATA_PATH.read_text(encoding="utf-8")
    return tiktoken.get_encoding("gpt2").encode(text)[:N_TOKENS]


@pytest.fixture(scope="module")
def logits_pair(model, model_int8, eval_ids):
    return model.forward(eval_ids), model_int8.forward(eval_ids)


def test_int8_perplexity_close_to_fp32(logits_pair, eval_ids):
    fp32, int8 = logits_pair
    p32 = perplexity(token_nll(fp32, eval_ids))
    p8 = perplexity(token_nll(int8, eval_ids))
    increase = 100 * (p8 / p32 - 1)
    print(f"\n  perplexity fp32 {p32:.3f}, int8 {p8:.3f} ({increase:+.2f}%)")
    assert increase < MAX_PERPLEXITY_INCREASE_PCT


def test_int8_top1_agreement(logits_pair):
    fp32, int8 = logits_pair
    agreement = float((fp32.argmax(-1) == int8.argmax(-1)).mean())
    print(f"\n  top-1 agreement {100 * agreement:.1f}%")
    assert agreement >= MIN_TOP1_AGREEMENT