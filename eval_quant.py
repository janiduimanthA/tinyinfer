"""
Measure what INT8 quantization costs, against the float32 model.

On real text (Tiny Shakespeare, public domain), at every position:
  - logit difference: how far each score moves
  - top-1 agreement:  how often both models pick the same most-likely next token
  - perplexity:       the standard language-model quality metric (lower is better)
Plus memory and decode speed.

Usage:
    python eval_quant.py
"""
import json
import urllib.request
from pathlib import Path

import numpy as np
import tiktoken

from bench import kv_cache_step, median_time
from reference.gpt2_numpy import GPT2
from reference.metrics import perplexity, token_nll

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = Path("data/tinyshakespeare.txt")
OUT_PATH = Path("benchmarks/quant_quality.json")
WINDOW = 512     # tokens per forward pass
N_WINDOWS = 4    # 4 x 512 = 2048 tokens evaluated


def load_tokens():
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(exist_ok=True)
        print(f"Downloading eval text from {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    text = DATA_PATH.read_text(encoding="utf-8")
    return tiktoken.get_encoding("gpt2").encode(text)


def main():
    tokens = load_tokens()
    windows = [tokens[i * WINDOW:(i + 1) * WINDOW] for i in range(N_WINDOWS)]

    print("Loading float32 and INT8 models...")
    models = {"fp32": GPT2("weights/gpt2.npz"), "int8": GPT2("weights/gpt2.npz", int8=True)}

    nll = {name: [] for name in models}
    diff_sum, diff_count, diff_max = 0.0, 0, 0.0
    agree = []

    for w, ids in enumerate(windows, start=1):
        print(f"  window {w}/{N_WINDOWS} ({len(ids)} tokens)...")
        logits = {name: m.forward(ids) for name, m in models.items()}
        for name, lg in logits.items():
            nll[name].append(token_nll(lg, ids))
        d = np.abs(logits["int8"] - logits["fp32"])
        diff_sum += float(d.sum())
        diff_count += d.size
        diff_max = max(diff_max, float(d.max()))
        agree.append(logits["int8"].argmax(-1) == logits["fp32"].argmax(-1))

    ppl = {name: perplexity(np.concatenate(v)) for name, v in nll.items()}

    print("Timing one decode step at context 128 (KV cache)...")
    speed_ids = np.asarray(tokens[:128])
    step_ms = {name: median_time(kv_cache_step(m, speed_ids)) for name, m in models.items()}

    r = {
        "tokens_evaluated": sum(len(ids) - 1 for ids in windows),
        "perplexity_fp32": ppl["fp32"],
        "perplexity_int8": ppl["int8"],
        "perplexity_increase_pct": 100 * (ppl["int8"] / ppl["fp32"] - 1),
        "top1_agreement_pct": 100 * float(np.concatenate(agree).mean()),
        "logit_diff_mean": diff_sum / diff_count,
        "logit_diff_max": diff_max,
        "weights_mb_fp32": models["fp32"].nbytes / 1e6,
        "weights_mb_int8": models["int8"].nbytes / 1e6,
        "decode_ms_fp32": step_ms["fp32"],
        "decode_ms_int8": step_ms["int8"],
    }

    print(f"\n{'':24} {'float32':>10} {'int8':>10}")
    print(f"{'weights (MB)':24} {r['weights_mb_fp32']:>10.0f} {r['weights_mb_int8']:>10.0f}")
    print(f"{'perplexity':24} {r['perplexity_fp32']:>10.3f} {r['perplexity_int8']:>10.3f}")
    print(f"{'decode step @128 (ms)':24} {r['decode_ms_fp32']:>10.1f} {r['decode_ms_int8']:>10.1f}")
    print(f"\nperplexity increase:   {r['perplexity_increase_pct']:+.2f}%")
    print(f"top-1 agreement:       {r['top1_agreement_pct']:.2f}%  ({r['tokens_evaluated']} tokens)")
    print(f"logit diff:            mean {r['logit_diff_mean']:.4f}, max {r['logit_diff_max']:.4f}")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(r, indent=2))
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()