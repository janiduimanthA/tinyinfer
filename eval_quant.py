"""
Measure what INT8 quantization costs, against the float32 model.

On real text (Tiny Shakespeare, public domain), at every position:
  - KL divergence:    how different the next-token probability distributions are
                      (the right metric: ignores harmless shifts of all logits together)
  - top-1 agreement:  how often both models pick the same most-likely next token
  - perplexity:       the standard language-model quality metric (lower is better)
  - logit difference: raw score change (misleading on its own: softmax ignores uniform shifts)

Usage:
    python eval_quant.py            # default INT8 (blocks only): quality, memory, speed
    python eval_quant.py --ablate   # quantize blocks / embeddings separately: which part hurts?
"""
import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import tiktoken

from bench import kv_cache_step, median_time
from reference.gpt2_numpy import GPT2
from reference.metrics import perplexity, token_nll
from reference.ops import log_softmax

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = Path("data/tinyshakespeare.txt")
WEIGHTS = "weights/gpt2.npz"
WINDOW = 512     # tokens per forward pass
N_WINDOWS = 4    # 4 x 512 = 2048 tokens evaluated

ABLATIONS = {
    "blocks_only":     dict(int8_blocks=True,  int8_embeddings=False),
    "embeddings_only": dict(int8_blocks=False, int8_embeddings=True),
    "all":             dict(int8_blocks=True,  int8_embeddings=True),
}


def load_tokens():
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(exist_ok=True)
        print(f"Downloading eval text from {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    text = DATA_PATH.read_text(encoding="utf-8")
    return tiktoken.get_encoding("gpt2").encode(text)


def compare(ref, quant, windows):
    """Quality of `quant` relative to `ref` on the same token windows."""
    nll_ref, nll_q, agree, kl = [], [], [], []
    diff_sum, diff_count = 0.0, 0
    worst = {"diff": 0.0, "window": None, "position": None}

    for w, ids in enumerate(windows):
        a, b = ref.forward(ids), quant.forward(ids)
        nll_ref.append(token_nll(a, ids))
        nll_q.append(token_nll(b, ids))
        agree.append(a.argmax(-1) == b.argmax(-1))

        # KL(ref || quant) at each position: sum over vocab of p * (log p - log q)
        la, lb = log_softmax(a), log_softmax(b)
        kl.append(np.sum(np.exp(la) * (la - lb), axis=-1))

        d = np.abs(b - a)
        diff_sum += float(d.sum())
        diff_count += d.size
        if d.max() > worst["diff"]:
            pos = int(np.unravel_index(d.argmax(), d.shape)[0])
            worst = {"diff": float(d.max()), "window": w, "position": pos}

    p_ref = perplexity(np.concatenate(nll_ref))
    p_q = perplexity(np.concatenate(nll_q))
    return {
        "weights_mb": quant.nbytes / 1e6,
        "perplexity_fp32": p_ref,
        "perplexity_quant": p_q,
        "perplexity_increase_pct": 100 * (p_q / p_ref - 1),
        "top1_agreement_pct": 100 * float(np.concatenate(agree).mean()),
        "kl_mean": float(np.concatenate(kl).mean()),
        "logit_diff_mean": diff_sum / diff_count,
        "logit_diff_max": worst["diff"],
        "worst_window": worst["window"],
        "worst_position": worst["position"],
    }


def run_full(fp32, windows, tokens):
    int8 = GPT2(WEIGHTS, int8=True)
    r = compare(fp32, int8, windows)

    print("Timing one decode step at context 128 (KV cache)...")
    speed_ids = np.asarray(tokens[:128])
    r["decode_ms_fp32"] = median_time(kv_cache_step(fp32, speed_ids))
    r["decode_ms_int8"] = median_time(kv_cache_step(int8, speed_ids))
    r["weights_mb_fp32"] = fp32.nbytes / 1e6

    print(f"\n{'':24} {'float32':>10} {'int8':>10}")
    print(f"{'weights (MB)':24} {r['weights_mb_fp32']:>10.0f} {r['weights_mb']:>10.0f}")
    print(f"{'perplexity':24} {r['perplexity_fp32']:>10.3f} {r['perplexity_quant']:>10.3f}")
    print(f"{'decode step @128 (ms)':24} {r['decode_ms_fp32']:>10.1f} {r['decode_ms_int8']:>10.1f}")
    print(f"\nperplexity increase:   {r['perplexity_increase_pct']:+.2f}%")
    print(f"top-1 agreement:       {r['top1_agreement_pct']:.2f}%")
    print(f"KL divergence (mean):  {r['kl_mean']:.5f}")
    print(f"logit diff:            mean {r['logit_diff_mean']:.4f}, max {r['logit_diff_max']:.4f}")
    return r


def run_ablation(fp32, windows):
    results = {}
    for name, flags in ABLATIONS.items():
        print(f"  evaluating {name} ...")
        results[name] = compare(fp32, GPT2(WEIGHTS, **flags), windows)

    print(f"\n{'config':16} | {'MB':>5} | {'ppl +%':>7} | {'top-1 %':>7} | {'KL':>8} | {'max diff':>8}")
    print("-" * 66)
    for name, r in results.items():
        print(f"{name:16} | {r['weights_mb']:>5.0f} | {r['perplexity_increase_pct']:>+6.2f}% | "
              f"{r['top1_agreement_pct']:>6.2f}% | {r['kl_mean']:>8.5f} | {r['logit_diff_max']:>8.3f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablate", action="store_true", help="quantize parts separately")
    args = parser.parse_args()

    tokens = load_tokens()
    windows = [tokens[i * WINDOW:(i + 1) * WINDOW] for i in range(N_WINDOWS)]
    print(f"Evaluating on {sum(len(w) - 1 for w in windows)} tokens...")
    fp32 = GPT2(WEIGHTS)

    if args.ablate:
        results = run_ablation(fp32, windows)
        out = Path("benchmarks/quant_ablation.json")
    else:
        results = run_full(fp32, windows, tokens)
        out = Path("benchmarks/quant_quality.json")

    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()