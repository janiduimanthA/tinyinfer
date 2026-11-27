"""
Batching benchmark: how do throughput and latency change with batch size?

1. Decode step: every sequence already has CONTEXT-1 tokens in the KV cache; we time
   one batched decode step (one new token for each of the B sequences).
2. Matmul micro-benchmark: the matmul shapes inside a decode step, alone, at each B,
   to see whether NumPy/BLAS handles "skinny" (few-row) matrices efficiently.

Usage:
    python bench_batch.py
"""
import json
import platform
from pathlib import Path

import numpy as np

from bench import median_time
from reference.gpt2_numpy import GPT2

BATCH_SIZES = [1, 2, 4, 8, 16]
CONTEXT = 64
OUT_PATH = Path("benchmarks/batch.json")

# (name, in_features, out_features) of the matmuls in one decode step
MATMULS = [
    ("attn.c_attn", 768, 2304),
    ("mlp.c_fc", 768, 3072),
    ("mlp.c_proj", 3072, 768),
    ("lm_head", 768, 50257),
]


def decode_step(model, B, rng):
    """One batched decode step at context CONTEXT for B sequences (setup is not timed)."""
    cache = model.new_batch_cache(B, capacity=CONTEXT)
    for b in range(B):
        model.prefill_slot(rng.integers(0, 50257, size=CONTEXT - 1), cache, b)
    prefilled = cache.lengths.copy()
    new_ids = rng.integers(0, 50257, size=B)

    def step():
        cache.lengths = prefilled.copy()  # rewind, so every run does the same work
        model.forward_batch_decode(new_ids, cache)

    return step


def bench_decode(model, rng):
    rows = []
    print(f"{'batch':>5} | {'step (ms)':>9} | {'throughput (tok/s)':>18} | {'vs batch 1':>10}")
    print("-" * 52)
    for B in BATCH_SIZES:
        ms = median_time(decode_step(model, B, rng))
        row = {"batch": B, "step_ms": ms, "throughput_tok_s": B * 1000 / ms}
        rows.append(row)
        speedup = row["throughput_tok_s"] / rows[0]["throughput_tok_s"]
        print(f"{B:>5} | {ms:>9.1f} | {row['throughput_tok_s']:>18.1f} | {speedup:>9.2f}x")
    return rows


def bench_matmuls(rng):
    rows = []
    header = " | ".join(f"B={B:<3}" for B in BATCH_SIZES)
    print(f"\n{'matmul (ms)':12} | {header} | B=2 / B=1")
    print("-" * (17 + 8 * len(BATCH_SIZES) + 12))
    for name, k, n in MATMULS:
        w = rng.standard_normal((k, n)).astype(np.float32)
        row = {"matmul": name, "weight_mb": w.nbytes / 1e6}
        for B in BATCH_SIZES:
            x = rng.standard_normal((B, k)).astype(np.float32)
            row[str(B)] = median_time(lambda: x @ w)
        rows.append(row)
        cells = " | ".join(f"{row[str(B)]:>5.2f}" for B in BATCH_SIZES)
        print(f"{name:12} | {cells} | {row['2'] / row['1']:>8.2f}x")
    return rows


def main():
    rng = np.random.default_rng(0)
    model = GPT2("weights/gpt2.npz")

    print(f"Batched decode step at context {CONTEXT}:\n")
    decode = bench_decode(model, rng)
    matmuls = bench_matmuls(rng)

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "machine": platform.processor(),
        "numpy": np.__version__,
        "context": CONTEXT,
        "decode": decode,
        "matmuls": matmuls,
    }, indent=2))
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()