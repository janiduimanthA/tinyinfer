"""
Benchmark the NumPy reference engine.

Measures the time of one decode step (generating one token) at different context
lengths. Each measurement is warmed up, repeated, and the median is reported,
because single timings are noisy.

Usage:
    python bench.py
"""
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from reference.gpt2_numpy import GPT2

LENGTHS = [8, 32, 64, 128, 256, 512, 1024]
REPEATS = 5
WARMUP = 2
OUT_PATH = Path("benchmarks/reference_numpy.json")


# Each variant takes (model, ids) and returns a zero-argument function that runs
# ONE decode step at context len(ids). Any setup happens outside the timed function.

def full_step(model, ids):
    return lambda: model.forward(ids)


def last_only_step(model, ids):
    return lambda: model.forward(ids, last_only=True)


def kv_cache_step(model, ids):
    """One decode step with the first T-1 tokens already in the cache."""
    cache = model.new_cache()
    model.forward_cached(ids[:-1], cache)  # prefill: setup, not timed
    prefilled = cache.length

    def step():
        cache.length = prefilled           # rewind, so every run does the same work
        model.forward_cached(ids[-1:], cache)

    return step


VARIANTS = {
    "full": full_step,
    "last_only": last_only_step,
    "kv_cache": kv_cache_step,
}


def median_time(fn):
    """Run fn WARMUP times (ignored), then REPEATS times; return the median in ms."""
    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1000


def main():
    model = GPT2("weights/gpt2.npz")
    rng = np.random.default_rng(0)

    print(f"{'context':>8} | " + " | ".join(f"{name + ' (ms)':>16}" for name in VARIANTS))
    print("-" * (11 + 19 * len(VARIANTS)))

    results = []
    for T in LENGTHS:
        ids = rng.integers(0, 50257, size=T)  # random tokens: content doesn't affect speed
        row = {"context": T}
        for name, make_step in VARIANTS.items():
            row[name] = median_time(make_step(model, ids))
        results.append(row)
        print(f"{T:>8} | " + " | ".join(f"{row[name]:>16.1f}" for name in VARIANTS))

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "machine": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "repeats": REPEATS,
        "results": results,
    }, indent=2))
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()