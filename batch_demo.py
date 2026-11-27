"""
Generate for several prompts at once, and compare with one-at-a-time.

Usage:
    python batch_demo.py
"""
import time

import tiktoken

from reference.gpt2_numpy import GPT2

PROMPTS = [
    "Sri Lanka is famous for",
    "The best way to learn programming is",
    "Once upon a time, in a small village,",
    "The future of artificial intelligence",
]
N_NEW = 30


def main():
    enc = tiktoken.get_encoding("gpt2")
    model = GPT2("weights/gpt2.npz")
    prompts = [enc.encode(p) for p in PROMPTS]

    t0 = time.perf_counter()
    one_by_one = [model.generate(p, N_NEW) for p in prompts]
    t_seq = time.perf_counter() - t0

    t0 = time.perf_counter()
    batched = model.generate_batch(prompts, N_NEW)
    t_batch = time.perf_counter() - t0

    for out in batched:
        print("•", enc.decode(out).replace("\n", " "), "\n")

    total = N_NEW * len(prompts)
    print(f"one at a time: {t_seq:.2f}s ({total / t_seq:.1f} tokens/s)")
    print(f"batched:       {t_batch:.2f}s ({total / t_batch:.1f} tokens/s)  -> {t_seq / t_batch:.2f}x faster")
    print(f"identical outputs: {batched == one_by_one}")


if __name__ == "__main__":
    main()