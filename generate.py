"""
Generate text with the tinyinfer NumPy reference engine.

Usage:
    python generate.py "Once upon a time" -n 30
"""
import argparse
import time

import tiktoken

from reference.gpt2_numpy import GPT2


def main():
    parser = argparse.ArgumentParser(description="tinyinfer: GPT-2 in pure NumPy")
    parser.add_argument("prompt", help="text for the model to continue")
    parser.add_argument("-n", "--max-new-tokens", type=int, default=30)
    args = parser.parse_args()

    enc = tiktoken.get_encoding("gpt2")

    print("Loading weights...")
    t0 = time.perf_counter()
    model = GPT2("weights/gpt2.npz")
    print(f"Loaded in {time.perf_counter() - t0:.1f}s\n")

    ids = enc.encode(args.prompt)
    t0 = time.perf_counter()
    out = model.generate(ids, args.max_new_tokens)
    elapsed = time.perf_counter() - t0

    print(enc.decode(out))
    n = args.max_new_tokens
    print(f"\n--- {n} tokens in {elapsed:.2f}s ({n / elapsed:.1f} tokens/s) ---")


if __name__ == "__main__":
    main()