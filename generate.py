"""
Generate text with the tinyinfer NumPy reference engine.

Usage:
    python generate.py "Once upon a time"                  # sampling (temperature 0.8, top-k 40)
    python generate.py "Once upon a time" --greedy         # deterministic
    python generate.py "Once upon a time" -t 1.2 -p 0.9    # more creative, nucleus sampling
    python generate.py "Once upon a time" --seed 42        # reproducible sampling
"""
import argparse
import time

import tiktoken

from reference.gpt2_numpy import GPT2


def main():
    parser = argparse.ArgumentParser(description="tinyinfer: GPT-2 in pure NumPy")
    parser.add_argument("prompt", help="text for the model to continue")
    parser.add_argument("-n", "--max-new-tokens", type=int, default=30)
    parser.add_argument("-t", "--temperature", type=float, default=0.8)
    parser.add_argument("-k", "--top-k", type=int, default=40)
    parser.add_argument("-p", "--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducible output")
    parser.add_argument("--greedy", action="store_true", help="always pick the top token")
    parser.add_argument("--no-cache", action="store_true", help="disable the KV cache (slow, for comparison)")
    parser.add_argument("--int8", action="store_true", help="use INT8-quantized weights (~4x less memory)")
    args = parser.parse_args()

    enc = tiktoken.get_encoding("gpt2")

    print("Loading weights...")
    t0 = time.perf_counter()
    model = GPT2("weights/gpt2.npz", int8=args.int8)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s ({model.nbytes / 1e6:.0f} MB of weights)\n")

    sampling = dict(
        temperature=0.0 if args.greedy else args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        use_cache=not args.no_cache,
    )

    ids = enc.encode(args.prompt)
    print(args.prompt, end="", flush=True)

    t0 = time.perf_counter()
    count = 0
    for token_id in model.stream(ids, args.max_new_tokens, **sampling):
        print(enc.decode([token_id]), end="", flush=True)  # print each token as soon as it's ready
        count += 1
    elapsed = time.perf_counter() - t0

    print(f"\n\n--- {count} tokens in {elapsed:.2f}s ({count / elapsed:.1f} tokens/s) ---")


if __name__ == "__main__":
    main()