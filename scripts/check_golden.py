"""Peek inside the golden file: what does GPT-2 predict after a prompt?"""
import numpy as np
import tiktoken

data = np.load("tests/golden/gpt2_golden.npz")
enc = tiktoken.get_encoding("gpt2")

print("Keys in file:", list(data.keys()))

logits = data["logits_1"]          # prompt 1: "The capital of France is"
print("logits_1 shape:", logits.shape)

last = logits[-1]                  # scores for the token AFTER "is"
top5 = np.argsort(last)[::-1][:5]  # 5 highest scores

print("\nTop 5 guesses after 'The capital of France is':")
for t in top5:
    print(f"  {enc.decode([int(t)])!r:12} score = {last[t]:.2f}")