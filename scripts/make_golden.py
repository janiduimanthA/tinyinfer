"""
Generate golden reference outputs from Hugging Face GPT-2.

These outputs are the "answer key" for tinyinfer: every engine we build
(NumPy, C++, quantized) must reproduce them.

Saves tests/golden/gpt2_golden.npz containing, for each prompt i:
  ids_i     - input token IDs                         shape (T,)
  logits_i  - next-token scores at every position     shape (T, 50257)
  greedy_i  - prompt + 20 greedily generated tokens   shape (T + 20,)
"""
from pathlib import Path

import numpy as np
import torch
import tiktoken
from transformers import GPT2LMHeadModel

PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
    "Sri Lanka is an island",
]
MAX_NEW_TOKENS = 20
OUT_PATH = Path("tests/golden/gpt2_golden.npz")


def main():
    enc = tiktoken.get_encoding("gpt2")                       # text <-> token IDs
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()    # eval() = inference mode

    out = {"prompts": np.array(PROMPTS)}

    for i, prompt in enumerate(PROMPTS):
        ids = enc.encode(prompt)
        x = torch.tensor([ids])                               # shape (1, T)

        with torch.no_grad():                                 # no training -> no gradients
            # 1. Scores for the next token at every position
            logits = model(x).logits[0].numpy()               # (T, 50257)

            # 2. Generate 20 tokens, always picking the highest score (greedy)
            greedy = model.generate(
                x,
                attention_mask=torch.ones_like(x),
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=enc.eot_token,
            )[0].tolist()

        out[f"ids_{i}"] = np.array(ids, dtype=np.int64)
        out[f"logits_{i}"] = logits.astype(np.float32)
        out[f"greedy_{i}"] = np.array(greedy, dtype=np.int64)

        print(f"[{i}] prompt : {prompt!r}")
        print(f"    tokens : {ids}")
        print(f"    output : {enc.decode(greedy)!r}\n")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_PATH, **out)
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"Saved {OUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()