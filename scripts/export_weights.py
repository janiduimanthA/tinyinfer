"""
Export GPT-2 weights from Hugging Face to a plain NumPy file.

After this, our engine never needs PyTorch: it only loads weights/gpt2.npz.
"""
from pathlib import Path

import numpy as np
from transformers import GPT2LMHeadModel

OUT_PATH = Path("weights/gpt2.npz")


def main():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    sd = model.state_dict()

    weights = {}
    for name, tensor in sd.items():
        # Causal-mask buffers (only in some library versions), not learned weights
        if name.endswith(".attn.bias") or name.endswith(".attn.masked_bias"):
            continue
        # lm_head is tied to (identical to) transformer.wte.weight, so skip it
        if name == "lm_head.weight":
            continue
        weights[name] = tensor.detach().cpu().numpy().astype(np.float32)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_PATH, **weights)

    total = sum(w.size for w in weights.values())
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"Saved {len(weights)} tensors, {total / 1e6:.1f}M params")
    print(f"-> {OUT_PATH} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()