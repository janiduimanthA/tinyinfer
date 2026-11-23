"""
Inspect the GPT-2 (124M) model weights.

Prints every weight tensor's name and shape, counts parameters,
and checks whether lm_head shares weights with the token embedding (wte).
"""
import torch
from transformers import GPT2LMHeadModel

# ===== Part 1: Load GPT-2 =====
print("Loading GPT-2... (first run downloads ~550MB)")
model = GPT2LMHeadModel.from_pretrained("gpt2")
sd = model.state_dict()  # dict: weight name -> tensor

# ===== Part 2: Print every weight's name and shape =====
total = 0
for name, tensor in sd.items():
    print(f"{name:45s} {tuple(tensor.shape)}")
    total += tensor.numel()  # number of values in this tensor

print(f"\nTotal parameters (as printed): {total / 1e6:.1f}M")

# ===== Part 3: Quick checks =====
print("\n--- Checks ---")

# First 5 values of token 0's embedding vector
print("Token 0 embedding (first 5):", sd["transformer.wte.weight"][0][:5])

# Are lm_head and wte the same tensor? (weight tying)
tied = torch.equal(sd["lm_head.weight"], sd["transformer.wte.weight"])
print("lm_head == wte ?", tied)

if tied:
    unique = total - sd["lm_head.weight"].numel()
    print(f"Unique parameters (lm_head not double-counted): {unique / 1e6:.1f}M")