# Learning Log and ## Day 1
## Day 1 — Transformer Core Concepts

### 1. Tokenization 🔤
- Computers only understand numbers, so text is split into **tokens** and each token gets an ID.
- `encode("hi")` → `[46, 47]`, `decode([46, 47])` → `"hi"`
- **vocab_size** = number of tokens the model knows (Shakespeare: 65 characters, GPT-2: 50,257 subword tokens)

### 2. Embedding 📋
- A **lookup table** that turns a token ID into a **vector** (a list of numbers).
- Token ID 46 → take row 46 of the table. That's all it does.
- **Token embedding** = "which token am I", **Position embedding** = "where am I in the sequence"
- Both are added together: `x = tok_emb + pos_emb`
- GPT-2: `wte (50257, 768)`, `wpe (1024, 768)`

### 3. Self-Attention ⭐
- Each token can look at **previous tokens** to gather information. **Future tokens are masked** (-∞ → softmax → 0).
- Each token produces three vectors:
  - 🔍 **Query (Q):** "What am I looking for?"
  - 🏷️ **Key (K):** "What do I contain?"
  - 🎁 **Value (V):** "What will I give if someone attends to me?"
- Formula: `softmax(Q @ Kᵀ / √d) @ V`
- Dividing by √d keeps values small so softmax doesn't focus all attention on a single token.
- Attention matrix shape is `(T, T)` → doubling sequence length means **4× more compute**.
- **KV Cache:** K and V of past tokens never change → compute once, store, and reuse.

### 4. Multi-Head Attention 🕵️
- One head can only look for one kind of pattern at a time.
- Multiple heads run **in parallel**, each looking for different things (who, when, nearby characters...).
- Outputs are concatenated → mixed together by a **projection** layer.
- GPT-2: 12 heads × 64 = 768. `c_attn (768, 2304)` = Q, K, V weights combined (3 × 768).

### 5. Residual + LayerNorm 🛣️⚖️
- **Residual:** `x = x + f(x)`. Instead of replacing old information, new information is **added** to it. Makes deep networks trainable.
- **LayerNorm:** Normalizes each token's vector to mean 0 and variance 1, then scales by a learned weight and adds a learned bias. Keeps numbers stable.
- **One Transformer Block:**

## Day 2: NumPy GPT-2 engine

### What I built
- Exported GPT-2 weights to NumPy (124.4M params, no PyTorch needed at runtime)
- Implemented ops (linear, gelu, softmax, layer_norm), each tested against PyTorch
- Implemented causal multi-head attention, with property tests proving it never sees the future
- Assembled the full GPT-2 forward pass and greedy generation
- **Matches Hugging Face GPT-2: max logit diff ~1e-4, identical greedy outputs** 🎉
- Baseline speed: X.X tokens/s

### Key insights
- Softmax subtracts the max first, so exp() never overflows (numerical stability)
- Heads are split with reshape + transpose, with no loops (like cutting a cake 🍰)
- Floating-point results are never bit-identical across libraries, so tests use a tolerance
- Generation gets slower as the sequence grows, because every step recomputes everything → KV cache will fix this