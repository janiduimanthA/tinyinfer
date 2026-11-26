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

## Day 3: Sampling, streaming, benchmarking

### What I built
- Temperature, top-k and top-p sampling, with property tests (randomness tested by rules)
- Seeded RNG for reproducible sampling
- Streaming output with Python generators (`yield`)
- A benchmark harness: warm-up, repeats, median, saved as JSON with machine info
- First optimization: compute lm_head logits only for the last position

### Key insights
- Greedy decoding loops ("cricketing prowess" x3). Sampling fixes it.
- Low temperature = safe but repetitive; high temperature = creative but weird
- A single timing is meaningless: my laptop gave 3.0 to 9.3 tokens/s for the same code
- Step time grows linearly with context because everything is recomputed
- Only position-independent layers (ln_f, lm_head) can drop positions. Attention can't,
  because it mixes positions. Caching K and V is how blocks can skip the past too.

## Day 4: KV cache

### What I built
- A pre-allocated KV cache: (layers, heads, 1024, head_size) for K and V, ~75 MB
- Cached attention for new tokens with a rectangular (T x end) causal mask
- Prefill + decode generation in a single loop (`pending` tokens)
- Tests: prefill vs full, token-by-token vs full (every step), cached vs uncached generation
- Moved shared fixtures into conftest.py (model loaded once per test session)
- Result: 5.5 → 21.1 tokens/s for 100 tokens (3.8x), identical text

### Key insights
- Causal attention means past K/V never change, so they can be cached
- Cache K and V, not Q: past queries only produced past outputs, which are done
- Positions must continue from cache.length, not restart at 0 (easy bug!)
- MLP and LayerNorm need no cache: they're per-token. Only attention mixes tokens.
- Speedup grows with length: uncached cost grows with context, cached cost stays ~flat
- KV cache trades memory for speed. For big models with many users, the cache can be
  5x bigger than the model → PagedAttention (vLLM), GQA, KV quantization
- Industry terms: TTFT (time to first token = prefill), TPOT (time per output token = decode)

## Day 5: INT8 quantization

### What I built
- Symmetric per-channel INT8 weight quantization (QTensor: int8 values + float32 scales)
- INT8 matmul via (x @ q) * scale: scale the outputs instead of every weight
- Quality evaluation on 2,044 tokens: perplexity, top-1 agreement, KL divergence, logit diff
- An ablation study (blocks only / embeddings only / everything)
- Mixed precision: INT8 blocks + float32 embeddings → 498 → 243 MB, perplexity −1.06%
- Quality regression tests with thresholds from measured results + safety margin

### Key insights
- First attempt (everything INT8): 3.9x smaller but +4.98% perplexity, 79% top-1 agreement.
  The evaluation tool caught it: the generated text still "looked fine"!
- Never loosen a failing test's threshold to make it pass. Find out why it fails.
- Ablation found the culprit: the embedding table, which is also the tied lm_head.
  Its errors directly reorder the next-token ranking. Blocks quantize almost losslessly.
- My hypothesis (outliers inside the blocks) was half wrong. Measure, don't assume!
- Raw logit diff is misleading: blocks-only had max diff 52 but 43x lower KL.
  Softmax ignores shifting all logits together. KL divergence measures what matters.
- "Greedy text changed" ≠ "quality got worse": one near-tie flips → butterfly effect
- Per-channel scales isolate outliers within a column; rounding error ≤ scale / 2 (tested)
- NumPy INT8 is slower (a float32 copy per matmul). Speed needs a native int8 kernel.
- Decode reads ~498 MB in ~23 ms ≈ 21 GB/s, so it's likely memory-bound
- Ship the safe fix (mixed precision) now; note the open question (group-wise INT8) for later