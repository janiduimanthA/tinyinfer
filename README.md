# tinyinfer

[![tests](https://github.com/janiduimanthA/tinyinfer/actions/workflows/tests.yml/badge.svg)](https://github.com/janiduimanthA/tinyinfer/actions/workflows/tests.yml)
![Python 3.13](https://img.shields.io/badge/python-3.13-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**A GPT-2 inference engine built from scratch, step by step, while learning how LLM inference really works.**

Milestone 1 is a pure-NumPy reference engine: no PyTorch at runtime, every component written by hand,
tested against Hugging Face GPT-2, and benchmarked. Next: a dependency-free C++ engine for laptops
and Raspberry Pi with an OpenAI-compatible API.

## Highlights

| | |
|---|---|
| ✅ **Correct** | Matches Hugging Face GPT-2 (124M) to ~1e-4 max logit difference, with identical greedy generations. 48 tests. |
| 🔑 **KV cache** | Per-token latency nearly independent of context: **96x faster at 1024 tokens** (3.0 s → 31 ms). |
| 🗜️ **INT8** | **498 → 243 MB** with −1.06% perplexity. Mixed precision chosen by an ablation study. |
| 🚌 **Batching** | **5.3x throughput** (213 tokens/s at batch 16), outputs identical to one-by-one decoding. |
| 🎲 **Sampling** | Temperature, top-k, top-p, seeded RNG, streaming output. |

![Time per generated token vs context length](docs/bench_step_time.png)

## Quick start

```bash
git clone https://github.com/janiduimanthA/tinyinfer.git
cd tinyinfer
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # only needed to export weights and run comparison tests
pip install -r requirements.txt

python scripts/export_weights.py       # one-time: download GPT-2 and save it as weights/gpt2.npz
python generate.py "Sri Lanka is famous for"
```

### Generation options

```bash
python generate.py "Once upon a time"                 # sampling: temperature 0.8, top-k 40
python generate.py "Once upon a time" --greedy        # deterministic
python generate.py "Once upon a time" -t 1.2 -p 0.9   # more creative, nucleus sampling
python generate.py "Once upon a time" --seed 42       # reproducible sampling
python generate.py "Once upon a time" --int8          # INT8 weights (~2x less memory)
python generate.py "Once upon a time" --no-cache      # disable the KV cache (slow, for comparison)
python batch_demo.py                                  # 4 prompts decoded together
```

## How it works

### Forward pass
Token + position embeddings → 12 transformer blocks (pre-LayerNorm, causal multi-head attention,
GELU MLP, residuals) → final LayerNorm → logits through the tied embedding table.
Weight layouts and gotchas are documented in [docs/architecture.md](docs/architecture.md).

Every op (linear, GELU, softmax, LayerNorm, log-softmax) is tested against PyTorch, attention has
property tests proving it never sees the future, and the full model is tested against golden
outputs saved from Hugging Face GPT-2.

### KV cache

Without a cache, generating token N recomputes all N tokens through all 12 layers.
But attention is causal: a token's keys and values never change once computed.
So the engine stores K and V for every layer in a pre-allocated cache (~75 MB for
GPT-2 at 1024 tokens) and splits generation into two phases:

- **Prefill:** process the whole prompt once, filling the cache (time to first token)
- **Decode:** process only the newest token, reading past K/V from the cache

Only K and V are cached, not Q: past queries were only needed for past outputs.
Correctness is tested token-by-token against full recomputation, and cached greedy
generation matches Hugging Face GPT-2 exactly.

**The cost is memory.** Run `python scripts/kv_memory.py`: for Llama-2-7B (fp16, 4096 context),
32 concurrent users need ~69 GB of KV cache, about 5x the model's own weights. This is why
production engines use PagedAttention, grouped-query attention, and KV quantization.

### INT8 quantization

Weight-only, symmetric, per-channel INT8: `w ≈ q × scale`, with one float32 scale per
output column, so an outlier weight only affects its own column. INT8 matmuls use
`(x @ q) × scale`, scaling the outputs instead of every weight.

```bash
python eval_quant.py            # quality, memory and speed vs float32
python eval_quant.py --ablate   # which parts can be quantized safely?
```

Results on 2,044 tokens of Tiny Shakespeare:

| | float32 | INT8 (mixed precision) |
|---|---|---|
| Weights in memory | 498 MB | **243 MB (2.05x smaller)** |
| Perplexity | 88.90 | 87.96 (−1.06%) |
| Top-1 agreement | | 97.46% |
| KL divergence (mean) | | 0.00135 |
| Greedy output | | identical to float32 on the test prompt |

**Ablation: why the embeddings stay float32.** Quantizing everything first gave a 3.9x smaller
model but **+4.98% perplexity and only 79% top-1 agreement**. Quantizing each part separately
found the cause:

| Quantized | Size | Perplexity | Top-1 | KL |
|---|---|---|---|---|
| Blocks only | 243 MB | −1.06% | 97.46% | 0.00135 |
| Embeddings only | 382 MB | +6.14% | 79.25% | 0.05809 |
| Everything | 128 MB | +4.98% | 79.05% | 0.05955 |

The token embedding table doubles as the output layer (weight tying), so its rounding
errors directly reorder the next-token ranking. The default `--int8` therefore quantizes
the block matmuls and keeps embeddings in float32. Next step: group-wise INT8
(one scale per 32–64 values, like llama.cpp's Q8_0) for the embeddings.

**Lesson learned: raw logit difference is a misleading metric.** "Blocks only" has a
*larger* max logit difference (52 vs 8) but 43x lower KL divergence. Softmax ignores
shifting all logits by the same amount, so a big uniform shift is harmless.
KL divergence measures what actually matters.

**Speed, honestly:** in NumPy, INT8 is slower (95.8 ms vs 23.4 ms per decode step at context 128):
NumPy can't multiply int8 by float32 directly, so every matmul first makes a temporary float32
copy of the weights. Decoding is memory-bound (see below), so smaller weights should pay off in a
native kernel that reads int8 directly, planned for the C++ engine.

### Batching

Several sequences are decoded together. Each prompt is prefilled separately into its own
slot of a batch KV cache (prompts have different lengths), then every decode step processes
one new token for **all** sequences at once: one matmul per layer serves every sequence, so
the weights are read once per step instead of once per sequence. Each slot keeps its own
position and attention mask. Batched outputs are tested to be identical to one-by-one
generation and to Hugging Face GPT-2.

![Throughput and latency vs batch size](docs/bench_batch.png)

| Batch size | Step latency (ms) | Throughput (tokens/s) | Throughput vs 1 | Latency vs 1 |
|---|---|---|---|---|
| 1 | 24.8 | 40.3 | 1.00x | 1.00x |
| 2 | 38.6 | 51.8 | 1.29x | 1.56x |
| 4 | 37.5 | 106.7 | 2.65x | 1.51x |
| 8 | 43.3 | 184.8 | **4.59x** | 1.75x |
| 16 | 75.1 | 213.0 | 5.29x | 3.03x |

*Context 64, median of 5 runs after 2 warm-ups.* Batch 8 is the sweet spot on this laptop:
4.6x the throughput for 1.75x the per-token latency.

**Why batch 2 barely helps: a matmul micro-benchmark.**

| Matmul (ms) | B=1 | B=2 | B=4 | B=8 | B=16 |
|---|---|---|---|---|---|
| attn.c_attn (768×2304) | 0.25 | 0.56 | 0.53 | 0.54 | 0.63 |
| mlp.c_fc (768×3072) | 0.23 | 0.66 | 0.66 | 0.67 | 0.87 |
| mlp.c_proj (3072×768) | 0.30 | 0.76 | 0.78 | 0.78 | 0.84 |
| lm_head (768×50257) | 5.85 | 7.16 | 6.99 | 7.55 | 9.10 |

- **B=1 → B=2 jumps 2.3–2.9x:** with one row, BLAS uses a fast matrix-vector path (gemv);
  with two or more rows it switches to matrix-matrix (gemm), which first packs (copies)
  the weights. For skinny matrices that overhead dominates.
- **B=2 → B=8 is almost flat:** once in gemm, more rows are nearly free, because the
  weights are read once regardless. This is batching working as intended.
- **lm_head: 16x the tokens for 1.6x the time.** At B=1 it streams 154 MB in 5.85 ms
  (≈26 GB/s), near this laptop's memory bandwidth. Decoding is memory-bound, which is
  why batching (and smaller weights) pay off.
- Smaller matrices stay hot in the CPU cache during the micro-benchmark, so their times
  look faster than real DRAM reads. Only lm_head (154 MB) reflects true memory bandwidth.

**Takeaway for the C++ engine:** a decode kernel built for skinny matrices (2–16 rows)
should close the gap to ideal scaling that generic BLAS leaves.

## Benchmarks

Time per generated token (one decode step), median of 5 runs after 2 warm-up runs.
Run `python bench.py` then `python scripts/plot_bench.py` to reproduce.

| Context | Full logits (ms) | Last-position logits (ms) | KV cache (ms) | KV cache speedup |
|---|---|---|---|---|
| 8 | 49.6 | 36.4 | 23.3 | 2.1x |
| 32 | 98.8 | 86.0 | 22.5 | 4.4x |
| 64 | 146.6 | 253.4* | 25.9 | 5.7x |
| 128 | 279.7 | 249.4 | 22.6 | 12.4x |
| 256 | 567.4 | 499.0 | 23.9 | 23.7x |
| 512 | 1221.0 | 1164.7 | 26.1 | 46.8x |
| 1024 | 3018.1 | 3063.8* | 31.3 | **96.4x** |

**The KV cache makes decoding nearly independent of context length:** 128x more
context (8 → 1024) costs only ~34% more time per token (23.3 → 31.3 ms), so the engine
stays above ~30 tokens/s all the way to GPT-2's full 1024-token window. Without the
cache, every new token recomputes the whole sequence, so a token at 1024 takes over 3 seconds.

**Last-position logits** help at short contexts (1.36x at 8) but the gain fades at long
ones: lm_head is ~30% of the FLOPs but much less of the wall time, since it's one large,
efficient BLAS matmul while the blocks run many smaller matmuls and memory-bound
elementwise ops. FLOPs ≠ time.

\* *Measurement noise: last-position logits never does more work than full logits.
Its small gain at long contexts is within this laptop's run-to-run variance
(thermal throttling and background load). The KV cache's 96x gain is far above the noise.*

### Optimization history
- Fixed a silent float64 upcast under NumPy 2 (1.3 → 2.0 tokens/s end-to-end)
- Compute lm_head logits only for the last position during generation (up to 1.36x)
- KV cache with prefill/decode: up to 96x faster per token at 1024 context;
  100-token generation went 5.5 → 21.1 tokens/s end-to-end
- Mixed-precision INT8 (blocks INT8, embeddings float32): 498 → 243 MB, perplexity −1.06%, top-1 97.46%
- Batched decoding with per-sequence positions and masks: up to 5.3x throughput (213 tokens/s at batch 16)

## Testing

```bash
pytest -v
```

48 tests: ops vs PyTorch, attention causality properties, golden logits and greedy outputs
from Hugging Face GPT-2, KV cache token-by-token equivalence, sampling properties,
quantization error bounds, INT8 quality regression thresholds, and batched vs one-by-one
equivalence. Tests that need the exported weights skip cleanly if they're missing.
CI runs the full suite, including the golden tests, on every push.

## Project structure

```
reference/          NumPy reference engine
  gpt2_numpy.py       forward pass, KV cache, batching
  ops.py              linear, gelu, softmax, layer_norm, log_softmax
  sampling.py         temperature, top-k, top-p
  quant.py            per-channel INT8 quantization
  metrics.py          NLL and perplexity
tests/              test suite + golden outputs from Hugging Face GPT-2
scripts/            weight export, golden outputs, plots, KV-cache memory calculator
benchmarks/         benchmark results (JSON)
docs/               architecture notes and benchmark plots
generate.py         text generation CLI
batch_demo.py       batched generation demo
bench.py            decode-step benchmark vs context length
bench_batch.py      batching benchmark + matmul micro-benchmark
eval_quant.py       INT8 quality evaluation and ablation
```

## Roadmap

- [x] **Milestone 1: NumPy reference engine.** Correctness, KV cache, sampling, INT8, batching
- [ ] **Milestone 2: C++ engine.** Tested op-by-op against the reference; multi-threading and SIMD;
      int8 kernels that read quantized weights directly; a decode kernel for skinny matmuls
- [ ] **Milestone 3: Product features.** Qwen3-0.6B support (RoPE, RMSNorm, GQA), streaming
      OpenAI-compatible HTTP server, CLI
- [ ] **Milestone 4: Edge.** Raspberry Pi / ARM (NEON), benchmarks vs llama.cpp

## Learning log

This project is built in public while learning. [LEARNING_LOG.md](LEARNING_LOG.md) records
what I built each day, the bugs I hit, and what I learned (including the times my predictions were wrong).

## Acknowledgements

- Andrej Karpathy's *Neural Networks: Zero to Hero* series, especially "Let's build GPT"
- OpenAI for the GPT-2 weights; Hugging Face `transformers` as the correctness reference

## License

[MIT](LICENSE)