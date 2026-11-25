# tinyinfer
A from-scratch, dependency-free LLM inference engine in C++ that runs fast
on laptops and Raspberry Pi, with an OpenAI-compatible API.

🚧 Work in progress, built step by step while learning.

## Status

✅ **Milestone 1: NumPy reference engine.** GPT-2 (124M) forward pass in pure NumPy,
matching Hugging Face outputs (max logit diff < 1e-3, identical greedy generations).

## Quick start

```bash
pip install -r requirements.txt
python scripts/export_weights.py      # one-time: export GPT-2 weights to NumPy
python generate.py "Sri Lanka is famous for" -n 30
pytest -v                             # run all tests
```

## Benchmarks

Time per generated token (one decode step), median of 5 runs after 2 warm-up runs.
Run `python bench.py` then `python scripts/plot_bench.py` to reproduce.

![Time per generated token vs context length](docs/bench_step_time.png)

| Context | Full logits (ms) | Last-position logits (ms) | Speedup |
|---|---|---|---|
| 8 | 62.9 | 42.1 | 1.49x |
| 32 | 93.4 | 82.3 | 1.13x |
| 64 | 147.3 | 128.4 | 1.15x |
| 128 | 311.5 | 262.0 | 1.19x |
| 256 | 630.4 | 595.9 | 1.06x |

**Why it grows with context:** the reference engine recomputes the whole sequence
for every new token, so a token at context 256 is ~14x slower than at context 8.
The KV cache (next milestone) removes this.

**Why the speedup shrinks at long contexts:** lm_head is ~30% of the FLOPs but much
less of the wall time. It's one large, highly efficient BLAS matmul, while the blocks
run many smaller matmuls plus memory-bound elementwise ops (gelu, softmax, layer_norm).
FLOPs ≠ time: measure, don't assume.

### History
- Fixed a silent float64 upcast under NumPy 2 (1.3 → 2.0 tokens/s end-to-end)
- Compute lm_head logits only for the last position during generation (1.06x–1.49x)