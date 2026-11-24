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

Prompt: "Sri Lanka is famous for", 30 new tokens, greedy decoding.

| Engine | Optimization | Tokens/s | Speedup |
|---|---|---|---|
| NumPy reference | none (accidental float64) | 1.3 | 1.0x |
| NumPy reference | float32 fix | 2.0 | 1.5x |