
# GPT-2 (124M) Architecture

## Configuration
| Setting | Value |
|---|---|
| vocab_size | 50,257 |
| n_embd (d_model) | 768 |
| n_head | 12 (head_size = 64) |
| n_layer | 12 |
| block_size (max context) | 1,024 |
| Parameters | 124.4M (163.0M if lm_head counted separately) |

## Weights

### Embeddings
| Weight | Shape | What it is |
|---|---|---|
| `transformer.wte.weight` | (50257, 768) | Token embedding table |
| `transformer.wpe.weight` | (1024, 768) | Position embedding table |

### Transformer Block (repeated 12×, `h.0` … `h.11`)
| Weight | Shape | What it is |
|---|---|---|
| `ln_1.weight`, `ln_1.bias` | (768,) | LayerNorm before attention |
| `attn.c_attn.weight` | (768, 2304) | Q, K, V projections combined (3 × 768) |
| `attn.c_attn.bias` | (2304,) | Q, K, V biases |
| `attn.c_proj.weight` | (768, 768) | Output projection after concatenating heads |
| `attn.c_proj.bias` | (768,) | |
| `ln_2.weight`, `ln_2.bias` | (768,) | LayerNorm before MLP |
| `mlp.c_fc.weight` | (768, 3072) | MLP expand (4 × 768) |
| `mlp.c_fc.bias` | (3072,) | |
| `mlp.c_proj.weight` | (3072, 768) | MLP project back down |
| `mlp.c_proj.bias` | (768,) | |

### Output
| Weight | Shape | What it is |
|---|---|---|
| `transformer.ln_f.weight`, `.bias` | (768,) | Final LayerNorm |
| `lm_head.weight` | (50257, 768) | Output logits, **tied to `wte`** (same tensor) |

## Forward Pass


## ⚠️ Gotchas
1. **Conv1D layout:** `c_attn`, `c_proj`, `c_fc` weights are stored as `(in, out)`,
   not `(out, in)` like `nn.Linear`. Use `x @ W + b` directly, with **no transpose**.
2. **Weight tying:** `lm_head.weight` is identical to `wte.weight`. Store once, saves ~150MB.
3. **MLP activation:** GPT-2 uses the **tanh approximation** of GELU.
4. **`attn.bias` buffer** (if present) is the causal mask, not a learned weight.