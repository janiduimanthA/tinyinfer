"""
How much memory does the KV cache need? Our GPT-2 vs bigger models.

KV cache bytes = 2 (K and V) x layers x n_embd x context x bytes per value
(n_embd = heads x head_size, so heads cancel out)
"""


def kv_cache_bytes(n_layer, n_embd, context, bytes_per_value):
    return 2 * n_layer * n_embd * context * bytes_per_value


MODELS = [
    # name,                layers, n_embd, context, bytes/value (fp32=4, fp16=2)
    ("GPT-2 small (ours)",     12,    768,    1024, 4),
    ("GPT-2 medium",           24,   1024,    1024, 4),
    ("GPT-2 large",            36,   1280,    1024, 4),
    ("GPT-2 XL",               48,   1600,    1024, 4),
    ("Llama-2-7B (fp16)",      32,   4096,    4096, 2),
]
USERS = 32  # a server answering 32 people at once keeps 32 caches


def main():
    print(f"{'model':<20} | {'per token':>10} | {'one full context':>16} | {f'{USERS} users':>9}")
    print("-" * 66)
    for name, layers, n_embd, context, nbytes in MODELS:
        per_token = kv_cache_bytes(layers, n_embd, 1, nbytes)
        full = kv_cache_bytes(layers, n_embd, context, nbytes)
        print(f"{name:<20} | {per_token / 1024:>7.0f} KB | {full / 1e6:>13.0f} MB "
              f"| {USERS * full / 1e9:>6.1f} GB")


if __name__ == "__main__":
    main()