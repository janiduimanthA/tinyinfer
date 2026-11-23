"""Property tests for causal self-attention (small random weights, no GPT-2 needed)."""
import numpy as np

from reference.gpt2_numpy import attention

rng = np.random.default_rng(0)
T, C, N_HEAD = 5, 8, 2


def random_params():
    return {
        "attn.c_attn.weight": rng.standard_normal((C, 3 * C)).astype(np.float32),
        "attn.c_attn.bias": rng.standard_normal(3 * C).astype(np.float32),
        "attn.c_proj.weight": rng.standard_normal((C, C)).astype(np.float32),
        "attn.c_proj.bias": rng.standard_normal(C).astype(np.float32),
    }


def test_output_shape():
    x = rng.standard_normal((T, C)).astype(np.float32)
    assert attention(x, random_params(), N_HEAD).shape == (T, C)


def test_is_causal():
    """Changing the LAST token must not change outputs for earlier tokens."""
    p = random_params()
    x1 = rng.standard_normal((T, C)).astype(np.float32)
    x2 = x1.copy()
    x2[-1] = rng.standard_normal(C)  # change only the last token

    out1 = attention(x1, p, N_HEAD)
    out2 = attention(x2, p, N_HEAD)

    np.testing.assert_allclose(out1[:-1], out2[:-1], atol=1e-6)  # past: unchanged
    assert not np.allclose(out1[-1], out2[-1])                    # last: changed


def test_first_token_only_sees_itself():
    """Token 0 can only attend to itself, so its output ignores all other tokens."""
    p = random_params()
    x = rng.standard_normal((T, C)).astype(np.float32)
    full = attention(x, p, N_HEAD)
    alone = attention(x[:1], p, N_HEAD)
    np.testing.assert_allclose(full[0], alone[0], atol=1e-5)