"""A causal attention output at position i must not depend on tokens > i."""
import pytest
import torch

from transformerlab.attention import AttentionConfig, build_attention

CAUSAL_VARIANTS = ["mha", "sdpa", "gqa", "mqa", "linear", "local", "local_flex"]


@pytest.mark.parametrize("name", CAUSAL_VARIANTS)
def test_causal_no_future_leak(name):
    torch.manual_seed(0)
    B, S, DIM, H = 2, 12, 32, 4
    kw = dict(dim=DIM, num_heads=H)
    if name == "gqa":
        kw["num_kv_heads"] = 2
    if name in ("local", "local_flex"):
        kw["window_size"] = S  # full window -> still strictly causal
    attn = build_attention(name, AttentionConfig(**kw)).eval()

    x = torch.randn(B, S, DIM)
    cut = S // 2
    x2 = x.clone()
    x2[:, cut:] = torch.randn(B, S - cut, DIM)  # perturb the future only

    with torch.no_grad():
        o1 = attn(x, is_causal=True)
        o2 = attn(x2, is_causal=True)
    # outputs up to the cut must be identical
    assert torch.allclose(o1[:, :cut], o2[:, :cut], atol=1e-5)
