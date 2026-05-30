import pytest
import torch

from transformerlab.attention import AttentionConfig, available_attentions, build_attention

B, S, DIM, HEADS = 2, 16, 32, 4


def _cfg(name):
    kw = dict(dim=DIM, num_heads=HEADS)
    if name == "gqa":
        kw["num_kv_heads"] = 2
    if name == "local":
        kw["window_size"] = 8
    return AttentionConfig(**kw)


@pytest.mark.parametrize("name", available_attentions())
def test_self_attention_shape(name):
    attn = build_attention(name, _cfg(name)).eval()
    x = torch.randn(B, S, DIM)
    out = attn(x, is_causal=True)
    assert out.shape == (B, S, DIM)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name", ["mha", "sdpa", "gqa", "mqa", "local"])
def test_cross_attention_shape(name):
    attn = build_attention(name, _cfg(name)).eval()
    x = torch.randn(B, S, DIM)
    kv = torch.randn(B, S + 5, DIM)
    out = attn(x, kv=kv, is_causal=False)
    assert out.shape == (B, S, DIM)


@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
def test_gqa_kv_head_counts(num_kv_heads):
    cfg = AttentionConfig(dim=DIM, num_heads=HEADS, num_kv_heads=num_kv_heads)
    out = build_attention("gqa", cfg).eval()(torch.randn(B, S, DIM), is_causal=True)
    assert out.shape == (B, S, DIM)
