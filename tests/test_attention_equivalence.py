"""Exact attention variants must produce matching outputs given identical weights.

This is what makes head-to-head benchmarks trustworthy: swapping the backend
does not change the math for the exact variants.
"""
import pytest
import torch

from transformerlab.attention import AttentionConfig, build_attention

B, S, DIM, HEADS = 2, 16, 64, 4


def _shared(name_a, name_b, cfg):
    a = build_attention(name_a, cfg).eval()
    b = build_attention(name_b, cfg).eval()
    b.load_state_dict(a.state_dict())
    return a, b


@pytest.mark.parametrize("is_causal", [True, False])
def test_mha_equals_sdpa(is_causal):
    cfg = AttentionConfig(dim=DIM, num_heads=HEADS)
    mha, sdpa = _shared("mha", "sdpa", cfg)
    x = torch.randn(B, S, DIM)
    with torch.no_grad():
        assert torch.allclose(mha(x, is_causal=is_causal), sdpa(x, is_causal=is_causal), atol=1e-5)


def test_gqa_full_equals_mha():
    # GQA with num_kv_heads == num_heads is plain MHA.
    cfg = AttentionConfig(dim=DIM, num_heads=HEADS, num_kv_heads=HEADS)
    mha, gqa = _shared("mha", "gqa", cfg)
    x = torch.randn(B, S, DIM)
    with torch.no_grad():
        assert torch.allclose(mha(x, is_causal=True), gqa(x, is_causal=True), atol=1e-5)


def test_local_full_window_equals_mha():
    cfg = AttentionConfig(dim=DIM, num_heads=HEADS, window_size=S)
    mha, local = _shared("mha", "local", cfg)
    x = torch.randn(B, S, DIM)
    with torch.no_grad():
        assert torch.allclose(mha(x, is_causal=True), local(x, is_causal=True), atol=1e-5)


def test_sink_no_sinks_full_window_equals_local():
    # sink with sink_size=0 and window>=S has no sink columns and a full band,
    # so its keep-mask collapses to local's full-causal mask.
    cfg = AttentionConfig(dim=DIM, num_heads=HEADS, window_size=S, extra={"sink_size": 0})
    local, sink = _shared("local", "sink", cfg)
    x = torch.randn(B, S, DIM)
    with torch.no_grad():
        assert torch.allclose(local(x, is_causal=True), sink(x, is_causal=True), atol=1e-5)


def test_sink_keeps_sinks_and_window():
    # With a narrow window, sink must attend to BOTH the first sink_size keys and
    # the recent window — strictly more keys than plain local, so outputs differ.
    cfg = AttentionConfig(dim=DIM, num_heads=HEADS, window_size=4, extra={"sink_size": 2})
    local, sink = _shared("local", "sink", cfg)
    x = torch.randn(B, S, DIM)
    with torch.no_grad():
        assert not torch.allclose(local(x, is_causal=True), sink(x, is_causal=True), atol=1e-4)


@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("window", [4, S])
def test_local_flex_matches_banded_local(is_causal, window):
    # The flex_attention block-mask backend must match the banded reference.
    cfg = AttentionConfig(dim=DIM, num_heads=HEADS, window_size=window)
    local, flex = _shared("local", "local_flex", cfg)
    x = torch.randn(B, S, DIM)
    with torch.no_grad():
        assert torch.allclose(local(x, is_causal=is_causal), flex(x, is_causal=is_causal), atol=1e-5)
