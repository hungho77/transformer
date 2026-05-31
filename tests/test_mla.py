import torch

from transformerlab.attention import AttentionConfig, build_attention
from transformerlab.models import GPT, GPTConfig


def test_mla_shapes_and_custom_latent():
    cfg = AttentionConfig(dim=64, num_heads=4, extra={"kv_latent_dim": 24, "rope_head_dim": 8})
    attn = build_attention("mla", cfg).eval()
    out = attn(torch.randn(2, 12, 64), is_causal=True)
    assert out.shape == (2, 12, 64)


def test_mla_kv_cache_matches_recompute():
    # Greedy generation with the latent KV cache must equal full recompute.
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=40, max_seq_len=48, dim=64, n_layers=2,
                          num_heads=4, attention_name="mla")).eval()
    prompt = torch.randint(0, 40, (2, 4))
    cached = model.generate(prompt.clone(), 20, temperature=1.0, top_k=1, use_cache=True)
    recompute = model.generate(prompt.clone(), 20, temperature=1.0, top_k=1, use_cache=False)
    assert torch.equal(cached, recompute)


def test_mla_cache_is_smaller_than_per_head_kv():
    # The MLA cache stores latent c_KV (d_c) + shared k_R (d_rope) per token, which
    # must be far less than full per-head K+V (2·H·head_dim).
    from transformerlab.attention import KVCache
    cfg = AttentionConfig(dim=256, num_heads=8)  # head_dim 32 -> per-head KV = 2*8*32 = 512 / token
    attn = build_attention("mla", cfg).eval()
    cache = KVCache()
    attn(torch.randn(1, 16, 256), is_causal=True, kv_cache=cache)
    cached_per_token = cache.k.shape[-1] + cache.v.shape[-1]  # d_c + d_rope
    full_kv_per_token = 2 * cfg.num_heads * (cfg.dim // cfg.num_heads)
    assert cached_per_token < full_kv_per_token
