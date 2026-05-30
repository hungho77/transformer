import torch

from transformerlab.layers import PatchEmbedding, RMSNorm, RotaryEmbedding, build_ffn, build_norm


def test_rmsnorm_shape_and_scale():
    norm = RMSNorm(32)
    x = torch.randn(4, 10, 32)
    out = norm(x)
    assert out.shape == x.shape
    # unit-RMS rows after norm (weight initialized to ones)
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2)


def test_build_norm_and_ffn():
    assert isinstance(build_norm("layer", 16), torch.nn.LayerNorm)
    for ffn_type in ["mlp", "swiglu"]:
        ffn = build_ffn(ffn_type, 16, 64)
        assert ffn(torch.randn(2, 5, 16)).shape == (2, 5, 16)


def test_rotary_preserves_norm():
    rope = RotaryEmbedding(head_dim=16)
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)
    q2, k2 = rope(q, k)
    assert q2.shape == q.shape
    # rotation is norm-preserving
    assert torch.allclose(q2.norm(dim=-1), q.norm(dim=-1), atol=1e-4)


def test_patch_embedding():
    pe = PatchEmbedding(image_size=32, patch_size=8, in_channels=3, dim=48)
    out = pe(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 16, 48)  # (32/8)^2 = 16 patches
