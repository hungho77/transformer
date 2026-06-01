"""ALiBi: the bias is m_h·(j−i) (causal-masked), and with zero slopes it reduces
to plain MHA. These pin the position-bias math that ALiBi rests on."""
import math

import torch

from transformerlab.attention import AttentionConfig, build_attention
from transformerlab.attention.alibi import _alibi_slopes

B, S, DIM, HEADS = 2, 12, 64, 8


def test_slopes_powers_of_two():
    # H a power of two -> slopes are exactly 2^(-1), 2^(-2), ..., 2^(-H).
    slopes = _alibi_slopes(HEADS).tolist()
    assert all(math.isclose(s, 2.0 ** -(i + 1), rel_tol=1e-6) for i, s in enumerate(slopes))


def test_slopes_non_power_of_two_distinct():
    # Non-power-of-two H still yields H distinct, positive slopes.
    slopes = _alibi_slopes(6)
    assert slopes.numel() == 6
    assert len(set(slopes.tolist())) == 6
    assert (slopes > 0).all()


def test_bias_matrix_values():
    # bias[h,i,j] = m_h·(j−i) on/below the diagonal, −inf above it (causal).
    attn = build_attention("alibi", AttentionConfig(dim=DIM, num_heads=HEADS)).eval()
    bias = attn._alibi_bias(S, S, is_causal=True, device=torch.device("cpu"), dtype=torch.float32)
    assert bias.shape == (1, HEADS, S, S)
    slopes = _alibi_slopes(HEADS)
    for h in range(HEADS):
        for i in range(S):
            assert math.isclose(bias[0, h, i, i].item(), 0.0, abs_tol=1e-6)   # diagonal: (i−i)=0
            if i > 0:                                                          # one step back: m_h·(−1)
                assert math.isclose(bias[0, h, i, i - 1].item(), -slopes[h].item(), rel_tol=1e-6)
            if i + 1 < S:                                                      # future key -> masked out
                assert bias[0, h, i, i + 1].item() == float("-inf")


def test_zero_slopes_equals_mha():
    # With all slopes forced to 0 the ALiBi bias vanishes, so given identical
    # q/k/v/o weights ALiBi must match plain MHA exactly.
    cfg = AttentionConfig(dim=DIM, num_heads=HEADS)
    mha = build_attention("mha", cfg).eval()
    alibi = build_attention("alibi", cfg).eval()
    alibi.load_state_dict(mha.state_dict(), strict=False)   # share projections (slopes is a non-persistent buffer)
    alibi.slopes.zero_()                                     # kill the position bias
    alibi._bias_cache.clear()                               # drop any cached non-zero bias
    x = torch.randn(B, S, DIM)
    with torch.no_grad():
        assert torch.allclose(mha(x, is_causal=True), alibi(x, is_causal=True), atol=1e-5)
