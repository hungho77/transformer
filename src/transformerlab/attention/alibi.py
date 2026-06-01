"""Attention with Linear Biases (ALiBi; Press et al., 2022).

ALiBi throws away positional *embeddings* entirely. Instead it adds a static,
per-head linear penalty to the attention scores that grows with the query-key
distance, so a query attends less to keys that are further back:

    scores_ij = q_i·k_jᵀ / √d_k  +  m_h · (j − i)        (causal: j ≤ i)
    out_i     = softmax_j(scores_ij) · v_j

The bias term ``m_h · (j − i)`` is ≤ 0 for past keys (j ≤ i) and becomes more
negative with distance, a recency prior. ``m_h`` is a *fixed* (non-learned),
head-specific slope drawn from a geometric sequence so different heads look back
over different ranges:

    m_h = 2^(−8h / H)   for head h = 1..H        (H a power of two)

For non-power-of-two H the paper interleaves the slopes of the next power of two
(implemented in :func:`_alibi_slopes`). Because position enters only through this
score bias — with nothing tied to a trained sequence length — ALiBi extrapolates
to sequences longer than those seen in training.

Note: ALiBi carries its own position signal, so it must be used *without* any
token-position embedding (no rotary, no learned positional embedding). Wiring it
into ``GPT`` cleanly needs a "no positional embedding" path (GPT currently adds a
learned one when ``use_rotary=False``); that is a documented follow-up.
"""
from __future__ import annotations

import math

import torch

from .base import MaskType
from .core import ProjAttention, make_causal_mask, sdpa_core
from .registry import register_attention


def _alibi_slopes(num_heads: int) -> torch.Tensor:
    r""":math:`m_h = 2^{-8h/H}` per head, with the Press et al. fallback for
    non-power-of-two head counts.

    For a power-of-two H the slopes are the geometric sequence with ratio
    ``2^(−8/H)`` starting at ``2^(−8/H)``. Otherwise we take the slopes of the
    nearest smaller power of two and interleave extra slopes from the next power
    of two up, so every head still gets a distinct, well-spread slope.
    """
    def pow2_slopes(n):
        start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3)))   # ratio = 2^(−8/n); first slope = ratio
        return [start * (start ** i) for i in range(n)]   # geometric: start^1, start^2, ... (≡ 2^(−8h/n))

    if math.log2(num_heads).is_integer():
        slopes = pow2_slopes(num_heads)
    else:
        closest = 2 ** math.floor(math.log2(num_heads))   # largest power of two ≤ H
        slopes = pow2_slopes(closest)
        # Fill the remaining heads with every other slope from the next power up.
        extra = pow2_slopes(2 * closest)[0::2]
        slopes += extra[: num_heads - closest]
    return torch.tensor(slopes, dtype=torch.float32)       # [H]


@register_attention("alibi")
class ALiBiAttention(ProjAttention):
    """Multi-head attention with ALiBi linear position biases (no embeddings)."""

    def __init__(self, cfg):
        super().__init__(cfg)
        # m_h: fixed per-head slopes. Buffer (not a Parameter) -> not learned and
        # not saved in checkpoints, mirroring RotaryEmbedding.inv_freq.
        self.register_buffer("slopes", _alibi_slopes(self.num_heads), persistent=False)
        self._bias_cache: dict = {}   # cache the [1, H, S_q, S_k] bias per shape/device/dtype

    @classmethod
    def supports_mask(cls, mask_type: MaskType) -> bool:
        # Bias is additive on the scores; honors causal/padding. No windowing.
        return mask_type in (MaskType.NONE, MaskType.CAUSAL, MaskType.PADDING)

    def _alibi_bias(self, seq_q, seq_k, is_causal, device, dtype):
        # bias[h, i, j] = m_h · (j − i), with −inf on disallowed (future) keys so
        # one fused additive mask carries both the ALiBi prior and causality.
        key = (seq_q, seq_k, is_causal, device, dtype)
        cached = self._bias_cache.get(key)
        if cached is not None:
            return cached
        offset = seq_k - seq_q                                          # align queries to end of key axis (cache-safe)
        q_idx = torch.arange(seq_q, device=device).unsqueeze(1) + offset  # absolute query pos i [S_q, 1]
        k_idx = torch.arange(seq_k, device=device).unsqueeze(0)         # key pos j [1, S_k]
        dist = (k_idx - q_idx).to(torch.float32)                       # (j − i): 0 on diagonal, <0 for past keys
        bias = self.slopes.view(self.num_heads, 1, 1) * dist           # m_h · (j − i) -> [H, S_q, S_k]
        if is_causal:
            keep = make_causal_mask(seq_q, seq_k, device)              # True where j ≤ i
            bias = bias.masked_fill(~keep, float("-inf"))              # future keys -> −inf (softmax sends to 0)
        bias = bias.unsqueeze(0).to(dtype)                            # [1, H, S_q, S_k] to broadcast over batch
        self._bias_cache[key] = bias
        return bias

    def _attend(self, q, k, v, attn_mask, is_causal):
        k, v = self._maybe_repeat_kv(k, v)              # broadcast KV heads (GQA/MQA) up to H
        # Build the additive ALiBi+causal bias and fold in any extra mask.
        bias = self._alibi_bias(q.shape[-2], k.shape[-2], is_causal, q.device, q.dtype)
        if attn_mask is not None:                       # combine with a caller-supplied mask
            if attn_mask.dtype == torch.bool:
                bias = bias.masked_fill(~attn_mask, float("-inf"))     # bool: drop disallowed positions
            else:
                bias = bias + attn_mask                 # float: additive bias stacks
        # is_causal already baked into `bias`; pass the float mask only so sdpa_core
        # adds it directly (scores = QKᵀ/√d + bias), then softmax over keys.
        return sdpa_core(q, k, v, attn_mask=bias)
