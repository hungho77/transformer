"""Attention backed by torch's fused scaled_dot_product_attention.

Picks the best available kernel (flash / mem-efficient / math) automatically.
"""
import torch.nn.functional as F

from .core import ProjAttention
from .registry import register_attention


@register_attention("sdpa")
class SDPAttention(ProjAttention):
    def _attend(self, q, k, v, attn_mask, is_causal):
        k, v = self._maybe_repeat_kv(k, v)
        dropout_p = self.dropout_p if self.training else 0.0
        # F.sdpa forbids passing both an explicit mask and is_causal.
        if attn_mask is not None:
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout_p
            )
        return F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout_p, is_causal=is_causal
        )
