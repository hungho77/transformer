"""Multi-query (MQA) and grouped-query (GQA) attention.

Both reduce the number of KV heads relative to query heads, shrinking the KV
cache and KV projection cost. GQA uses ``num_kv_heads`` groups; MQA is the
special case of a single KV head. The KV-head broadcasting lives in
``ProjAttention._maybe_repeat_kv`` (via ``repeat_kv``), so the kernel itself is
identical to standard attention.
"""
import torch.nn.functional as F

from .base import AttentionConfig
from .core import ProjAttention
from .registry import register_attention


@register_attention("gqa")
class GroupedQueryAttention(ProjAttention):
    """Grouped-query attention. Set ``num_kv_heads`` < ``num_heads`` in the config."""

    def _attend(self, q, k, v, attn_mask, is_causal):
        k, v = self._maybe_repeat_kv(k, v)
        dropout_p = self.dropout_p if self.training else 0.0
        if attn_mask is not None:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=is_causal)


@register_attention("mqa")
class MultiQueryAttention(GroupedQueryAttention):
    """Multi-query attention: a single shared KV head."""

    def __init__(self, cfg: AttentionConfig):
        if cfg.num_kv_heads not in (None, 1):
            raise ValueError("MQA requires num_kv_heads == 1 (or unset).")
        cfg = AttentionConfig(**{**cfg.__dict__, "num_kv_heads": 1})
        super().__init__(cfg)
