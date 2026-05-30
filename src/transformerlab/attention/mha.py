"""Standard multi-head attention with explicit softmax math (reference path)."""
from .core import ProjAttention, sdpa_core
from .registry import register_attention


@register_attention("mha")
class MultiHeadAttention(ProjAttention):
    """Textbook scaled-dot-product MHA. Ground truth for equivalence tests."""

    def _attend(self, q, k, v, attn_mask, is_causal):
        k, v = self._maybe_repeat_kv(k, v)
        return sdpa_core(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )
