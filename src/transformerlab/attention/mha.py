"""Standard multi-head attention with explicit softmax math (reference path)."""
from .core import ProjAttention, sdpa_core
from .registry import register_attention


@register_attention("mha")
class MultiHeadAttention(ProjAttention):
    """Textbook scaled-dot-product MHA. Ground truth for equivalence tests."""

    def _attend(self, q, k, v, attn_mask, is_causal):
        k, v = self._maybe_repeat_kv(k, v)          # broadcast KV heads (no-op for plain MHA)
        return sdpa_core(                           # softmax(QKᵀ/√d_k + mask)·V, computed explicitly
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,  # drop attention weights only while training
            is_causal=is_causal,
        )
