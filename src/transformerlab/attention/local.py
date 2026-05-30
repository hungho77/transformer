"""Sliding-window (local) attention.

Each query attends only to keys within ``window_size`` steps. The default
backend builds a banded boolean mask and reuses ``sdpa_core`` (always correct);
with ``window_size >= seq_len`` it degenerates to full attention, which the
equivalence test relies on. A true-sparsity flex_attention backend is left as a
future opt-in.
"""
import torch

from .base import MaskType
from .core import ProjAttention, make_sliding_window_mask, sdpa_core
from .registry import register_attention

_FULL = 1 << 30


@register_attention("local")
class SlidingWindowAttention(ProjAttention):
    @classmethod
    def supports_mask(cls, mask_type: MaskType) -> bool:
        return mask_type in (MaskType.NONE, MaskType.CAUSAL, MaskType.PADDING, MaskType.SLIDING)

    def _attend(self, q, k, v, attn_mask, is_causal):
        k, v = self._maybe_repeat_kv(k, v)
        window = self.cfg.window_size or _FULL
        keep = make_sliding_window_mask(q.shape[-2], k.shape[-2], window, q.device, causal=is_causal)
        if attn_mask is not None:
            extra = attn_mask if attn_mask.dtype == torch.bool else (attn_mask >= 0)
            keep = keep & extra
        return sdpa_core(
            q, k, v,
            attn_mask=keep,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
