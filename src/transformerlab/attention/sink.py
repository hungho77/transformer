"""Attention with sinks (StreamingLLM; Xiao et al., 2023).

Pure sliding-window attention lets you decode a stream with a *fixed* cache, but
quality collapses once the stream is longer than the window. The cause: softmax
must distribute probability over *something*, and early on the model learns to
park surplus mass on the first few tokens — the "attention sinks". Evict those
and the distribution is forced onto in-window tokens, corrupting the output.

StreamingLLM keeps both: the first ``sink_size`` tokens AND a sliding window of
the last ``window_size`` keys, so the cache stays bounded at
``sink_size + window_size`` while quality holds.

    keep[i, j] = (j < sink_size)  OR  (i-window+1 ≤ j ≤ i)      (then ∧ causal)
    out_i      = softmax( q_i·k_jᵀ / √d_k  over the kept j ) · v

With ``sink_size = 0`` and ``window_size ≥ seq_len`` this degenerates to full
causal attention (== ``local`` with a full window), which the equivalence test
relies on.

Config (via AttentionConfig): ``window_size`` (sliding window W), and
``extra["sink_size"]`` (number of sink tokens, default 4).
"""
import torch

from .base import MaskType
from .core import ProjAttention, make_sink_window_mask, sdpa_core
from .registry import register_attention

_FULL = 1 << 30  # sentinel "window": ≥ any real seq_len, so the band becomes full attention


@register_attention("sink")
class SinkAttention(ProjAttention):
    """Sliding-window attention that also always keeps the first ``sink_size`` keys."""

    @classmethod
    def supports_mask(cls, mask_type: MaskType) -> bool:
        # Builds an explicit keep-mask over the score matrix; windowed like `local`.
        return mask_type in (MaskType.NONE, MaskType.CAUSAL, MaskType.SLIDING)

    def _attend(self, q, k, v, attn_mask, is_causal):
        k, v = self._maybe_repeat_kv(k, v)              # broadcast KV heads (GQA/MQA) up to H
        window = self.cfg.window_size or _FULL          # no window configured -> behave like full attention
        sink_size = int(self.cfg.extra.get("sink_size", 4))  # k_sink: tokens always kept
        dropout_p = self.dropout_p if self.training else 0.0

        # keep[i, j] = sink (j < sink_size) ∪ window (i-W+1 ≤ j ≤ i), ∧ causal
        keep = make_sink_window_mask(q.shape[-2], k.shape[-2], window, sink_size,
                                     q.device, causal=is_causal)
        if attn_mask is not None:                       # AND in any extra (e.g. key-padding) mask
            extra = attn_mask if attn_mask.dtype == torch.bool else (attn_mask >= 0)
            keep = keep & extra
        return sdpa_core(q, k, v, attn_mask=keep, dropout_p=dropout_p)  # softmax(QKᵀ/√d on the kept set)·V
