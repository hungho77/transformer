"""Sliding-window (local) attention.

Two backends, both registered:

- ``local``      — builds a banded boolean mask and reuses ``sdpa_core``. Always
  correct and dependency-free, but materializes the score matrix (no real
  sparsity savings). With ``window_size >= seq_len`` it degenerates to full
  attention, which the equivalence test relies on.
- ``local_flex`` — uses ``torch.nn.attention.flex_attention`` with a block mask
  so out-of-window blocks are skipped entirely (true sparsity, lower memory at
  long sequences). Falls back to the banded path when flex_attention is
  unavailable, when an explicit ``attn_mask`` is given, or when attention
  dropout is active during training.
"""
import torch

from .base import MaskType
from .core import ProjAttention, make_sliding_window_mask, sdpa_core
from .registry import register_attention

_FULL = 1 << 30  # sentinel "window": ≥ any real seq_len, so the band becomes full attention

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    _HAS_FLEX = True
except Exception:  # pragma: no cover - older torch
    create_block_mask = flex_attention = None
    _HAS_FLEX = False

# flex_attention only realizes its block-sparse memory/speed benefit when
# compiled; the eager path materializes the full attention. Compile once on
# first CUDA use. On CPU we run eager (correct, used by the test suite).
_compiled_flex = None


def _flex_call(q, k, v, block_mask):
    global _compiled_flex
    if q.is_cuda:                                   # the fused/compiled kernel is the whole point on GPU
        if _compiled_flex is None:
            _compiled_flex = torch.compile(flex_attention)  # compile once; cached across calls
        return _compiled_flex(q, k, v, block_mask=block_mask)
    return flex_attention(q, k, v, block_mask=block_mask)   # CPU: eager is correct (used by tests)


def _banded(q, k, v, window, is_causal, attn_mask, dropout_p):
    # Reference local attention: build the [S_q, S_k] window mask, then run the
    # ordinary softmax kernel. Correct but still O(S²) memory (mask is dense).
    keep = make_sliding_window_mask(q.shape[-2], k.shape[-2], window, q.device, causal=is_causal)
    if attn_mask is not None:                       # AND in any extra (e.g. key-padding) mask
        extra = attn_mask if attn_mask.dtype == torch.bool else (attn_mask >= 0)
        keep = keep & extra
    return sdpa_core(q, k, v, attn_mask=keep, dropout_p=dropout_p)  # softmax(QKᵀ/√d on the band)·V


@register_attention("local")
class SlidingWindowAttention(ProjAttention):
    @classmethod
    def supports_mask(cls, mask_type: MaskType) -> bool:
        return mask_type in (MaskType.NONE, MaskType.CAUSAL, MaskType.PADDING, MaskType.SLIDING)

    def _attend(self, q, k, v, attn_mask, is_causal):
        k, v = self._maybe_repeat_kv(k, v)              # broadcast KV heads (GQA/MQA) up to H
        window = self.cfg.window_size or _FULL          # no window configured -> behave like full attention
        dropout_p = self.dropout_p if self.training else 0.0
        return _banded(q, k, v, window, is_causal, attn_mask, dropout_p)


@register_attention("local_flex")
class FlexSlidingWindowAttention(ProjAttention):
    """Sliding-window attention via flex_attention's block mask (true sparsity)."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._mask_cache = {}

    @classmethod
    def supports_mask(cls, mask_type: MaskType) -> bool:
        return mask_type in (MaskType.NONE, MaskType.CAUSAL, MaskType.SLIDING)

    def _block_mask(self, seq_q, seq_k, window, is_causal, device):
        # A BlockMask tells flex_attention which (query-block, key-block) tiles can
        # be skipped entirely -> the source of the real sparsity savings. Building
        # it is non-trivial, so cache per (shapes, window, causal, device).
        key = (seq_q, seq_k, window, is_causal, str(device))
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        offset = seq_k - seq_q                          # align queries to the end of the key axis (KV-cache safe)

        def mask_mod(b, h, q_idx, kv_idx):
            # Predicate evaluated per (query, key): same window rule as the banded mask.
            qi = q_idx + offset                         # absolute query position
            in_window = (qi - kv_idx) < window          # key not older than `window` steps
            if is_causal:
                return (kv_idx <= qi) & in_window       # causal: also no future keys
            return in_window & ((kv_idx - qi) < window)  # non-causal: symmetric future band

        block_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=seq_q, KV_LEN=seq_k, device=device)
        self._mask_cache[key] = block_mask
        return block_mask

    def _attend(self, q, k, v, attn_mask, is_causal):
        k, v = self._maybe_repeat_kv(k, v)              # broadcast KV heads (GQA/MQA) up to H
        window = self.cfg.window_size or _FULL
        dropout_p = self.dropout_p if self.training else 0.0

        # flex can't express an arbitrary additive mask or attention dropout here,
        # so only take the sparse path when neither is requested.
        use_flex = _HAS_FLEX and attn_mask is None and dropout_p == 0.0
        if not use_flex:
            return _banded(q, k, v, window, is_causal, attn_mask, dropout_p)

        try:
            block_mask = self._block_mask(q.shape[-2], k.shape[-2], window, is_causal, q.device)
            # .contiguous(): flex_attention wants contiguous q/k/v (post-transpose they may not be).
            return _flex_call(q.contiguous(), k.contiguous(), v.contiguous(), block_mask)
        except Exception:  # noqa: BLE001 - never fail; fall back to the correct banded path
            return _banded(q, k, v, window, is_causal, attn_mask, dropout_p)
