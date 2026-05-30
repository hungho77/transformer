"""FlashAttention backend (optional).

Routes to the ``flash_attn`` package when it is installed and the inputs are on
CUDA in fp16/bf16; otherwise falls back to torch's fused SDPA so the variant is
always usable. flash-attn is an optional dependency (``pip install .[flash]``).
"""
import warnings

import torch
import torch.nn.functional as F

from .core import ProjAttention
from .registry import register_attention

try:  # pragma: no cover - depends on optional CUDA package
    from flash_attn import flash_attn_func

    _HAS_FLASH = True
except Exception:  # noqa: BLE001
    flash_attn_func = None
    _HAS_FLASH = False

_warned = False


@register_attention("flash")
class FlashAttention(ProjAttention):
    def _attend(self, q, k, v, attn_mask, is_causal):
        global _warned
        k, v = self._maybe_repeat_kv(k, v)
        dropout_p = self.dropout_p if self.training else 0.0
        usable = (
            _HAS_FLASH
            and q.is_cuda
            and q.dtype in (torch.float16, torch.bfloat16)
            and attn_mask is None
        )
        if usable:
            # flash_attn expects [B, S, H, d].
            out = flash_attn_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                dropout_p=dropout_p, causal=is_causal,
            )
            return out.transpose(1, 2)

        if not _warned and not _HAS_FLASH:
            warnings.warn("flash-attn not available; FlashAttention falls back to torch SDPA.")
            _warned = True
        if attn_mask is not None:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=is_causal)
