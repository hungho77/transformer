"""Linear attention: replace softmax with a feature map for O(S * d^2) cost.

Uses phi(x) = elu(x) + 1 (Katharopoulos et al., 2020). The non-causal form
contracts keys/values first (KV summary), the causal form keeps a running KV
state via cumulative sums. No S x S score matrix is ever materialized.
"""
import torch
import torch.nn.functional as F

from .base import MaskType
from .core import ProjAttention
from .registry import register_attention

_EPS = 1e-6


def _phi(x):
    return F.elu(x) + 1.0


@register_attention("linear")
class LinearAttention(ProjAttention):
    @classmethod
    def supports_mask(cls, mask_type: MaskType) -> bool:
        # No arbitrary score masking; causal + key-padding only.
        return mask_type in (MaskType.NONE, MaskType.CAUSAL, MaskType.PADDING)

    def _key_keep(self, attn_mask, ref):
        m = attn_mask if attn_mask.dtype == torch.bool else (attn_mask >= 0)
        while m.dim() < 4:
            m = m.unsqueeze(0)
        return m.any(dim=-2).unsqueeze(-1)  # [.., S_k, 1], broadcasts over heads/dim

    def _attend(self, q, k, v, attn_mask, is_causal):
        k, v = self._maybe_repeat_kv(k, v)
        fq, fk = _phi(q), _phi(k)
        if attn_mask is not None:
            fk = fk.masked_fill(~self._key_keep(attn_mask, fk), 0.0)

        if is_causal:
            kv = torch.einsum("bhsd,bhse->bhsde", fk, v).cumsum(dim=2)
            z = fk.cumsum(dim=2)
            num = torch.einsum("bhsd,bhsde->bhse", fq, kv)
            den = torch.einsum("bhsd,bhsd->bhs", fq, z).unsqueeze(-1)
        else:
            kv = torch.einsum("bhsd,bhse->bhde", fk, v)
            z = fk.sum(dim=2)
            num = torch.einsum("bhsd,bhde->bhse", fq, kv)
            den = torch.einsum("bhsd,bhd->bhs", fq, z).unsqueeze(-1)
        return num / (den + _EPS)
