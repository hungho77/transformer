"""Linear attention: replace softmax with a feature map for O(S · d²) cost.

Softmax attention costs O(S²·d) because it forms the S×S score matrix. Linear
attention (Katharopoulos et al., 2020) replaces exp(qᵢ·kⱼ) with a kernel
φ(qᵢ)·φ(kⱼ) using a feature map φ(x) = elu(x) + 1 (≥ 0). Associativity then lets
us contract K with V *first*:

    non-causal:  outᵢ = φ(qᵢ) · (Σⱼ φ(kⱼ) vⱼᵀ)  /  (φ(qᵢ) · Σⱼ φ(kⱼ))
    causal:      same, but the sums run over j ≤ i (a running prefix state)

No S×S matrix is ever materialized, so cost/memory scale linearly in S.
"""
import torch
import torch.nn.functional as F

from .base import MaskType
from .core import ProjAttention
from .registry import register_attention

_EPS = 1e-6  # guards the denominator (φ(q)·Σφ(k)) against divide-by-zero


def _phi(x):
    return F.elu(x) + 1.0  # feature map φ: elu+1 is smooth and strictly positive (a valid similarity kernel)


@register_attention("linear")
class LinearAttention(ProjAttention):
    @classmethod
    def supports_mask(cls, mask_type: MaskType) -> bool:
        # The KV-contraction trick can't apply an arbitrary S×S mask; only
        # causality (via prefix sums) and key-padding (zeroing φ(k)) are expressible.
        return mask_type in (MaskType.NONE, MaskType.CAUSAL, MaskType.PADDING)

    def _key_keep(self, attn_mask, ref):
        # Reduce any mask to a per-key keep flag: a key survives if any query may
        # attend to it. Shape -> [.., S_k, 1] so it broadcasts over heads and d.
        m = attn_mask if attn_mask.dtype == torch.bool else (attn_mask >= 0)  # float mask: ≥0 means keep
        while m.dim() < 4:
            m = m.unsqueeze(0)
        return m.any(dim=-2).unsqueeze(-1)

    def _attend(self, q, k, v, attn_mask, is_causal):
        k, v = self._maybe_repeat_kv(k, v)            # broadcast KV heads (GQA/MQA) up to H
        fq, fk = _phi(q), _phi(k)                      # φ(Q), φ(K): the positive feature maps
        if attn_mask is not None:
            fk = fk.masked_fill(~self._key_keep(attn_mask, fk), 0.0)  # padded keys contribute 0 to the KV sums

        if is_causal:
            # Causal: each position needs sums over j ≤ i, so accumulate a running
            # state with cumsum along the sequence axis (dim=2).
            kv = torch.einsum("bhsd,bhse->bhsde", fk, v).cumsum(dim=2)  # Sᵢ = Σ_{j≤i} φ(kⱼ) vⱼᵀ  (running KV)
            z = fk.cumsum(dim=2)                                        # zᵢ = Σ_{j≤i} φ(kⱼ)        (running normalizer)
            num = torch.einsum("bhsd,bhsde->bhse", fq, kv)             # numerator   φ(qᵢ)·Sᵢ
            den = torch.einsum("bhsd,bhsd->bhs", fq, z).unsqueeze(-1)  # denominator φ(qᵢ)·zᵢ
        else:
            # Non-causal: one global KV summary shared by every query (O(S·d²)).
            kv = torch.einsum("bhsd,bhse->bhde", fk, v)               # S = Σⱼ φ(kⱼ) vⱼᵀ   [B,H,d,d]
            z = fk.sum(dim=2)                                          # z = Σⱼ φ(kⱼ)       [B,H,d]
            num = torch.einsum("bhsd,bhde->bhse", fq, kv)             # numerator   φ(qᵢ)·S
            den = torch.einsum("bhsd,bhd->bhs", fq, z).unsqueeze(-1)  # denominator φ(qᵢ)·z
        return num / (den + _EPS)                     # normalize -> weighted average of values
