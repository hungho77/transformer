"""Core attention contract: config, mask types, KV cache, and the base module.

Every attention variant is a *full self/cross-attention sublayer* that owns its
own q/k/v/output projections. This is required so that variants which change the
projection shape (MQA/GQA reduce KV heads; linear attention applies a feature
map) can express that difference internally, while models stay agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class MaskType(Enum):
    NONE = "none"
    CAUSAL = "causal"        # autoregressive (query i attends to keys <= i)
    PADDING = "padding"      # key-padding mask (bool [B, S_k])
    FULL = "full"            # arbitrary [B, 1|H, S_q, S_k] mask
    SLIDING = "sliding"      # local / windowed


@dataclass
class AttentionConfig:
    dim: int
    num_heads: int
    num_kv_heads: Optional[int] = None   # None -> MHA; 1 -> MQA; g -> GQA
    dropout: float = 0.0
    bias: bool = False
    head_dim: Optional[int] = None       # defaults to dim // num_heads
    window_size: Optional[int] = None    # for sliding-window/local attention
    use_rotary: bool = False
    extra: dict = field(default_factory=dict)

    def resolved_head_dim(self) -> int:
        return self.head_dim if self.head_dim is not None else self.dim // self.num_heads

    def resolved_kv_heads(self) -> int:
        return self.num_kv_heads if self.num_kv_heads is not None else self.num_heads


class KVCache:
    """Minimal incremental KV cache for autoregressive decoding.

    Stores keys/values as [B, n_kv_heads, S, head_dim] and appends on each step.
    """

    def __init__(self):
        self.k: Optional[Tensor] = None
        self.v: Optional[Tensor] = None

    @property
    def seq_len(self) -> int:
        return 0 if self.k is None else self.k.shape[2]

    def update(self, k: Tensor, v: Tensor):
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)
        return self.k, self.v


class AttentionBase(nn.Module):
    """Interface implemented by every registered attention variant."""

    name: str = "base"

    def __init__(self, cfg: AttentionConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        x: Tensor,
        *,
        kv: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        rotary=None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tensor:
        raise NotImplementedError

    @classmethod
    def supports_mask(cls, mask_type: MaskType) -> bool:
        """Whether this variant can honor a given mask type. Default: all."""
        return True
