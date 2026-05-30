"""Generic transformer block, parameterized over attention/norm/ffn choices.

The block is architecture-agnostic: GPT, ViT, and the seq2seq encoder/decoder
all stack the same block, differing only in config (causal vs not, cross-attn,
norm/ffn type, and crucially ``attention_name`` — which selects the attention
variant from the registry).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch.nn as nn

from ..attention import AttentionConfig, build_attention
from .feedforward import build_ffn
from .norm import build_norm


@dataclass
class BlockConfig:
    dim: int
    num_heads: int
    num_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    attention_name: str = "mha"
    ffn_type: str = "mlp"
    ffn_mult: float = 4.0
    ffn_hidden: Optional[int] = None
    norm_type: str = "rms"
    pre_norm: bool = True
    dropout: float = 0.0
    bias: bool = False
    window_size: Optional[int] = None
    use_rotary: bool = False
    cross_attention: bool = False
    extra: dict = field(default_factory=dict)

    def attention_config(self) -> AttentionConfig:
        return AttentionConfig(
            dim=self.dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            dropout=self.dropout,
            bias=self.bias,
            head_dim=self.head_dim,
            window_size=self.window_size,
            use_rotary=self.use_rotary,
            extra=dict(self.extra),
        )

    def hidden_dim(self) -> int:
        return self.ffn_hidden if self.ffn_hidden is not None else int(self.ffn_mult * self.dim)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: BlockConfig):
        super().__init__()
        self.cfg = cfg
        self.pre_norm = cfg.pre_norm

        self.norm1 = build_norm(cfg.norm_type, cfg.dim)
        self.attn = build_attention(cfg.attention_name, cfg.attention_config())

        self.cross_attn = None
        if cfg.cross_attention:
            self.norm_cross = build_norm(cfg.norm_type, cfg.dim)
            self.cross_attn = build_attention(cfg.attention_name, cfg.attention_config())

        self.norm2 = build_norm(cfg.norm_type, cfg.dim)
        self.ffn = build_ffn(cfg.ffn_type, cfg.dim, cfg.hidden_dim(), dropout=cfg.dropout, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def _sublayer(self, x, norm, fn):
        """Pre-norm: x + fn(norm(x)). Post-norm: norm(x + fn(x))."""
        if self.pre_norm:
            return x + self.dropout(fn(norm(x)))
        return norm(x + self.dropout(fn(x)))

    def forward(self, x, *, attn_mask=None, is_causal=False, rotary=None, kv_cache=None,
                memory=None, memory_mask=None):
        x = self._sublayer(
            x, self.norm1,
            lambda h: self.attn(h, attn_mask=attn_mask, is_causal=is_causal, rotary=rotary, kv_cache=kv_cache),
        )
        if self.cross_attn is not None:
            x = self._sublayer(
                x, self.norm_cross,
                lambda h: self.cross_attn(h, kv=memory, attn_mask=memory_mask),
            )
        x = self._sublayer(x, self.norm2, self.ffn)
        return x


def make_blocks(base_cfg: BlockConfig, n_layers: int, attention_name=None) -> nn.ModuleList:
    """Build a stack of blocks. ``attention_name`` may be a single name (applied
    to all layers) or a per-layer list, enabling heterogeneous-attention models."""
    names = attention_name if attention_name is not None else base_cfg.attention_name
    if isinstance(names, str):
        names = [names] * n_layers
    if len(names) != n_layers:
        raise ValueError(f"attention_name list length {len(names)} != n_layers {n_layers}")
    blocks = []
    for name in names:
        cfg = BlockConfig(**{**base_cfg.__dict__, "attention_name": name})
        blocks.append(TransformerBlock(cfg))
    return nn.ModuleList(blocks)
