"""Vision Transformer (Dosovitskiy et al., 2020)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import BlockConfig, PatchEmbedding, build_norm, make_blocks
from .base import BaseModel
from .configs import ViTConfig


class VisionTransformer(BaseModel):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbedding(cfg.image_size, cfg.patch_size, cfg.in_channels, cfg.dim)
        n_patches = self.patch_embed.num_patches

        self.use_cls = cfg.pool == "cls"
        n_tokens = n_patches + (1 if self.use_cls else 0)
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, cfg.dim))
        self.drop = nn.Dropout(cfg.dropout)

        block_cfg = BlockConfig(
            dim=cfg.dim, num_heads=cfg.num_heads, num_kv_heads=cfg.num_kv_heads, head_dim=cfg.head_dim,
            attention_name=cfg.attention_name if isinstance(cfg.attention_name, str) else "mha",
            ffn_type=cfg.ffn_type, ffn_mult=cfg.ffn_mult, norm_type=cfg.norm_type,
            dropout=cfg.dropout, bias=cfg.bias, use_rotary=False,
        )
        self.blocks = make_blocks(block_cfg, cfg.n_layers, attention_name=cfg.attention_name)
        self.norm = build_norm(cfg.norm_type, cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.use_cls:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, images, targets=None):
        x = self.patch_embed(images)
        if self.use_cls:
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)
        x = self.drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x, is_causal=False)
        x = self.norm(x)
        pooled = x[:, 0] if self.use_cls else x.mean(dim=1)
        logits = self.head(pooled)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss
