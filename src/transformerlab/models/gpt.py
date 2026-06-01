"""Decoder-only causal language model (GPT-style)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from ..attention import KVCache
from ..layers import BlockConfig, LearnedPositionalEmbedding, RotaryEmbedding, TokenEmbedding, build_norm, make_blocks
from .base import BaseModel
from .configs import GPTConfig


class GPT(BaseModel):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        head_dim = cfg.head_dim or (cfg.dim // cfg.num_heads)

        self.tok_emb = TokenEmbedding(cfg.vocab_size, cfg.dim)
        self.pos_emb = None if cfg.use_rotary else LearnedPositionalEmbedding(cfg.max_seq_len, cfg.dim)
        self.rotary = RotaryEmbedding(head_dim, max_seq_len=cfg.max_seq_len) if cfg.use_rotary else None
        self.drop = nn.Dropout(cfg.dropout)

        block_cfg = BlockConfig(
            dim=cfg.dim, num_heads=cfg.num_heads, num_kv_heads=cfg.num_kv_heads, head_dim=cfg.head_dim,
            attention_name=cfg.attention_name if isinstance(cfg.attention_name, str) else "mha",
            ffn_type=cfg.ffn_type, ffn_mult=cfg.ffn_mult, norm_type=cfg.norm_type,
            dropout=cfg.dropout, bias=cfg.bias, window_size=cfg.window_size, use_rotary=cfg.use_rotary,
            extra=dict(cfg.extra),  # variant-specific knobs (e.g. MLA latent dims) flow to AttentionConfig
        )
        self.blocks = make_blocks(block_cfg, cfg.n_layers, attention_name=cfg.attention_name)
        self.norm_f = build_norm(cfg.norm_type, cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = self.pos_emb(x)
        x = self.drop(x)
        # Gradient checkpointing: in training, don't keep each block's activations;
        # recompute them in the backward pass instead. Trades ~30% compute for a
        # large activation-memory cut (enables longer context / deeper models).
        ckpt = self.cfg.grad_checkpoint and self.training
        for block in self.blocks:
            if ckpt:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, is_causal=True, rotary=self.rotary, use_reentrant=False)
            else:
                x = block(x, is_causal=True, rotary=self.rotary)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        return logits, loss

    def _forward_cached(self, idx, caches, is_causal):
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            offset = caches[0].seq_len
            positions = torch.arange(offset, offset + idx.shape[1], device=idx.device)
            x = x + self.pos_emb.pos(positions)[None, :, :]
        x = self.drop(x)
        for block, cache in zip(self.blocks, caches):
            x = block(x, is_causal=is_causal, rotary=self.rotary, kv_cache=cache)
        return self.lm_head(self.norm_f(x))

    def _sample(self, logits, temperature, top_k):
        logits = logits / max(temperature, 1e-6)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k=None, use_cache: bool = True):
        """Autoregressive sampling. With ``use_cache`` (default) the KV cache
        avoids recomputing attention over the whole prefix each step."""
        self.eval()
        if not use_cache:
            for _ in range(max_new_tokens):
                logits, _ = self(idx[:, -self.cfg.max_seq_len:])
                idx = torch.cat([idx, self._sample(logits[:, -1, :], temperature, top_k)], dim=1)
            return idx

        caches = [KVCache() for _ in self.blocks]
        logits = self._forward_cached(idx, caches, is_causal=True)[:, -1, :]
        for _ in range(max_new_tokens):
            next_id = self._sample(logits, temperature, top_k)
            idx = torch.cat([idx, next_id], dim=1)
            logits = self._forward_cached(next_id, caches, is_causal=False)[:, -1, :]
        return idx
