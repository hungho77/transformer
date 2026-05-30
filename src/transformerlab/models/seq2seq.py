"""Encoder-decoder transformer (Vaswani et al., 2017 / T5-style)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import BlockConfig, LearnedPositionalEmbedding, TokenEmbedding, build_norm, make_blocks
from .base import BaseModel
from .configs import Seq2SeqConfig


class EncoderDecoder(BaseModel):
    def __init__(self, cfg: Seq2SeqConfig):
        super().__init__()
        self.cfg = cfg

        self.src_emb = TokenEmbedding(cfg.src_vocab_size, cfg.dim)
        self.tgt_emb = TokenEmbedding(cfg.tgt_vocab_size, cfg.dim)
        self.src_pos = LearnedPositionalEmbedding(cfg.max_seq_len, cfg.dim)
        self.tgt_pos = LearnedPositionalEmbedding(cfg.max_seq_len, cfg.dim)

        attn = cfg.attention_name if isinstance(cfg.attention_name, str) else "mha"
        common = dict(
            dim=cfg.dim, num_heads=cfg.num_heads, num_kv_heads=cfg.num_kv_heads, head_dim=cfg.head_dim,
            attention_name=attn, ffn_type=cfg.ffn_type, ffn_mult=cfg.ffn_mult,
            norm_type=cfg.norm_type, dropout=cfg.dropout, bias=cfg.bias, use_rotary=False,
        )
        self.encoder = make_blocks(BlockConfig(cross_attention=False, **common),
                                   cfg.n_encoder_layers, attention_name=cfg.attention_name)
        self.decoder = make_blocks(BlockConfig(cross_attention=True, **common),
                                   cfg.n_decoder_layers, attention_name=cfg.attention_name)

        self.norm_enc = build_norm(cfg.norm_type, cfg.dim)
        self.norm_dec = build_norm(cfg.norm_type, cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.tgt_vocab_size, bias=False)

    def encode(self, src, src_mask=None):
        x = self.src_pos(self.src_emb(src))
        for block in self.encoder:
            x = block(x, is_causal=False, attn_mask=src_mask)
        return self.norm_enc(x)

    def decode(self, tgt, memory, memory_mask=None):
        x = self.tgt_pos(self.tgt_emb(tgt))
        for block in self.decoder:
            x = block(x, is_causal=True, memory=memory, memory_mask=memory_mask)
        return self.norm_dec(x)

    def forward(self, src, tgt_in, targets=None, src_mask=None, memory_mask=None):
        memory = self.encode(src, src_mask=src_mask)
        dec = self.decode(tgt_in, memory, memory_mask=memory_mask)
        logits = self.head(dec)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=self.cfg.pad_id
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, src, bos_id: int, eos_id: int, max_len: int = 64):
        self.eval()
        memory = self.encode(src)
        tgt = torch.full((src.shape[0], 1), bos_id, dtype=torch.long, device=src.device)
        for _ in range(max_len):
            logits = self.head(self.decode(tgt, memory))
            next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_id], dim=1)
            if (next_id == eos_id).all():
                break
        return tgt
