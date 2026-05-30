"""Encoder-only transformer (BERT-style) with a masked-LM head.

Bidirectional self-attention (never causal). Reuses the same TransformerBlock as
the other models; only the embeddings (token + position + segment), the MLM head,
and the optional [CLS] pooler are BERT-specific.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import BlockConfig, TokenEmbedding, build_norm, make_blocks
from .base import BaseModel
from .configs import BERTConfig


class BertMLMHead(nn.Module):
    def __init__(self, dim, vocab_size, norm_type):
        super().__init__()
        self.dense = nn.Linear(dim, dim)
        self.norm = build_norm(norm_type, dim)
        self.decoder = nn.Linear(dim, vocab_size)

    def forward(self, x):
        return self.decoder(self.norm(F.gelu(self.dense(x))))


class BERT(BaseModel):
    def __init__(self, cfg: BERTConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = TokenEmbedding(cfg.vocab_size, cfg.dim)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.dim)
        self.tok_type_emb = nn.Embedding(cfg.num_token_types, cfg.dim) if cfg.num_token_types > 0 else None
        self.emb_norm = build_norm(cfg.norm_type, cfg.dim)
        self.emb_drop = nn.Dropout(cfg.dropout)

        block_cfg = BlockConfig(
            dim=cfg.dim, num_heads=cfg.num_heads, num_kv_heads=cfg.num_kv_heads, head_dim=cfg.head_dim,
            attention_name=cfg.attention_name if isinstance(cfg.attention_name, str) else "mha",
            ffn_type=cfg.ffn_type, ffn_mult=cfg.ffn_mult, norm_type=cfg.norm_type,
            pre_norm=cfg.pre_norm, dropout=cfg.dropout, bias=cfg.bias, use_rotary=False,
        )
        self.encoder = make_blocks(block_cfg, cfg.n_layers, attention_name=cfg.attention_name)
        self.norm_f = build_norm(cfg.norm_type, cfg.dim)

        self.mlm_head = BertMLMHead(cfg.dim, cfg.vocab_size, cfg.norm_type)
        self.pooler = nn.Linear(cfg.dim, cfg.dim)
        if cfg.tie_weights:
            self.mlm_head.decoder.weight = self.tok_emb.emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _embed(self, input_ids, token_type_ids):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)[None, :, :]
        if self.tok_type_emb is not None:
            if token_type_ids is None:
                token_type_ids = torch.zeros_like(input_ids)
            x = x + self.tok_type_emb(token_type_ids)
        return self.emb_drop(self.emb_norm(x))

    def encode(self, input_ids, token_type_ids=None, attention_mask=None):
        # attention_mask: [B, S] with 1 = keep, 0 = pad -> bool [B, 1, 1, S]
        mask = attention_mask[:, None, None, :].bool() if attention_mask is not None else None
        x = self._embed(input_ids, token_type_ids)
        for block in self.encoder:
            x = block(x, is_causal=False, attn_mask=mask)
        return self.norm_f(x)

    def pooled_output(self, sequence_output):
        return torch.tanh(self.pooler(sequence_output[:, 0]))

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, mlm_labels=None):
        sequence_output = self.encode(input_ids, token_type_ids, attention_mask)
        if mlm_labels is not None:
            logits = self.mlm_head(sequence_output)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), mlm_labels.reshape(-1), ignore_index=-100
            )
            return logits, loss
        return sequence_output, None
