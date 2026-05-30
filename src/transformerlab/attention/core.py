"""Shared attention machinery: projections, head reshaping, and the SDP kernel.

Mask convention (matches torch.nn.functional.scaled_dot_product_attention):
a boolean mask value of ``True`` means the position IS allowed to participate.
Float masks are additive (added to the pre-softmax scores).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import AttentionBase, AttentionConfig, KVCache


def make_causal_mask(seq_q: int, seq_k: int, device) -> Tensor:
    """Lower-triangular boolean mask [seq_q, seq_k]; True = allowed (keep).

    Aligned to the *end* of the key sequence so it also works when seq_k > seq_q
    (e.g. incremental decoding with a KV cache).
    """
    offset = seq_k - seq_q
    q_idx = torch.arange(seq_q, device=device).unsqueeze(1)
    k_idx = torch.arange(seq_k, device=device).unsqueeze(0)
    return k_idx <= (q_idx + offset)


def make_sliding_window_mask(seq_q: int, seq_k: int, window: int, device, causal: bool = True) -> Tensor:
    """Banded boolean mask [seq_q, seq_k]; True = allowed.

    Each query attends to keys within ``window`` steps to the left (and, if not
    causal, ``window`` steps to the right).
    """
    offset = seq_k - seq_q
    q_idx = torch.arange(seq_q, device=device).unsqueeze(1) + offset
    k_idx = torch.arange(seq_k, device=device).unsqueeze(0)
    left = k_idx >= (q_idx - window + 1)
    if causal:
        right = k_idx <= q_idx
    else:
        right = k_idx <= (q_idx + window - 1)
    return left & right


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """[B, n_kv, S, d] -> [B, n_kv * n_rep, S, d] (broadcast KV heads to all heads)."""
    if n_rep == 1:
        return x
    b, n_kv, s, d = x.shape
    return (
        x[:, :, None, :, :]
        .expand(b, n_kv, n_rep, s, d)
        .reshape(b, n_kv * n_rep, s, d)
    )


def sdpa_core(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> Tensor:
    """Explicit scaled-dot-product attention. q/k/v: [B, H, S, d]. Reference path."""
    scale = scale or (1.0 / math.sqrt(q.shape[-1]))
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if is_causal:
        keep = make_causal_mask(q.shape[-2], k.shape[-2], q.device)
        scores = scores.masked_fill(~keep, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask
    attn = torch.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)
    return torch.matmul(attn, v)


class ProjAttention(AttentionBase):
    """Base for variants that share q/k/v/o projections and head reshaping.

    Subclasses implement ``_attend(q, k, v, attn_mask, is_causal)`` with q/k/v in
    [B, H, S, head_dim] layout (KV heads already broadcast to H), returning the
    same layout.
    """

    def __init__(self, cfg: AttentionConfig):
        super().__init__(cfg)
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.resolved_kv_heads()
        self.head_dim = cfg.resolved_head_dim()
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.n_rep = self.num_heads // self.num_kv_heads
        self.dropout_p = cfg.dropout

        q_out = self.num_heads * self.head_dim
        kv_out = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(cfg.dim, q_out, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.dim, kv_out, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.dim, kv_out, bias=cfg.bias)
        self.o_proj = nn.Linear(q_out, cfg.dim, bias=cfg.bias)

    def _shape(self, t: Tensor, n_heads: int) -> Tensor:
        b, s, _ = t.shape
        return t.view(b, s, n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, *, kv=None, attn_mask=None, is_causal=False, rotary=None, kv_cache: Optional[KVCache] = None):
        b, s_q, _ = x.shape
        src = x if kv is None else kv

        q = self._shape(self.q_proj(x), self.num_heads)
        k = self._shape(self.k_proj(src), self.num_kv_heads)
        v = self._shape(self.v_proj(src), self.num_kv_heads)

        if rotary is not None:
            offset = kv_cache.seq_len if kv_cache is not None else 0
            q, k = rotary(q, k, offset=offset)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        out = self._attend(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        out = out.transpose(1, 2).reshape(b, s_q, self.num_heads * self.head_dim)
        return self.o_proj(out)

    def _maybe_repeat_kv(self, k: Tensor, v: Tensor):
        if self.n_rep > 1:
            k = repeat_kv(k, self.n_rep)
            v = repeat_kv(v, self.n_rep)
        return k, v

    def _attend(self, q, k, v, attn_mask, is_causal):  # pragma: no cover - abstract
        raise NotImplementedError
