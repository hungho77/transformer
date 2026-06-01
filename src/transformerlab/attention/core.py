"""Shared attention machinery: projections, head reshaping, and the SDP kernel.

The scaled-dot-product attention this module implements is:

    Attention(Q, K, V) = softmax( (Q Kᵀ) / √d_k + M ) V

where Q,K,V are [B, H, S, d_k] (batch, heads, sequence, head-dim) and M is an
additive mask (0 where allowed, -inf where disallowed).

Mask convention (matches torch.nn.functional.scaled_dot_product_attention):
a boolean mask value of ``True`` means the position IS allowed to participate;
a float mask is added directly to the pre-softmax scores.
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

    Causality rule: query position i may attend to key position j iff j ≤ i, so
    a token never sees the future. We align queries to the *end* of the key axis
    (``offset = seq_k - seq_q``) so this stays correct when seq_k > seq_q, i.e.
    during incremental decoding where K/V hold the whole past but Q is just the
    new token(s).
    """
    offset = seq_k - seq_q                                  # absolute index of query row 0 within the key axis
    q_idx = torch.arange(seq_q, device=device).unsqueeze(1)  # column vector [seq_q, 1]: query rows
    k_idx = torch.arange(seq_k, device=device).unsqueeze(0)  # row vector   [1, seq_k]: key columns
    return k_idx <= (q_idx + offset)                        # keep[i, j] = (j ≤ i + offset)  -> lower-triangular


def make_sliding_window_mask(seq_q: int, seq_k: int, window: int, device, causal: bool = True) -> Tensor:
    """Banded boolean mask [seq_q, seq_k]; True = allowed.

    Local attention: query i only sees keys within ``window`` steps. Causal keeps
    the past window [i-window+1, i]; non-causal keeps a symmetric band of half
    width ``window``. This bounds the cost from O(S²) toward O(S·window).
    """
    offset = seq_k - seq_q                                          # align queries to the end of the key axis
    q_idx = torch.arange(seq_q, device=device).unsqueeze(1) + offset  # absolute query positions [seq_q, 1]
    k_idx = torch.arange(seq_k, device=device).unsqueeze(0)         # key positions [1, seq_k]
    left = k_idx >= (q_idx - window + 1)                            # j ≥ i-window+1  -> not older than the window
    if causal:
        right = k_idx <= q_idx                                     # j ≤ i           -> no peeking ahead
    else:
        right = k_idx <= (q_idx + window - 1)                       # j ≤ i+window-1  -> symmetric future band
    return left & right                                            # keep keys inside the band


def make_sink_window_mask(seq_q: int, seq_k: int, window: int, sink_size: int,
                          device, causal: bool = True) -> Tensor:
    """Banded mask + attention sinks [seq_q, seq_k]; True = allowed (StreamingLLM).

    StreamingLLM (Xiao et al., 2023) keeps the recent window *and* the first
    ``sink_size`` "sink" tokens, which softmax dumps probability mass onto:

        keep[i, j] = (j < sink_size)            # always attend to the sink tokens
                     OR (i-window+1 ≤ j ≤ i)     # plus the sliding window

    intersected with causality (j ≤ i) so no token sees the future. Keeping the
    sinks is what lets a *fixed* cache (sinks + last ``window``) hold quality as
    the stream grows past the window — pure sliding window collapses without them.
    """
    keep = make_sliding_window_mask(seq_q, seq_k, window, device, causal=causal)  # recent window band
    k_idx = torch.arange(seq_k, device=device).unsqueeze(0)        # key positions [1, seq_k]
    sink_cols = k_idx < sink_size                                  # j < sink_size -> a sink token
    if causal:
        offset = seq_k - seq_q                                     # align queries to the end of the key axis
        q_idx = torch.arange(seq_q, device=device).unsqueeze(1) + offset  # absolute query positions [seq_q, 1]
        sink_cols = sink_cols & (k_idx <= q_idx)                   # sinks still obey causality (j ≤ i)
    return keep | sink_cols                                        # union: window ∪ (causal) sinks


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """[B, n_kv, S, d] -> [B, n_kv * n_rep, S, d].

    Grouped/Multi-query attention uses fewer KV heads than query heads; here each
    KV head is replicated ``n_rep`` times so every query head has a K/V to match.
    expand() is a view (no copy); reshape() then lays the repeats out as heads.
    """
    if n_rep == 1:                                          # MHA case: KV heads already == query heads
        return x
    b, n_kv, s, d = x.shape
    return (
        x[:, :, None, :, :]                                 # [B, n_kv, 1, S, d]: add a "repeat" axis
        .expand(b, n_kv, n_rep, s, d)                       # broadcast it to n_rep (no memory copy)
        .reshape(b, n_kv * n_rep, s, d)                     # fold (n_kv, n_rep) into a single head axis
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
    """Explicit scaled-dot-product attention. q/k/v: [B, H, S, d]. Reference path.

    Computes softmax((QKᵀ)/√d_k + mask) V term by term so the math is auditable;
    the ``sdpa`` variant swaps this for the fused kernel and must match it.
    """
    scale = scale or (1.0 / math.sqrt(q.shape[-1]))         # 1/√d_k: keeps score variance ~1 (softmax won't saturate)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale   # QKᵀ/√d_k -> [B, H, S_q, S_k] similarity logits
    if is_causal:
        keep = make_causal_mask(q.shape[-2], k.shape[-2], q.device)  # True where query may see key
        scores = scores.masked_fill(~keep, float("-inf"))  # disallowed -> -inf  (softmax sends e^-inf -> 0)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))   # bool mask: False positions -> -inf
        else:
            scores = scores + attn_mask                    # float mask: additive bias on the logits
    attn = torch.softmax(scores, dim=-1)                    # normalize over keys: rows sum to 1 (attention weights)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)                 # regularize the attention distribution
    return torch.matmul(attn, v)                            # weighted sum of values: Σ_j a_ij v_j -> [B, H, S_q, d]


class ProjAttention(AttentionBase):
    """Base for variants that share q/k/v/o projections and head reshaping.

    Projects the input into per-head queries/keys/values, lets a subclass run the
    actual attention via ``_attend(q, k, v, attn_mask, is_causal)`` on
    [B, H, S, head_dim] tensors (KV heads already broadcast to H), then merges
    heads and applies the output projection W_O. Centralizing this here is what
    lets every variant differ only in the kernel, not the plumbing.
    """

    def __init__(self, cfg: AttentionConfig):
        super().__init__(cfg)
        self.num_heads = cfg.num_heads                      # H: number of query heads
        self.num_kv_heads = cfg.resolved_kv_heads()         # H_kv: KV heads (== H for MHA, <H for GQA, 1 for MQA)
        self.head_dim = cfg.resolved_head_dim()             # d_k: per-head dimension (dim // H by default)
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")  # each KV head must serve a whole group
        self.n_rep = self.num_heads // self.num_kv_heads    # how many query heads share one KV head
        self.dropout_p = cfg.dropout

        q_out = self.num_heads * self.head_dim              # W_Q maps dim -> H·d_k
        kv_out = self.num_kv_heads * self.head_dim          # W_K/W_V: dim -> H_kv·d_k (smaller for GQA/MQA = less KV)
        self.q_proj = nn.Linear(cfg.dim, q_out, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.dim, kv_out, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.dim, kv_out, bias=cfg.bias)
        self.o_proj = nn.Linear(q_out, cfg.dim, bias=cfg.bias)  # W_O maps the concatenated heads back to dim

    def _shape(self, t: Tensor, n_heads: int) -> Tensor:
        # [B, S, n_heads·d] -> [B, n_heads, S, d]: split the channel axis into heads
        # and move heads ahead of the sequence so attention is batched per head.
        b, s, _ = t.shape
        return t.view(b, s, n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, *, kv=None, attn_mask=None, is_causal=False, rotary=None, kv_cache: Optional[KVCache] = None):
        b, s_q, _ = x.shape
        src = x if kv is None else kv                       # self-attention reads K/V from x; cross-attention from kv

        q = self._shape(self.q_proj(x), self.num_heads)     # Q = x·W_Q -> [B, H, S_q, d]
        k = self._shape(self.k_proj(src), self.num_kv_heads)  # K = src·W_K -> [B, H_kv, S_k, d]
        v = self._shape(self.v_proj(src), self.num_kv_heads)  # V = src·W_V -> [B, H_kv, S_k, d]

        if rotary is not None:
            offset = kv_cache.seq_len if kv_cache is not None else 0  # new tokens start after the cached past
            q, k = rotary(q, k, offset=offset)              # rotate Q,K by position (RoPE) before scoring

        if kv_cache is not None:
            k, v = kv_cache.update(k, v)                    # append new K,V and read back the full past+present

        out = self._attend(q, k, v, attn_mask=attn_mask, is_causal=is_causal)  # variant-specific attention
        out = out.transpose(1, 2).reshape(b, s_q, self.num_heads * self.head_dim)  # concat heads -> [B, S_q, H·d]
        return self.o_proj(out)                             # project back to model dim

    def _maybe_repeat_kv(self, k: Tensor, v: Tensor):
        # GQA/MQA store fewer KV heads; broadcast them up to H so the kernel can
        # treat every query head uniformly. No-op for MHA (n_rep == 1).
        if self.n_rep > 1:
            k = repeat_kv(k, self.n_rep)
            v = repeat_kv(v, self.n_rep)
        return k, v

    def _attend(self, q, k, v, attn_mask, is_causal):  # pragma: no cover - abstract
        raise NotImplementedError                          # each variant supplies its own kernel here
