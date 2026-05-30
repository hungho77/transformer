"""Multi-head Latent Attention (MLA; DeepSeek-V2/V3).

MLA shrinks the KV cache — the real bottleneck for long-context inference — by
*not* caching per-head K/V. Instead it caches a small shared latent and
reconstructs K/V on the fly:

    c_KV = x W_DKV                      # [.., d_c]   compressed KV latent (small)
    K_C  = c_KV W_UK                    # per-head content keys   (recomputed from c_KV)
    V    = c_KV W_UV                    # per-head values         (recomputed from c_KV)

Because the up-projection W_UK would entangle position with content, RoPE is
*decoupled*: each query/key carries an extra rope-only slice that does the
positional work, and the key's rope slice is shared across heads:

    q = [q_C ; q_R]      k = [k_C ; k_R]         (k_R shared over heads)
    score_ij = q_C·k_C + q_R·k_R                 (then /√(d_nope+d_rope))

So only c_KV (d_c) and k_R (d_rope) are cached per token instead of H·d_h·2.

Config (via AttentionConfig.extra, all optional — sensible defaults derived from
head_dim): kv_latent_dim, q_latent_dim, nope_head_dim, rope_head_dim, v_head_dim.
Manages its own position, so use with use_rotary=True (GPT then adds no absolute
pos embedding); the ``rotary`` passed by the model is intentionally ignored.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AttentionBase, AttentionConfig, KVCache, MaskType
from .core import make_causal_mask
from .registry import register_attention


def _even(n: int) -> int:
    # RoPE rotates features in pairs, so the rope slice must be even (and ≥ 2).
    n -= n % 2
    return max(2, n)


def _rotate_half(x):
    # [x1, x2] -> [-x2, x1]: the 90° partner that makes x·cos + rot·sin a rotation.
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _rope_tables(seq_len, dim, device, dtype, base=10000.0):
    # cos(mθ), sin(mθ) for positions m∈[0,seq_len) and per-pair freq θ_i=base^(-2i/d).
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)   # absolute positions
    freqs = torch.outer(pos, inv_freq)                               # mθ -> [seq_len, dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)                          # duplicate to line up with rotate_half
    return emb.cos().to(dtype), emb.sin().to(dtype)                  # each [seq_len, dim]


def _apply_rope(x, cos, sin):
    # x[..., S, d] · cos[S, d] + rotate_half(x) · sin[S, d]  (cos/sin broadcast over B,H).
    return x * cos + _rotate_half(x) * sin


@register_attention("mla")
class MultiHeadLatentAttention(AttentionBase):
    @classmethod
    def supports_mask(cls, mask_type: MaskType) -> bool:
        # Forms explicit S×S scores, so any score-level mask works; not windowed.
        return mask_type in (MaskType.NONE, MaskType.CAUSAL, MaskType.PADDING, MaskType.FULL)

    def __init__(self, cfg: AttentionConfig):
        super().__init__(cfg)
        e = cfg.extra
        self.num_heads = cfg.num_heads
        head_dim = cfg.resolved_head_dim()                       # default per-head width = dim // num_heads
        self.nope = e.get("nope_head_dim", head_dim)             # content (no-position) slice of q/k
        self.rope = e.get("rope_head_dim", _even(head_dim // 2))  # decoupled positional slice
        self.v_head = e.get("v_head_dim", head_dim)              # value slice
        kv_lat = e.get("kv_latent_dim", max(head_dim, cfg.dim // 2))  # d_c: cached KV latent width
        q_lat = e.get("q_latent_dim", max(head_dim, cfg.dim // 2))    # query latent width
        self.dropout_p = cfg.dropout
        self.scale = 1.0 / math.sqrt(self.nope + self.rope)      # 1/√(d_nope+d_rope) for the joint score

        qk = self.nope + self.rope                               # per-head query/key width (content+rope)
        # Query path: x -> latent c_Q -> per-head [content | rope].
        self.q_down = nn.Linear(cfg.dim, q_lat, bias=cfg.bias)   # W_DQ
        self.q_norm = nn.LayerNorm(q_lat)                        # stabilize the latent (DeepSeek uses RMSNorm)
        self.q_up = nn.Linear(q_lat, self.num_heads * qk, bias=cfg.bias)  # W_UQ + W_QR fused
        # KV path: x -> latent c_KV (this is what gets cached) -> per-head [content key | value].
        self.kv_down = nn.Linear(cfg.dim, kv_lat, bias=cfg.bias)  # W_DKV
        self.kv_norm = nn.LayerNorm(kv_lat)
        self.kv_up = nn.Linear(kv_lat, self.num_heads * (self.nope + self.v_head), bias=cfg.bias)  # W_UK + W_UV
        # Shared (single-head) rope key, projected straight from x (decoupled from the latent).
        self.kr_proj = nn.Linear(cfg.dim, self.rope, bias=cfg.bias)  # W_KR
        self.o_proj = nn.Linear(self.num_heads * self.v_head, cfg.dim, bias=cfg.bias)  # W_O

    def forward(self, x, *, kv=None, attn_mask=None, is_causal=False, rotary=None, kv_cache: KVCache = None):
        if kv is not None:
            raise NotImplementedError("MLA is self-attention only.")
        b, s_q, _ = x.shape
        h, nope, rope, vh = self.num_heads, self.nope, self.rope, self.v_head

        # ---- queries: compress then expand to per-head [content | rope] ----
        q = self.q_up(self.q_norm(self.q_down(x)))               # [B, S_q, H·(nope+rope)]
        q = q.view(b, s_q, h, nope + rope).transpose(1, 2)       # -> [B, H, S_q, nope+rope]
        q_c, q_r = q.split([nope, rope], dim=-1)                 # content vs positional slices

        # ---- KV: build the small latent + shared rope key (the only things cached) ----
        c_kv = self.kv_down(x).unsqueeze(1)                      # [B, 1, S_q, d_c]  (head axis = 1 to fit KVCache)
        k_r = self.kr_proj(x).unsqueeze(1)                       # [B, 1, S_q, rope] shared pre-rope key
        if kv_cache is not None:
            c_kv, k_r = kv_cache.update(c_kv, k_r)               # append + read back the full past+present
        s_k = c_kv.shape[2]                                      # full key length (= S_q without a cache)

        kv = self.kv_up(self.kv_norm(c_kv.squeeze(1)))           # [B, S_k, H·(nope+v_head)]
        kv = kv.view(b, s_k, h, nope + vh).transpose(1, 2)       # -> [B, H, S_k, nope+v_head]
        k_c, v = kv.split([nope, vh], dim=-1)                    # per-head content keys / values

        # ---- decoupled RoPE: rotate q_r (per-head) and the shared k_r by absolute position ----
        cos, sin = _rope_tables(s_k, rope, x.device, x.dtype)    # tables for positions [0, S_k)
        k_r = _apply_rope(k_r, cos, sin)                         # rotate the shared rope key  [B,1,S_k,rope]
        q_r = _apply_rope(q_r, cos[s_k - s_q:], sin[s_k - s_q:])  # queries are the last S_q positions

        # ---- scores = content·content + rope·rope (k_r broadcasts over heads) ----
        scores = (torch.matmul(q_c, k_c.transpose(-2, -1))       # q_C·k_C   over the nope dims
                  + torch.matmul(q_r, k_r.transpose(-2, -1))) * self.scale  # + q_R·k_R, then /√(nope+rope)

        if is_causal:
            keep = make_causal_mask(s_q, s_k, x.device)          # query i sees key j ≤ i (end-aligned for caches)
            scores = scores.masked_fill(~keep, float("-inf"))
        if attn_mask is not None:
            scores = (scores.masked_fill(~attn_mask, float("-inf"))
                      if attn_mask.dtype == torch.bool else scores + attn_mask)

        attn = torch.softmax(scores, dim=-1)                     # attention weights over keys
        if self.dropout_p > 0.0 and self.training:
            attn = F.dropout(attn, p=self.dropout_p)
        out = torch.matmul(attn, v)                              # Σ_j a_ij v_j -> [B, H, S_q, v_head]
        out = out.transpose(1, 2).reshape(b, s_q, h * vh)        # concat heads
        return self.o_proj(out)                                  # project back to model dim
