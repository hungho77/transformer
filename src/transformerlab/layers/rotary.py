"""Rotary position embeddings (RoPE; Su et al., 2021).

RoPE injects *relative* position by rotating each 2-D slice of a query/key
vector by an angle proportional to its absolute position m and the slice's
frequency θ:

    RoPE(x, m) = x · cos(mθ) + rotate_half(x) · sin(mθ)

Because rotations compose, the dot product qₘ·kₙ ends up depending only on the
relative offset (m − n) — that's what gives RoPE its length-extrapolation behavior.
"""
import torch
import torch.nn as nn


def _rotate_half(x):
    # Maps [x1, x2] (the two halves of the feature dim) -> [-x2, x1]. This is the
    # 90° rotation partner used so that x·cos + rotate_half(x)·sin realizes the
    # 2-D rotation matrix on each (xⁱ, xⁱ⁺ᵈ/²) pair.
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Applies RoPE to query/key tensors of shape [B, H, S, head_dim].

    Call as ``q, k = rotary(q, k, offset=past_len)`` where ``offset`` accounts
    for cached positions during incremental decoding.
    """

    def __init__(self, head_dim: int, base: float = 10000.0, max_seq_len: int = 4096):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim.")  # features are rotated in pairs
        # Per-pair frequencies θ_i = base^(-2i/d): low i rotate fast (local detail),
        # high i rotate slowly (long-range position) — a geometric spectrum.
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)  # not a learned param; don't save in ckpt
        self.max_seq_len = max_seq_len

    def _cos_sin(self, seq_len, offset, device, dtype):
        # Build the rotation angles mθ for positions [offset, offset+seq_len).
        pos = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)  # m
        freqs = torch.outer(pos, self.inv_freq.to(device))  # mθ_i  -> [seq_len, d/2]
        emb = torch.cat((freqs, freqs), dim=-1)             # duplicate so each half lines up with rotate_half
        return emb.cos().to(dtype), emb.sin().to(dtype)     # cos(mθ), sin(mθ)

    def forward(self, q, k, offset: int = 0):
        seq_q, seq_k = q.shape[-2], k.shape[-2]
        cos_k, sin_k = self._cos_sin(seq_k, offset, q.device, q.dtype)  # angles for the full key span
        # With a KV cache seq_k ≥ seq_q; the queries are the *last* seq_q positions,
        # so slice the tail of the angle tables for them.
        cos_q, sin_q = cos_k[seq_k - seq_q:], sin_k[seq_k - seq_q:]
        q_out = q * cos_q + _rotate_half(q) * sin_q         # rotate each query by its position
        k_out = k * cos_k + _rotate_half(k) * sin_k         # rotate each key by its position
        return q_out, k_out
