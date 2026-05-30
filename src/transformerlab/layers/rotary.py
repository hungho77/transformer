"""Rotary position embeddings (Su et al., 2021)."""
import torch
import torch.nn as nn


def _rotate_half(x):
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
            raise ValueError("RoPE requires an even head_dim.")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len

    def _cos_sin(self, seq_len, offset, device, dtype):
        pos = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(self, q, k, offset: int = 0):
        seq_q, seq_k = q.shape[-2], k.shape[-2]
        cos_k, sin_k = self._cos_sin(seq_k, offset, q.device, q.dtype)
        # Queries occupy the last seq_q positions of the key window.
        cos_q, sin_q = cos_k[seq_k - seq_q:], sin_k[seq_k - seq_q:]
        q_out = q * cos_q + _rotate_half(q) * sin_q
        k_out = k * cos_k + _rotate_half(k) * sin_k
        return q_out, k_out
