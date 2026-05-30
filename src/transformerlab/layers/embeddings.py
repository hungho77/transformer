"""Token, positional, and patch embeddings."""
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)

    def forward(self, idx):
        return self.emb(idx)


class LearnedPositionalEmbedding(nn.Module):
    """Absolute learned positional embedding, sliced to the input length."""

    def __init__(self, max_seq_len: int, dim: int):
        super().__init__()
        self.pos = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pos(positions)[None, :, :]


class PatchEmbedding(nn.Module):
    """Conv-based patchifier for vision transformers: image -> [B, n_patches, dim]."""

    def __init__(self, image_size: int, patch_size: int, in_channels: int, dim: int):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                  # [B, dim, H/p, W/p]
        return x.flatten(2).transpose(1, 2)  # [B, n_patches, dim]
