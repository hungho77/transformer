"""Normalization layers."""
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm * self.weight


def build_norm(norm_type: str, dim: int) -> nn.Module:
    norm_type = norm_type.lower()
    if norm_type == "rms":
        return RMSNorm(dim)
    if norm_type == "layer":
        return nn.LayerNorm(dim)
    raise ValueError(f"Unknown norm_type '{norm_type}' (expected 'rms' or 'layer').")
