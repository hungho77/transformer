"""Normalization layers."""
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (Zhang & Sennrich, 2019).

        RMSNorm(x) = x / sqrt(mean(x²) + ε) · g

    Like LayerNorm but with no mean-subtraction and no bias — only rescaling by
    the RMS. Cheaper, and empirically as effective; used by LLaMA-style models.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps                              # ε: numerical floor so we never divide by 0
        self.weight = nn.Parameter(torch.ones(dim))  # learned per-channel gain g (starts at identity)

    def forward(self, x):
        # rsqrt(mean(x²)+ε) = 1/RMS(x); scale x to unit RMS along the feature axis.
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm * self.weight                   # apply the learned gain


def build_norm(norm_type: str, dim: int) -> nn.Module:
    norm_type = norm_type.lower()
    if norm_type == "rms":
        return RMSNorm(dim)
    if norm_type == "layer":
        return nn.LayerNorm(dim)
    raise ValueError(f"Unknown norm_type '{norm_type}' (expected 'rms' or 'layer').")
