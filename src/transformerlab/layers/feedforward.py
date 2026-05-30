"""Position-wise feed-forward networks."""
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Standard two-layer FFN with a configurable activation."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0, bias: bool = True, activation="gelu"):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.drop = nn.Dropout(dropout)
        self.act = getattr(F, activation)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class SwiGLU(nn.Module):
    """Gated FFN with SiLU gating (Shazeer, 2020), as used by LLaMA."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.up = nn.Linear(dim, hidden_dim, bias=bias)
        self.down = nn.Linear(hidden_dim, dim, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


def build_ffn(ffn_type: str, dim: int, hidden_dim: int, dropout: float = 0.0, bias: bool = True) -> nn.Module:
    ffn_type = ffn_type.lower()
    if ffn_type == "mlp":
        return MLP(dim, hidden_dim, dropout=dropout, bias=bias)
    if ffn_type == "swiglu":
        return SwiGLU(dim, hidden_dim, dropout=dropout, bias=bias)
    raise ValueError(f"Unknown ffn_type '{ffn_type}' (expected 'mlp' or 'swiglu').")
