"""Base class for all models."""
import torch.nn as nn


class BaseModel(nn.Module):
    def num_params(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def extra_repr(self) -> str:
        return f"params={self.num_params():,}"
