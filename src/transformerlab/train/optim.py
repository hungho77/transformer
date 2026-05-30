"""Optimizer and LR-schedule builders."""
import math

import torch


def build_optimizer(params, spec: dict):
    """spec = {"type": "AdamW", "lr": ..., "weight_decay": ...}."""
    spec = dict(spec)
    name = spec.pop("type", "AdamW")
    if not hasattr(torch.optim, name):
        raise ValueError(f"Unknown optimizer '{name}'.")
    return getattr(torch.optim, name)(params, **spec)


def cosine_warmup(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
    """Linear warmup then cosine decay to ``min_lr_ratio`` of the base LR."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_scheduler(optimizer, spec: dict, total_steps: int):
    """spec = {"type": "cosine_warmup", "warmup_steps": ...} or empty for none."""
    if not spec:
        return None
    spec = dict(spec)
    name = spec.pop("type")
    if name == "cosine_warmup":
        return cosine_warmup(optimizer, total_steps=total_steps, **spec)
    if hasattr(torch.optim.lr_scheduler, name):
        return getattr(torch.optim.lr_scheduler, name)(optimizer, **spec)
    raise ValueError(f"Unknown scheduler '{name}'.")
