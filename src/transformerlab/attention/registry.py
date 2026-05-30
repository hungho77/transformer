"""Name -> attention-class registry. Variants register via @register_attention."""
from __future__ import annotations

from typing import Type

from .base import AttentionBase, AttentionConfig

ATTENTION_REGISTRY: dict[str, Type[AttentionBase]] = {}


def register_attention(name: str):
    def deco(cls: Type[AttentionBase]) -> Type[AttentionBase]:
        if name in ATTENTION_REGISTRY:
            raise ValueError(f"Attention '{name}' is already registered.")
        cls.name = name
        ATTENTION_REGISTRY[name] = cls
        return cls

    return deco


def build_attention(name: str, cfg: AttentionConfig) -> AttentionBase:
    if name not in ATTENTION_REGISTRY:
        raise KeyError(
            f"Unknown attention '{name}'. Available: {available_attentions()}"
        )
    return ATTENTION_REGISTRY[name](cfg)


def available_attentions() -> list[str]:
    return sorted(ATTENTION_REGISTRY)
