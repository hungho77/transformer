"""Pluggable attention variants.

Importing this package registers every variant in ATTENTION_REGISTRY by name.
New variants: subclass ProjAttention (or AttentionBase) and decorate with
@register_attention("name").
"""
from .base import AttentionBase, AttentionConfig, KVCache, MaskType
from .core import (
    ProjAttention,
    make_causal_mask,
    make_sink_window_mask,
    make_sliding_window_mask,
    repeat_kv,
    sdpa_core,
)
from .registry import ATTENTION_REGISTRY, available_attentions, build_attention, register_attention

# Import variant modules for their registration side effects.
from . import mha, sdpa, mqa_gqa, linear, local, flash, mla, sink  # noqa: E402,F401

__all__ = [
    "AttentionBase",
    "AttentionConfig",
    "KVCache",
    "MaskType",
    "ProjAttention",
    "make_causal_mask",
    "make_sink_window_mask",
    "make_sliding_window_mask",
    "repeat_kv",
    "sdpa_core",
    "ATTENTION_REGISTRY",
    "available_attentions",
    "build_attention",
    "register_attention",
]
