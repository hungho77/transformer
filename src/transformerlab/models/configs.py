"""Dataclass configs for each model. These are the source of truth; YAML run
files are loaded into these via train/config.py."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass
class GPTConfig:
    vocab_size: int
    max_seq_len: int = 256
    dim: int = 256
    n_layers: int = 4
    num_heads: int = 4
    num_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    attention_name: Union[str, list] = "mha"
    ffn_type: str = "mlp"
    ffn_mult: float = 4.0
    norm_type: str = "rms"
    dropout: float = 0.0
    bias: bool = False
    use_rotary: bool = True
    window_size: Optional[int] = None
    tie_weights: bool = True
    extra: dict = field(default_factory=dict)


@dataclass
class ViTConfig:
    image_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    num_classes: int = 10
    dim: int = 192
    n_layers: int = 6
    num_heads: int = 3
    num_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    attention_name: Union[str, list] = "mha"
    ffn_type: str = "mlp"
    ffn_mult: float = 4.0
    norm_type: str = "layer"
    dropout: float = 0.0
    bias: bool = True
    pool: str = "cls"  # 'cls' or 'mean'
    extra: dict = field(default_factory=dict)


@dataclass
class Seq2SeqConfig:
    src_vocab_size: int
    tgt_vocab_size: int
    max_seq_len: int = 128
    dim: int = 256
    n_encoder_layers: int = 3
    n_decoder_layers: int = 3
    num_heads: int = 4
    num_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    attention_name: Union[str, list] = "mha"
    ffn_type: str = "mlp"
    ffn_mult: float = 4.0
    norm_type: str = "layer"
    dropout: float = 0.0
    bias: bool = True
    pad_id: int = 0
    extra: dict = field(default_factory=dict)
