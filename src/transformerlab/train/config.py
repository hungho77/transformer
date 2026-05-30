"""Run configuration: a dataclass loaded from YAML."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class RunConfig:
    arch: str = "gpt"               # gpt | vit | seq2seq
    name: str = "run"
    seed: int = 123
    device: str = "auto"            # auto | cpu | cuda
    epochs: int = 1
    max_steps: Optional[int] = None
    batch_size: int = 64
    num_workers: int = 2
    grad_clip: float = 1.0
    amp: bool = False
    log_interval: int = 50
    eval_interval: int = 0          # 0 disables periodic eval
    save_dir: str = "saved"
    model: dict = field(default_factory=dict)
    optimizer: dict = field(default_factory=lambda: {"type": "AdamW", "lr": 3e-4})
    scheduler: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)


def load_run_config(path) -> RunConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return RunConfig(**raw)
