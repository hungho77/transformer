"""Character-level language-modeling dataset (tiny-shakespeare by default).

Downloads the corpus on first use, builds a char<->id vocab, and yields
(input, target) blocks for next-token prediction.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import Dataset

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def download_tiny_shakespeare(data_dir="data") -> str:
    path = Path(data_dir) / "tinyshakespeare.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
    return path.read_text(encoding="utf-8")


class CharDataset(Dataset):
    def __init__(self, text: str, block_size: int):
        self.block_size = block_size
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    @classmethod
    def from_tiny_shakespeare(cls, block_size: int, data_dir="data"):
        return cls(download_tiny_shakespeare(data_dir), block_size)

    def encode(self, s: str):
        return torch.tensor([self.stoi[c] for c in s], dtype=torch.long)

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]
