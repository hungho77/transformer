"""Masked-language-modeling dataset for BERT-style pretraining (char-level).

Wraps a text corpus, reserving one extra vocab id for [MASK]. Each item returns
(input_ids, labels) where labels are -100 except at masked positions (BERT's
15% rule: 80% -> [MASK], 10% random token, 10% unchanged).
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .char_dataset import download_tiny_shakespeare

IGNORE_INDEX = -100


class MLMCharDataset(Dataset):
    def __init__(self, text: str, block_size: int, mask_prob: float = 0.15, seed: int = 0):
        self.block_size = block_size
        self.mask_prob = mask_prob
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.mask_id = len(chars)               # extra id reserved for [MASK]
        self.vocab_size = len(chars) + 1
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self._gen = torch.Generator().manual_seed(seed)

    @classmethod
    def from_tiny_shakespeare(cls, block_size: int, mask_prob: float = 0.15, data_dir="data"):
        return cls(download_tiny_shakespeare(data_dir), block_size, mask_prob)

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        tokens = self.data[idx: idx + self.block_size].clone()
        labels = torch.full_like(tokens, IGNORE_INDEX)

        probs = torch.rand(self.block_size, generator=self._gen)
        masked = probs < self.mask_prob
        labels[masked] = tokens[masked]

        decision = torch.rand(self.block_size, generator=self._gen)
        replace_mask = masked & (decision < 0.8)
        replace_rand = masked & (decision >= 0.8) & (decision < 0.9)
        tokens[replace_mask] = self.mask_id
        n_rand = int(replace_rand.sum())
        if n_rand:
            tokens[replace_rand] = torch.randint(0, self.vocab_size, (n_rand,), generator=self._gen)
        return tokens, labels
