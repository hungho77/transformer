"""Synthetic seq2seq tasks (copy / reverse) for fast, download-free training.

Token convention: 0=pad, 1=bos, 2=eos, content tokens start at 3.
"""
import torch
from torch.utils.data import Dataset

PAD_ID, BOS_ID, EOS_ID = 0, 1, 2
N_SPECIAL = 3


class CopyDataset(Dataset):
    def __init__(self, n_samples=4096, seq_len=12, n_symbols=20, mode="copy", seed=0):
        self.seq_len = seq_len
        self.mode = mode
        self.vocab_size = n_symbols + N_SPECIAL
        g = torch.Generator().manual_seed(seed)
        self.data = torch.randint(N_SPECIAL, self.vocab_size, (n_samples, seq_len), generator=g)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        src = self.data[idx]
        body = torch.flip(src, dims=[0]) if self.mode == "reverse" else src
        tgt_in = torch.cat([torch.tensor([BOS_ID]), body])
        tgt_out = torch.cat([body, torch.tensor([EOS_ID])])
        return src, tgt_in, tgt_out
