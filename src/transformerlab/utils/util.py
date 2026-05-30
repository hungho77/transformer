"""General utilities (carried over from the original template, pandas removed)."""
import json
from pathlib import Path
from itertools import repeat
from collections import OrderedDict

import torch


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=True)


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """Wrapper that yields batches from a data loader endlessly."""
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """Pick the training device and the list of GPU ids for DataParallel."""
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: no GPU available, training will run on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: {n_gpu_use} GPU(s) requested but only {n_gpu} available.")
        n_gpu_use = n_gpu
    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids


class MetricTracker:
    """Tracks running averages of named scalar metrics (plain-dict, no pandas)."""

    def __init__(self, *keys):
        self._keys = list(keys)
        self._total = {}
        self._counts = {}
        self.reset()

    def reset(self):
        for key in self._keys:
            self._total[key] = 0.0
            self._counts[key] = 0

    def update(self, key, value, n=1):
        if key not in self._total:
            self._keys.append(key)
            self._total[key] = 0.0
            self._counts[key] = 0
        self._total[key] += float(value) * n
        self._counts[key] += n

    def avg(self, key):
        count = self._counts.get(key, 0)
        return self._total[key] / count if count else 0.0

    def result(self):
        return {key: self.avg(key) for key in self._keys}
