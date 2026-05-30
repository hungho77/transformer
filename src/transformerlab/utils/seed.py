"""Reproducibility helpers (extracted from the original train.py)."""
import random

import numpy as np
import torch


def set_seed(seed: int = 123, deterministic: bool = True):
    """Fix RNG seeds across random/numpy/torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
