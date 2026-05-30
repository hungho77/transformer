"""Low-level measurement helpers: latency, peak memory, and FLOPs."""
from __future__ import annotations

import time
from typing import Callable

import torch

try:
    from torch.utils.flop_counter import FlopCounterMode
    _HAS_FLOP_COUNTER = True
except Exception:  # pragma: no cover
    _HAS_FLOP_COUNTER = False


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_callable(fn: Callable, device, warmup: int = 3, iters: int = 10) -> float:
    """Mean wall-clock milliseconds per call of ``fn`` (includes whatever fn does)."""
    for _ in range(warmup):
        fn()
    _sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - start) * 1000.0 / iters


def measure_peak_memory(fn: Callable, device) -> float:
    """Peak allocated memory in MB during one call (CUDA only; 0.0 on CPU)."""
    if device.type != "cuda":
        fn()
        return 0.0
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def measure_flops(fn: Callable) -> float:
    """Forward-pass GFLOPs via FlopCounterMode, or float('nan') if unavailable."""
    if not _HAS_FLOP_COUNTER:
        return float("nan")
    counter = FlopCounterMode(display=False)
    with counter:
        fn()
    return counter.get_total_flops() / 1e9
