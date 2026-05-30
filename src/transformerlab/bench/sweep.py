"""Sweep attention variants across sequence lengths and report a comparison table."""
from __future__ import annotations

import gc

import torch

from ..attention import AttentionConfig, available_attentions, build_attention
from .profile import measure_flops, measure_peak_memory, time_callable

_COLUMNS = ["variant", "seq_len", "fwd_ms", "fwd_bwd_ms", "peak_mem_MB", "GFLOPs"]


def benchmark_attention(name, *, dim, num_heads, seq_len, batch, device, dtype=torch.float32,
                        num_kv_heads=None, window_size=None, is_causal=True, warmup=3, iters=10):
    cfg = AttentionConfig(
        dim=dim, num_heads=num_heads, num_kv_heads=num_kv_heads, window_size=window_size,
    )
    attn = build_attention(name, cfg).to(device=device, dtype=dtype)

    def fwd():
        x = torch.randn(batch, seq_len, dim, device=device, dtype=dtype)
        return attn(x, is_causal=is_causal)

    def fwd_bwd():
        x = torch.randn(batch, seq_len, dim, device=device, dtype=dtype, requires_grad=True)
        attn(x, is_causal=is_causal).sum().backward()

    return {
        "variant": name,
        "seq_len": seq_len,
        "fwd_ms": time_callable(fwd, device, warmup, iters),
        "fwd_bwd_ms": time_callable(fwd_bwd, device, warmup, iters),
        "peak_mem_MB": measure_peak_memory(fwd_bwd, device),
        "GFLOPs": measure_flops(fwd),
    }


def run_sweep(*, variants=None, seq_lens=(256, 512, 1024), dim=512, num_heads=8, batch=8,
              device=None, dtype="float32", is_causal=True, **kw):
    variants = variants or available_attentions()
    device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, dtype)
    rows = []
    for name in variants:
        kv = 2 if name == "gqa" else None
        win = max(seq_lens) // 4 if name in ("local", "local_flex") else None
        for seq_len in seq_lens:
            try:
                rows.append(benchmark_attention(
                    name, dim=dim, num_heads=num_heads, seq_len=seq_len, batch=batch,
                    device=device, dtype=dtype, num_kv_heads=kv, window_size=win,
                    is_causal=is_causal, **kw,
                ))
            except Exception as exc:  # noqa: BLE001
                rows.append({**{c: float("nan") for c in _COLUMNS}, "variant": name, "seq_len": seq_len})
                print(f"  [skip] {name} @ {seq_len}: {exc}")
            finally:
                _free_memory(device)
    return rows


def _free_memory(device):
    """Release allocator caches and compiled-kernel state between measurements so
    one variant's leftover memory does not OOM the next (notably the compiled
    flex_attention kernel, whose buffers otherwise persist across sequence lengths)."""
    try:
        torch._dynamo.reset()
    except Exception:  # noqa: BLE001
        pass
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def format_table(rows, columns=None) -> str:
    columns = columns or _COLUMNS
    widths = {c: max(len(c), *(len(_fmt(r.get(c))) for r in rows)) for c in columns}
    header = "  ".join(c.rjust(widths[c]) for c in columns)
    lines = [header, "  ".join("-" * widths[c] for c in columns)]
    for r in rows:
        lines.append("  ".join(_fmt(r.get(c)).rjust(widths[c]) for c in columns))
    return "\n".join(lines)


def _fmt(v):
    if isinstance(v, bool):
        return "*" if v else ""
    if isinstance(v, float):
        return "nan" if v != v else f"{v:.2f}"
    return "" if v is None else str(v)
