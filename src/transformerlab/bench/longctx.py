"""Long-context benchmark: quality x efficiency as a function of context length.

The efficient-attention variants only earn their keep at long context — that's
exactly the regime the fixed-length quality bench never exercises. This sweeps
the *context length* and, for each (variant, ctx_len), trains the same small GPT
and reports validation perplexity, throughput, and peak memory. The headline is
which variants keep running (and at what quality/cost) as ctx_len grows while
`mha`/`local` blow up quadratically and OOM.

Reuses ``run_quality_sweep`` wholesale — here we only vary ``model.max_seq_len``
(which also drives the dataset block size and the per-variant window) per length.
"""
from __future__ import annotations

from .quality import run_quality_sweep
from .sweep import format_table

LONGCTX_COLUMNS = ["variant", "ctx_len", "val_ppl", "tokens_per_s", "peak_mem_MB"]


def run_longctx_sweep(*, variants=None, ctx_lens=(256, 512, 1024, 2048), model=None,
                      train_steps=100, batch_size=16, optimizer=None, seed=123,
                      device=None, num_workers=2):
    model = dict(model or {})
    rows = []
    for ctx_len in ctx_lens:
        # Train/eval entirely at this context length: max_seq_len drives the model,
        # the tiny-shakespeare block size, and the sliding-window size for local*.
        per_len_model = {**model, "max_seq_len": ctx_len}
        for r in run_quality_sweep(
            variants=variants, model=per_len_model, train_steps=train_steps,
            batch_size=batch_size, optimizer=optimizer, seed=seed, device=device,
            num_workers=num_workers,
        ):
            r["ctx_len"] = ctx_len          # tag the length so rows are comparable across the sweep
            rows.append(r)
    return rows


def format_longctx_table(rows) -> str:
    return format_table(rows, columns=LONGCTX_COLUMNS)
