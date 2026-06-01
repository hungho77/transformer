"""Quality x efficiency benchmark.

Trains the *same* model under each attention variant and reports validation
perplexity alongside training throughput and peak memory, so cheaper attention
can be judged against the quality it costs. A Pareto flag marks variants that no
other variant beats on both quality (val_ppl) and cost (peak_mem_MB).

GPT char-LM is the built-in task; the orchestration takes a ``build_model`` /
``loss_fn`` so other tasks can be slotted in later.
"""
from __future__ import annotations

import math
import time

import torch
from torch.utils.data import DataLoader

from ..attention import available_attentions
from ..data import tiny_shakespeare_splits
from ..models import GPT, GPTConfig
from ..train import Trainer, build_optimizer
from ..utils import set_seed
from .sweep import _free_memory, format_table

QUALITY_COLUMNS = ["variant", "params", "train_loss", "val_loss", "val_ppl",
                   "tokens_per_s", "peak_mem_MB", "pareto"]


def lm_loss(model, batch):
    _, loss = model(batch[0], batch[1])
    return loss, {}


@torch.no_grad()
def evaluate_ppl(model, val_loader, device, loss_fn, max_batches=50):
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        batch = tuple(t.to(device) for t in batch)
        loss, _ = loss_fn(model, batch)
        total += loss.item()
        n += 1
    val_loss = total / max(n, 1)
    return val_loss, math.exp(val_loss)


def train_and_evaluate(variant, *, build_model, loss_fn, train_loader, val_loader,
                       optimizer_spec, train_steps, batch_size, seq_len, device,
                       seed=123, grad_clip=1.0, max_eval_batches=50):
    set_seed(seed)
    model = build_model(variant).to(device)
    params = sum(p.numel() for p in model.parameters())

    optimizer = build_optimizer(model.parameters(), optimizer_spec)
    # log_interval == train_steps -> a single summary line at the end (and a
    # populated history) instead of per-step spam.
    trainer = Trainer(model, optimizer, train_loader, device, loss_fn=loss_fn,
                      grad_clip=grad_clip, log_interval=max(train_steps, 1),
                      eval_interval=0, save_dir="saved", name=f"quality_{variant}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    start = time.perf_counter()
    history = trainer.train(max_steps=train_steps)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
    val_loss, val_ppl = evaluate_ppl(model, val_loader, device, loss_fn, max_eval_batches)
    return {
        "variant": variant,
        "params": params,
        "train_loss": history["loss"][-1] if history["loss"] else float("nan"),
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "tokens_per_s": train_steps * batch_size * seq_len / elapsed,
        "peak_mem_MB": peak_mem,
    }


def _gpt_builder(vocab_size, model_cfg, num_heads, seq_len):
    base = {k: v for k, v in model_cfg.items() if k != "attention_name"}

    def build(variant):
        kw = dict(base)
        kw["attention_name"] = variant
        if variant == "gqa" and "num_kv_heads" not in kw:
            kw["num_kv_heads"] = max(1, num_heads // 2)
        if variant in ("local", "local_flex") and not kw.get("window_size"):
            kw["window_size"] = max(8, seq_len // 4)
        if variant == "alibi":
            # ALiBi carries its own position (score bias), so disable rotary and
            # suppress the learned positional embedding to avoid double-counting.
            kw["use_rotary"] = False
            kw["extra"] = {**kw.get("extra", {}), "no_pos_emb": True}
        return GPT(GPTConfig(vocab_size=vocab_size, **kw))

    return build


def run_quality_sweep(*, variants=None, model=None, data=None, train_steps=400,
                      batch_size=64, optimizer=None, seed=123, device=None,
                      num_workers=2):
    variants = variants or available_attentions()
    model = dict(model or {})
    data = dict(data or {})
    optimizer = optimizer or {"type": "AdamW", "lr": 6e-4, "weight_decay": 0.1}
    device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq_len = model.get("max_seq_len", 128)
    num_heads = model.get("num_heads", 4)
    train_ds, val_ds = tiny_shakespeare_splits(seq_len, val_frac=data.get("val_frac", 0.1))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=True)
    build_model = _gpt_builder(train_ds.vocab_size, model, num_heads, seq_len)

    rows = []
    for variant in variants:
        try:
            rows.append(train_and_evaluate(
                variant, build_model=build_model, loss_fn=lm_loss,
                train_loader=train_loader, val_loader=val_loader,
                optimizer_spec=optimizer, train_steps=train_steps,
                batch_size=batch_size, seq_len=seq_len, device=device, seed=seed,
                max_eval_batches=data.get("max_eval_batches", 50),
            ))
        except Exception as exc:  # noqa: BLE001
            rows.append({**{c: float("nan") for c in QUALITY_COLUMNS}, "variant": variant})
            print(f"  [skip] {variant}: {exc}")
        finally:
            _free_memory(device)
    return rows


def mark_pareto(rows, quality="val_ppl", cost="peak_mem_MB"):
    """Flag rows on the quality/cost Pareto frontier (both lower-is-better)."""
    valid = [r for r in rows if r.get(quality) == r.get(quality) and r.get(cost) == r.get(cost)]
    for r in rows:
        r["pareto"] = False
    for r in valid:
        dominated = any(
            o is not r and o[quality] <= r[quality] and o[cost] <= r[cost]
            and (o[quality] < r[quality] or o[cost] < r[cost])
            for o in valid
        )
        r["pareto"] = not dominated
    return rows


def format_quality_table(rows) -> str:
    return format_table(rows, columns=QUALITY_COLUMNS)
