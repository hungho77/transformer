"""Task-agnostic training loop.

Generalizes the original template's train/valid-epoch + checkpoint structure.
The task is injected as ``loss_fn(model, batch) -> (loss, metrics_dict)`` so the
same trainer drives language modeling, classification, and seq2seq.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import torch

from ..utils import MetricTracker, ensure_dir


def _to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, (list, tuple)):
        return type(batch)(_to_device(b, device) for b in batch)
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    return batch


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        train_loader,
        device,
        *,
        loss_fn: Callable,
        valid_loader=None,
        scheduler=None,
        grad_clip: float = 1.0,
        amp: bool = False,
        log_interval: int = 50,
        eval_interval: int = 0,
        save_dir: str = "saved",
        name: str = "run",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.amp = amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.out_dir = Path(save_dir) / name
        self.history = {"step": [], "loss": []}

    def _step(self, batch):
        batch = _to_device(batch, self.device)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(self.device.type, enabled=self.amp):
            loss, metrics = self.loss_fn(self.model, batch)
        self.scaler.scale(loss).backward()
        if self.grad_clip:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()
        return loss.item(), metrics

    def train(self, epochs: int = 1, max_steps: Optional[int] = None):
        step = 0
        tracker = MetricTracker("loss")
        start = time.perf_counter()
        for epoch in range(epochs):
            self.model.train()
            for batch in self.train_loader:
                loss, metrics = self._step(batch)
                tracker.update("loss", loss)
                for k, v in metrics.items():
                    tracker.update(k, v)
                step += 1

                if step % self.log_interval == 0:
                    avg = tracker.result()
                    self.history["step"].append(step)
                    self.history["loss"].append(avg["loss"])
                    extra = "  ".join(f"{k}={v:.4f}" for k, v in avg.items() if k != "loss")
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(f"step {step:6d} | loss {avg['loss']:.4f} | {extra} | lr {lr:.2e}")
                    tracker.reset()

                if self.eval_interval and self.valid_loader is not None and step % self.eval_interval == 0:
                    self.evaluate()
                    self.model.train()

                if max_steps and step >= max_steps:
                    self.save_checkpoint("last")
                    print(f"done {step} steps in {time.perf_counter()-start:.1f}s")
                    return self.history
        self.save_checkpoint("last")
        print(f"done {step} steps in {time.perf_counter()-start:.1f}s")
        return self.history

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        tracker = MetricTracker("val_loss")
        for batch in self.valid_loader:
            batch = _to_device(batch, self.device)
            loss, metrics = self.loss_fn(self.model, batch)
            tracker.update("val_loss", loss.item())
            for k, v in metrics.items():
                tracker.update("val_" + k, v)
        result = tracker.result()
        print("  eval | " + "  ".join(f"{k}={v:.4f}" for k, v in result.items()))
        return result

    def save_checkpoint(self, tag: str = "last"):
        ensure_dir(self.out_dir)
        path = self.out_dir / f"{tag}.pt"
        torch.save(
            {"model": self.model.state_dict(), "config": getattr(self.model, "cfg", None)},
            path,
        )
        return path
