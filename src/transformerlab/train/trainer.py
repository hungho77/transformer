"""Task-agnostic training loop.

Generalizes the original template's train/valid-epoch + checkpoint structure.
The task is injected as ``loss_fn(model, batch) -> (loss, metrics_dict)`` so the
same trainer drives language modeling, classification, and seq2seq.

Supports gradient accumulation (effective large batch on one GPU), full-state
checkpoint/resume (model + optimizer + scheduler + scaler + RNG + step), and
best-checkpoint tracking with early stopping. Step counts are in *optimizer
steps*, so ``max_steps``/``log_interval``/``eval_interval`` are independent of
the accumulation factor.
"""
from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
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
        accum_steps: int = 1,
        log_interval: int = 50,
        eval_interval: int = 0,
        save_dir: str = "saved",
        name: str = "run",
        monitor: str = "",          # metric to track for best/early-stop, e.g. "val_loss" ("" disables)
        mode: str = "min",          # "min" -> lower is better, "max" -> higher is better
        patience: int = 0,          # stop after this many non-improving evals (0 disables early stop)
        save_best: bool = False,    # also keep best.pt by `monitor`
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
        self.accum_steps = max(1, accum_steps)
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.out_dir = Path(save_dir) / name
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.save_best = save_best
        self.best = float("inf") if mode == "min" else float("-inf")
        self.bad_evals = 0
        self.history = {"step": [], "loss": []}

    def _micro_step(self, batch):
        # One micro-batch: forward + scaled backward (grads accumulate; no opt step).
        batch = _to_device(batch, self.device)
        with torch.autocast(self.device.type, enabled=self.amp):
            loss, metrics = self.loss_fn(self.model, batch)
        # Divide by accum_steps so the summed grads equal the mean over the
        # effective (accum_steps · micro) batch, matching a single big batch.
        self.scaler.scale(loss / self.accum_steps).backward()
        return loss.item(), metrics

    def _optimizer_step(self):
        # Apply the accumulated grads: unscale -> clip -> step -> zero -> schedule.
        if self.grad_clip:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None:
            self.scheduler.step()

    def _is_improved(self, value):
        return value < self.best if self.mode == "min" else value > self.best

    def _maybe_eval(self, step) -> bool:
        """Run eval; update best/early-stop. Returns True if training should stop."""
        result = self.evaluate()
        self.model.train()
        if not self.monitor or self.monitor not in result:
            return False
        if self._is_improved(result[self.monitor]):
            self.best = result[self.monitor]
            self.bad_evals = 0
            if self.save_best:
                self.save_checkpoint("best", step)
        else:
            self.bad_evals += 1
        return self.patience and self.bad_evals > self.patience

    def train(self, epochs: int = 1, max_steps: Optional[int] = None, start_step: int = 0):
        step = start_step
        micro = 0
        tracker = MetricTracker("loss")
        start = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        for epoch in range(epochs):
            self.model.train()
            for batch in self.train_loader:
                loss, metrics = self._micro_step(batch)
                tracker.update("loss", loss)
                for k, v in metrics.items():
                    tracker.update(k, v)
                micro += 1
                if micro % self.accum_steps != 0:
                    continue                         # keep accumulating until a full effective batch
                self._optimizer_step()
                step += 1                            # one optimizer step

                if step % self.log_interval == 0:
                    avg = tracker.result()
                    self.history["step"].append(step)
                    self.history["loss"].append(avg["loss"])
                    extra = "  ".join(f"{k}={v:.4f}" for k, v in avg.items() if k != "loss")
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(f"step {step:6d} | loss {avg['loss']:.4f} | {extra} | lr {lr:.2e}")
                    tracker.reset()

                if self.eval_interval and self.valid_loader is not None and step % self.eval_interval == 0:
                    if self._maybe_eval(step):
                        print(f"early stop at step {step} (no {self.monitor} improvement)")
                        self.save_checkpoint("last", step)
                        return self.history

                if max_steps and step >= max_steps:
                    self.save_checkpoint("last", step)
                    print(f"done {step} steps in {time.perf_counter()-start:.1f}s")
                    return self.history
        self.save_checkpoint("last", step)
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

    def save_checkpoint(self, tag: str = "last", step: int = 0):
        # Full training state so a run can resume bit-for-bit (incl. RNG/scheduler).
        ensure_dir(self.out_dir)
        path = self.out_dir / f"{tag}.pt"
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler": self.scaler.state_dict(),
            "step": step,
            "best": self.best,
            "config": getattr(self.model, "cfg", None),
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }, path)
        return path

    def load_checkpoint(self, path) -> int:
        """Restore model/optimizer/scheduler/scaler/RNG; return the step to resume from."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler is not None and ckpt.get("scheduler") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.best = ckpt.get("best", self.best)
        rng = ckpt.get("rng")
        if rng is not None:
            # RNG states must be CPU uint8 tensors; map_location may have moved
            # them to the model's device, so force them back to CPU.
            torch.set_rng_state(rng["torch"].cpu())
            if rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
        return ckpt.get("step", 0)
