"""Train a char-level GPT on tiny-shakespeare.

    python examples/train_gpt.py --config configs/gpt_char_tiny.yaml
    python examples/train_gpt.py --config configs/gpt_char_tiny.yaml --steps 300
    python examples/train_gpt.py --config configs/gpt_char_tiny.yaml --resume saved/gpt_char_tiny/last.pt
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from transformerlab.data import tiny_shakespeare_splits
from transformerlab.models import GPT, GPTConfig
from transformerlab.train import Trainer, build_optimizer, build_scheduler, load_run_config
from transformerlab.utils import prepare_device, set_seed


def lm_loss(model, batch):
    x, y = batch
    _, loss = model(x, y)
    return loss, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=None, help="override max_steps")
    ap.add_argument("--attention", default=None, help="override attention_name")
    ap.add_argument("--resume", default=None, help="checkpoint path to resume from")
    ap.add_argument("--precision", default=None, choices=["fp32", "bf16", "fp16"],
                    help="override mixed-precision (autocast) dtype")
    args = ap.parse_args()

    cfg = load_run_config(args.config)
    if args.steps is not None:
        cfg.max_steps = args.steps
    if args.attention is not None:
        cfg.model["attention_name"] = args.attention
    if args.resume is not None:
        cfg.resume = args.resume
    if args.precision is not None:
        cfg.amp_dtype = args.precision

    set_seed(cfg.seed)
    device, _ = prepare_device(0 if cfg.device == "cpu" else 1)

    block_size = cfg.model.get("max_seq_len", 128)
    train_ds, val_ds = tiny_shakespeare_splits(block_size, val_frac=cfg.data.get("val_frac", 0.1))
    print(f"train chars={len(train_ds.data):,}  val chars={len(val_ds.data):,}  vocab={train_ds.vocab_size}")

    model = GPT(GPTConfig(vocab_size=train_ds.vocab_size, **cfg.model))
    print(f"model params={model.num_params():,}  attention={cfg.model.get('attention_name', 'mha')}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, drop_last=True)
    total_steps = cfg.max_steps or len(train_loader) * cfg.epochs // cfg.accum_steps
    optimizer = build_optimizer(model.parameters(), cfg.optimizer)
    scheduler = build_scheduler(optimizer, cfg.scheduler, total_steps)

    trainer = Trainer(
        model, optimizer, train_loader, device, loss_fn=lm_loss, valid_loader=val_loader,
        scheduler=scheduler, grad_clip=cfg.grad_clip, amp=cfg.amp, amp_dtype=cfg.amp_dtype,
        accum_steps=cfg.accum_steps,
        log_interval=cfg.log_interval, eval_interval=cfg.eval_interval, save_dir=cfg.save_dir,
        name=cfg.name, monitor=cfg.monitor, mode=cfg.mode, patience=cfg.patience, save_best=cfg.save_best,
    )

    start_step = trainer.load_checkpoint(cfg.resume) if cfg.resume else 0
    if start_step:
        print(f"resumed from {cfg.resume} at step {start_step}")
    history = trainer.train(epochs=cfg.epochs, max_steps=cfg.max_steps, start_step=start_step)

    out_dir = Path(cfg.save_dir) / cfg.name
    torch.save(
        {"stoi": train_ds.stoi, "itos": train_ds.itos, "model_cfg": model.cfg.__dict__},
        out_dir / "meta.pt",
    )
    if len(history["loss"]) >= 2:
        print(f"loss: {history['loss'][0]:.4f} -> {history['loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
