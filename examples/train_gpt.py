"""Train a char-level GPT on tiny-shakespeare.

    python examples/train_gpt.py --config configs/gpt_char_tiny.yaml
    python examples/train_gpt.py --config configs/gpt_char_tiny.yaml --steps 300
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from transformerlab.data import CharDataset
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
    args = ap.parse_args()

    cfg = load_run_config(args.config)
    if args.steps is not None:
        cfg.max_steps = args.steps
    if args.attention is not None:
        cfg.model["attention_name"] = args.attention

    set_seed(cfg.seed)
    device, _ = prepare_device(0 if cfg.device == "cpu" else 1)

    block_size = cfg.model.get("max_seq_len", 128)
    dataset = CharDataset.from_tiny_shakespeare(block_size)
    print(f"corpus chars={len(dataset.data):,}  vocab={dataset.vocab_size}")

    model = GPT(GPTConfig(vocab_size=dataset.vocab_size, **cfg.model))
    print(f"model params={model.num_params():,}  attention={cfg.model.get('attention_name', 'mha')}")

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.num_workers, drop_last=True)
    total_steps = cfg.max_steps or len(loader) * cfg.epochs
    optimizer = build_optimizer(model.parameters(), cfg.optimizer)
    scheduler = build_scheduler(optimizer, cfg.scheduler, total_steps)

    trainer = Trainer(
        model, optimizer, loader, device, loss_fn=lm_loss, scheduler=scheduler,
        grad_clip=cfg.grad_clip, amp=cfg.amp, log_interval=cfg.log_interval,
        save_dir=cfg.save_dir, name=cfg.name,
    )
    history = trainer.train(epochs=cfg.epochs, max_steps=cfg.max_steps)

    out_dir = Path(cfg.save_dir) / cfg.name
    torch.save(
        {"stoi": dataset.stoi, "itos": dataset.itos, "model_cfg": model.cfg.__dict__},
        out_dir / "meta.pt",
    )
    if len(history["loss"]) >= 2:
        print(f"loss: {history['loss'][0]:.4f} -> {history['loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
