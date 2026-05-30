"""Train a Vision Transformer on CIFAR-10 (or 'fake' data for a quick smoke run).

    python examples/train_vit.py --config configs/vit_cifar10.yaml
    python examples/train_vit.py --config configs/vit_cifar10.yaml --steps 50 --dataset fake
"""
import argparse

from torch.utils.data import DataLoader

from transformerlab.data import build_vision_dataset
from transformerlab.models import VisionTransformer, ViTConfig
from transformerlab.train import Trainer, build_optimizer, build_scheduler, load_run_config
from transformerlab.utils import prepare_device, set_seed


def classification_loss(model, batch):
    images, labels = batch
    logits, loss = model(images, labels)
    acc = (logits.argmax(dim=-1) == labels).float().mean().item()
    return loss, {"acc": acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--attention", default=None)
    ap.add_argument("--dataset", default=None, help="override data.name (e.g. fake)")
    args = ap.parse_args()

    cfg = load_run_config(args.config)
    if args.steps is not None:
        cfg.max_steps = args.steps
    if args.attention is not None:
        cfg.model["attention_name"] = args.attention
    if args.dataset is not None:
        cfg.data["name"] = args.dataset

    set_seed(cfg.seed)
    device, _ = prepare_device(0 if cfg.device == "cpu" else 1)

    dataset = build_vision_dataset(cfg.data.get("name", "cifar10"), train=True)
    model = VisionTransformer(ViTConfig(**cfg.model))
    print(f"model params={model.num_params():,}  attention={cfg.model.get('attention_name', 'mha')}")

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.num_workers, drop_last=True)
    total_steps = cfg.max_steps or len(loader) * cfg.epochs
    optimizer = build_optimizer(model.parameters(), cfg.optimizer)
    scheduler = build_scheduler(optimizer, cfg.scheduler, total_steps)

    trainer = Trainer(
        model, optimizer, loader, device, loss_fn=classification_loss, scheduler=scheduler,
        grad_clip=cfg.grad_clip, amp=cfg.amp, log_interval=cfg.log_interval,
        save_dir=cfg.save_dir, name=cfg.name,
    )
    trainer.train(epochs=cfg.epochs, max_steps=cfg.max_steps)


if __name__ == "__main__":
    main()
