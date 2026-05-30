"""Pretrain a BERT-style encoder with masked language modeling on tiny-shakespeare.

    python examples/train_bert.py --config configs/bert_char_tiny.yaml
    python examples/train_bert.py --config configs/bert_char_tiny.yaml --attention sdpa
"""
import argparse

from torch.utils.data import DataLoader

from transformerlab.data import MLMCharDataset
from transformerlab.models import BERT, BERTConfig
from transformerlab.train import Trainer, build_optimizer, build_scheduler, load_run_config
from transformerlab.utils import prepare_device, set_seed


def mlm_loss(model, batch):
    input_ids, labels = batch
    logits, loss = model(input_ids, mlm_labels=labels)
    masked = labels != -100
    acc = ((logits.argmax(-1) == labels) & masked).float().sum() / masked.float().sum().clamp(min=1)
    return loss, {"mlm_acc": acc.item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--attention", default=None)
    args = ap.parse_args()

    cfg = load_run_config(args.config)
    if args.steps is not None:
        cfg.max_steps = args.steps
    if args.attention is not None:
        cfg.model["attention_name"] = args.attention

    set_seed(cfg.seed)
    device, _ = prepare_device(0 if cfg.device == "cpu" else 1)

    block_size = cfg.model.get("max_seq_len", 128)
    dataset = MLMCharDataset.from_tiny_shakespeare(block_size, mask_prob=cfg.data.get("mask_prob", 0.15))
    print(f"corpus chars={len(dataset.data):,}  vocab={dataset.vocab_size} (incl. [MASK])")

    model = BERT(BERTConfig(vocab_size=dataset.vocab_size, **cfg.model))
    print(f"model params={model.num_params():,}  attention={cfg.model.get('attention_name', 'mha')}")

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.num_workers, drop_last=True)
    total_steps = cfg.max_steps or len(loader) * cfg.epochs
    optimizer = build_optimizer(model.parameters(), cfg.optimizer)
    scheduler = build_scheduler(optimizer, cfg.scheduler, total_steps)

    trainer = Trainer(
        model, optimizer, loader, device, loss_fn=mlm_loss, scheduler=scheduler,
        grad_clip=cfg.grad_clip, amp=cfg.amp, log_interval=cfg.log_interval,
        save_dir=cfg.save_dir, name=cfg.name,
    )
    trainer.train(epochs=cfg.epochs, max_steps=cfg.max_steps)


if __name__ == "__main__":
    main()
