"""Train an encoder-decoder transformer on a synthetic copy/reverse task.

    python examples/train_seq2seq.py --config configs/seq2seq_copy.yaml
"""
import argparse

from torch.utils.data import DataLoader

from transformerlab.data import BOS_ID, EOS_ID, PAD_ID, CopyDataset
from transformerlab.models import EncoderDecoder, Seq2SeqConfig
from transformerlab.train import Trainer, build_optimizer, build_scheduler, load_run_config
from transformerlab.utils import prepare_device, set_seed


def make_loss_fn(pad_id):
    def seq2seq_loss(model, batch):
        src, tgt_in, tgt_out = batch
        logits, loss = model(src, tgt_in, targets=tgt_out)
        mask = tgt_out != pad_id
        acc = ((logits.argmax(-1) == tgt_out) & mask).float().sum() / mask.float().sum()
        return loss, {"acc": acc.item()}
    return seq2seq_loss


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

    d = cfg.data
    dataset = CopyDataset(n_samples=d.get("n_samples", 4096), seq_len=d.get("seq_len", 12),
                          n_symbols=d.get("n_symbols", 20), mode=d.get("mode", "copy"))
    model = EncoderDecoder(Seq2SeqConfig(
        src_vocab_size=dataset.vocab_size, tgt_vocab_size=dataset.vocab_size,
        pad_id=PAD_ID, **cfg.model,
    ))
    print(f"model params={model.num_params():,}  attention={cfg.model.get('attention_name', 'mha')}")

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.num_workers, drop_last=True)
    total_steps = cfg.max_steps or len(loader) * cfg.epochs
    optimizer = build_optimizer(model.parameters(), cfg.optimizer)
    scheduler = build_scheduler(optimizer, cfg.scheduler, total_steps)

    trainer = Trainer(
        model, optimizer, loader, device, loss_fn=make_loss_fn(PAD_ID), scheduler=scheduler,
        grad_clip=cfg.grad_clip, amp=cfg.amp, log_interval=cfg.log_interval,
        save_dir=cfg.save_dir, name=cfg.name,
    )
    trainer.train(epochs=cfg.epochs, max_steps=cfg.max_steps)

    # quick greedy decode sanity check
    src = dataset.data[:4].to(device)
    out = model.generate(src, BOS_ID, EOS_ID, max_len=dataset.seq_len + 1)
    print("src  :", src[0].tolist())
    print("decode:", out[0, 1:].tolist())


if __name__ == "__main__":
    main()
