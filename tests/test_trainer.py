import torch
from torch.utils.data import DataLoader, TensorDataset

from transformerlab.models import GPT, GPTConfig
from transformerlab.train import Trainer, build_optimizer
from transformerlab.utils import set_seed


def _lm_loss(model, batch):
    x, y = batch
    _, loss = model(x, y)
    return loss, {}


def _tiny_loader(n=32, seq=16, vocab=40, bs=8):
    x = torch.randint(0, vocab, (n, seq))
    y = torch.randint(0, vocab, (n, seq))
    return DataLoader(TensorDataset(x, y), batch_size=bs)


def _gpt(vocab=40, **kw):
    return GPT(GPTConfig(vocab_size=vocab, max_seq_len=16, dim=48, n_layers=2, num_heads=4, **kw))


def _make_trainer(model, tmp_path, **kw):
    opt = build_optimizer(model.parameters(), {"type": "AdamW", "lr": 1e-3})
    return Trainer(model, opt, _tiny_loader(), torch.device("cpu"), loss_fn=_lm_loss,
                   save_dir=str(tmp_path), name="t", log_interval=100, **kw)


def test_grad_accumulation_counts_optimizer_steps(tmp_path):
    set_seed(0)
    trainer = _make_trainer(_gpt(), tmp_path, accum_steps=2)
    # 4 batches/epoch, accum 2 -> 2 optimizer steps/epoch
    trainer.train(epochs=1)
    ckpt = torch.load(tmp_path / "t" / "last.pt", weights_only=False)
    assert ckpt["step"] == 2


def test_checkpoint_resume_roundtrip(tmp_path):
    set_seed(0)
    t1 = _make_trainer(_gpt(), tmp_path, accum_steps=1)
    t1.train(epochs=1)
    saved_step = torch.load(tmp_path / "t" / "last.pt", weights_only=False)["step"]
    assert saved_step == 4

    # Fresh trainer resumes and continues from the saved step.
    t2 = _make_trainer(_gpt(), tmp_path)
    start = t2.load_checkpoint(tmp_path / "t" / "last.pt")
    assert start == 4
    t2.train(epochs=1, start_step=start)
    assert torch.load(tmp_path / "t" / "last.pt", weights_only=False)["step"] == 8


def test_early_stop_on_no_improvement(tmp_path):
    set_seed(0)
    model = _gpt()
    trainer = _make_trainer(model, tmp_path, eval_interval=1, monitor="val_loss",
                            mode="min", patience=1, save_best=True)
    trainer.valid_loader = _tiny_loader()
    # best is forced very low so eval never "improves" -> must early stop.
    trainer.best = -1.0
    trainer.train(epochs=5)
    assert trainer.bad_evals > trainer.patience


def test_grad_checkpointing_matches_plain():
    set_seed(0)
    plain = _gpt(grad_checkpoint=False)
    ckpt = _gpt(grad_checkpoint=True)
    ckpt.load_state_dict(plain.state_dict())
    x = torch.randint(0, 40, (2, 16))
    y = torch.randint(0, 40, (2, 16))
    _, l1 = plain(x, y)
    l1.backward()
    _, l2 = ckpt(x, y)
    l2.backward()
    assert torch.allclose(l1, l2, atol=1e-5)
    g1 = plain.tok_emb.emb.weight.grad
    g2 = ckpt.tok_emb.emb.weight.grad
    assert torch.allclose(g1, g2, atol=1e-4)
