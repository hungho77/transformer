import pytest
import torch

from transformerlab.models import (
    GPT,
    GPTConfig,
    EncoderDecoder,
    Seq2SeqConfig,
    VisionTransformer,
    ViTConfig,
)


def _backward_ok(model):
    return any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("attention_name", ["mha", "sdpa", "gqa", "linear", "local"])
def test_gpt_forward_backward(attention_name):
    kw = dict(vocab_size=50, max_seq_len=32, dim=48, n_layers=2, num_heads=4, attention_name=attention_name)
    if attention_name == "gqa":
        kw["num_kv_heads"] = 2
    if attention_name == "local":
        kw["window_size"] = 8
    model = GPT(GPTConfig(**kw))
    x = torch.randint(0, 50, (3, 16))
    logits, loss = model(x, x)
    assert logits.shape == (3, 16, 50)
    loss.backward()
    assert _backward_ok(model)


def test_gpt_generate_shapes():
    model = GPT(GPTConfig(vocab_size=50, max_seq_len=32, dim=48, n_layers=2, num_heads=4))
    out = model.generate(torch.zeros(2, 3, dtype=torch.long), max_new_tokens=10)
    assert out.shape == (2, 13)


def test_vit_forward_backward():
    model = VisionTransformer(ViTConfig(image_size=16, patch_size=4, dim=48, n_layers=2, num_heads=4, num_classes=5))
    logits, loss = model(torch.randn(3, 3, 16, 16), torch.randint(0, 5, (3,)))
    assert logits.shape == (3, 5)
    loss.backward()
    assert _backward_ok(model)


def test_seq2seq_forward_backward():
    cfg = Seq2SeqConfig(src_vocab_size=30, tgt_vocab_size=30, dim=48,
                        n_encoder_layers=2, n_decoder_layers=2, num_heads=4)
    model = EncoderDecoder(cfg)
    src = torch.randint(3, 30, (3, 10))
    tgt = torch.randint(3, 30, (3, 11))
    logits, loss = model(src, tgt, targets=tgt)
    assert logits.shape == (3, 11, 30)
    loss.backward()
    assert _backward_ok(model)
