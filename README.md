# transformerlab

A from-scratch PyTorch playground for **transformer architectures** and
**efficient attention** research. Every attention mechanism is a pluggable,
registry-selected component behind one interface, so the same model (GPT, ViT,
encoder-decoder) can run any attention variant and be **benchmarked head-to-head**
for speed, memory, and quality.

## Install

```bash
pip install -e .            # core (torch, torchvision, numpy, tqdm, pyyaml)
pip install -e ".[dev]"     # + pytest, flake8
pip install -e ".[flash]"   # + flash-attn (optional CUDA backend)
```

## Attention variants

Swap attention by changing one config field (`attention_name`):

| name     | description                                  | reuses SDP core |
|----------|----------------------------------------------|:---------------:|
| `mha`    | standard multi-head, explicit softmax (ref)  | ✓ |
| `sdpa`   | torch fused `scaled_dot_product_attention`   | ✓ |
| `flash`  | flash-attn backend, falls back to SDPA       | ✓ |
| `gqa`    | grouped-query (set `num_kv_heads`)           | ✓ |
| `mqa`    | multi-query (single KV head)                 | ✓ |
| `linear` | feature-map linear attention (O(S·d²))       | — (cumsum causal) |
| `local`  | sliding-window (set `window_size`)           | ✓ (banded mask) |

```python
from transformerlab.attention import available_attentions, build_attention, AttentionConfig
print(available_attentions())   # ['flash','gqa','linear','local','mha','mqa','sdpa']
attn = build_attention("gqa", AttentionConfig(dim=512, num_heads=8, num_kv_heads=2))
```

## Quickstart

```bash
# Decoder-only GPT on tiny-shakespeare, then sample
python examples/train_gpt.py --config configs/gpt_char_tiny.yaml
python examples/sample_gpt.py --ckpt saved/gpt_char_tiny --prompt "ROMEO:"

# Vision Transformer on CIFAR-10
python examples/train_vit.py --config configs/vit_cifar10.yaml

# Encoder-decoder on a synthetic copy task
python examples/train_seq2seq.py --config configs/seq2seq_copy.yaml

# Try any attention variant without editing the config
python examples/train_gpt.py --config configs/gpt_char_tiny.yaml --attention linear
```

## Benchmark

```bash
python examples/run_bench.py --config configs/bench_attention.yaml
```

Prints latency / peak memory / FLOPs per (variant × sequence length). The fused
`sdpa`/`flash` paths cut peak memory roughly an order of magnitude versus the
explicit `mha` at long sequences; `gqa`/`mqa` reduce KV cost; `linear` shows
sub-quadratic FLOP growth. Note: the reference `linear` cumsum path trades memory
for simplicity, and `local` currently uses a banded mask (no sparsity savings yet
— a flex_attention backend is the planned upgrade).

## Layout

```
src/transformerlab/
  attention/   base contract, registry, shared SDP core, and all variants
  layers/      embeddings, rotary, norms, FFN/SwiGLU, generic TransformerBlock
  models/      base, dataclass configs, GPT, ViT, EncoderDecoder
  data/        char-level LM, vision, synthetic seq2seq tasks
  train/       task-agnostic Trainer, optim/schedule, YAML run config
  bench/       latency/memory/FLOPs profiling + attention sweep
configs/   YAML run configs        examples/  runnable train/sample/bench scripts
tests/     shapes, causal mask, equivalence, layers, models, registry
```

## Adding a new attention variant

Subclass `ProjAttention` (shared q/k/v/o projections + head reshaping) and
implement `_attend`, then register it:

```python
from transformerlab.attention import ProjAttention, register_attention, sdpa_core

@register_attention("myattn")
class MyAttention(ProjAttention):
    def _attend(self, q, k, v, attn_mask, is_causal):
        k, v = self._maybe_repeat_kv(k, v)   # handles GQA/MQA grouping
        return sdpa_core(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
```

Add an import in `attention/__init__.py` and it's instantly usable everywhere
via `attention_name: myattn`.

## Tests

```bash
pytest -q          # 41 tests: shapes, causal-mask, equivalence, layers, models
flake8 src tests examples
```
