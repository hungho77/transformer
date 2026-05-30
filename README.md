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
| `local`  | sliding-window, banded mask (set `window_size`) | ✓ (banded mask) |
| `local_flex` | sliding-window via `flex_attention` block mask (true sparsity) | — (flex kernel) |
| `mla`    | multi-head latent attention (compressed KV cache + decoupled RoPE) | — (own kernel) |

```python
from transformerlab.attention import available_attentions, build_attention, AttentionConfig
print(available_attentions())   # ['flash','gqa','linear','local','local_flex','mha','mla','mqa','sdpa']
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

# Encoder-only BERT, masked-LM pretraining on tiny-shakespeare
python examples/train_bert.py --config configs/bert_char_tiny.yaml

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
for simplicity, and `local` (banded mask) materializes the score matrix — use
`local_flex` for true block-sparse windowed attention via `flex_attention`, which
skips out-of-window blocks and saves memory at long sequences (CUDA-accelerated;
falls back to the banded path when flex is unavailable).

### Quality vs efficiency

Speed/memory alone doesn't tell you what a cheaper variant *costs* in accuracy.
This trains the same GPT under each attention and reports validation perplexity
next to throughput and peak memory, marking the quality/cost Pareto frontier:

```bash
python examples/run_quality_bench.py --config configs/quality_gpt.yaml
```

```
   variant   params  train_loss  val_loss  val_ppl  tokens_per_s  peak_mem_MB  pareto
       mha  1783680        2.16      1.85     6.36     574420.80       776.69
      sdpa  1783680        2.16      1.85     6.36     721347.53       561.10
       gqa  1636224        2.16      1.84     6.29     722008.86       559.69       *
       mqa  1537920        2.15      1.85     6.34     758183.07       517.56       *
    linear  1783680        2.42      2.24     9.40     206523.57      1892.39
     local  1783680        2.12      1.81     6.12     592234.93       776.41       *
local_flex  1783680        2.12      1.81     6.12     592215.15       776.41       *
```

`mha == sdpa` quality (a fairness check); `gqa`/`mqa` cut params and memory at
near-equal quality; `linear` is the clearest quality cost. Writes
`saved/<name>/quality.{csv,md}`. GPT char-LM today; the harness takes a
`build_model`/`loss_fn`, so ViT (accuracy) and BERT (MLM) can be added later.

### Long context

Efficient attention only earns its keep at long context. This trains the same
GPT at growing context lengths and reports where each variant still runs, at
what quality and memory:

```bash
python examples/run_longctx_bench.py --config configs/longctx_gpt.yaml
```

Peak memory (MB) per context length — `mha`/`local` (dense score matrix) grow
quadratically and OOM at 4096, while the efficient variants scale and keep going:

```
variant       512    1024    2048    4096      val_ppl@4096
mha           442    1504    5551    OOM       —
local         442    1504    5551    OOM       —      (banded mask: no savings)
sdpa          194     366     712    1389      10.57
mqa           181     342     664    1293      10.77
local_flex    202     382     744    1453      10.23   (best quality)
linear        794    1567    3112    6202      12.58   (cheap FLOPs, weak quality)
```

`local_flex` gives the best perplexity at 4096 using ~4× less memory than `mha`
would; `mha`/`local` can't run there at all. Writes `saved/<name>/longctx.{csv,md}`.

### Decode / KV cache

Generation is bottlenecked by the **KV cache**, not prefill. This drives each
variant's incremental decode and reports the actual cached bytes, prefill
latency, decode throughput, and peak memory:

```bash
python examples/run_decode_bench.py --config configs/decode_gpt.yaml
```

```
variant   params    kv_cache_MB  prefill_ms  decode_tok_s  peak_mem_MB
mha       25305600     96.00        4.04        1473        206.70
sdpa      25305600     96.00        3.58        1589        206.70
gqa       23208448     48.00        3.47        1428        157.70   (½ the KV heads)
mqa       21635584     12.00        3.25        1471        116.67   (one KV head)
mla       24920576     27.00        4.62        1053        149.99   (compressed latent)
```

The cache size follows what each variant stores per token (K/V are cached
*before* the GQA/MQA head-broadcast): `mha`/`sdpa` keep full K+V; `gqa` halves
it; `mqa` keeps one KV head; `mla` caches a compressed latent + shared rope key.
**MLA shrinks the cache ~3.5× vs MHA** while keeping near-MHA quality (see the
quality bench), trading a little decode speed for it; `mqa` shrinks it most but
costs more quality. Writes `saved/<name>/decode.{csv,md}`.

## Layout

```
src/transformerlab/
  attention/   base contract, registry, shared SDP core, and all variants
  layers/      embeddings, rotary, norms, FFN/SwiGLU, generic TransformerBlock
  models/      base, dataclass configs, GPT, ViT, EncoderDecoder, BERT
  data/        char-level LM, vision, synthetic seq2seq, masked-LM tasks
  train/       task-agnostic Trainer, optim/schedule, YAML run config
  bench/       latency/memory/FLOPs sweep + quality, long-context, decode harnesses
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
pytest -q          # shapes, causal-mask, equivalence, layers, models, bench
flake8 src tests examples
```
