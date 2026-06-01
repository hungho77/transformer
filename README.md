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

On a fresh CUDA server, `setup_env.sh` creates a venv, installs the CUDA build
of PyTorch, and (optionally) compiles flash-attn:

```bash
bash setup_env.sh --dev          # venv + torch + transformerlab + pytest/flake8
bash setup_env.sh --flash --dev  # also builds flash-attn (CUDA kernel, ~15 min)
```

See [INSTALL.md](INSTALL.md) for the full guide (manual steps, troubleshooting,
H100 notes). Benchmarks below were run on an H100 in `bfloat16`.

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
| `sink`   | StreamingLLM: first-k sink tokens + sliding window (set `window_size`, `extra.sink_size`) | ✓ (banded mask) |
| `alibi`  | linear-bias positions `m_h·(j−i)`, no embeddings (carries its own position) | ✓ (additive bias) |
| `mla`    | multi-head latent attention (compressed KV cache + decoupled RoPE) | — (own kernel) |

> `alibi` replaces positional embeddings with a per-head distance penalty, so it
> must be used with **no** token-position embedding. Wiring it into `GPT` cleanly
> needs a "no positional embedding" path (GPT adds a learned one when
> `use_rotary=false`); that's a documented follow-up — today `alibi` is usable and
> tested at the attention-module level.

```python
from transformerlab.attention import available_attentions, build_attention, AttentionConfig
print(available_attentions())   # ['flash','gqa','linear','local','local_flex','mha','mla','mqa','sdpa','sink']
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

## Training

The `Trainer` is task-agnostic (inject `loss_fn(model, batch) -> (loss, metrics)`)
and supports the features needed for real runs, all configurable from YAML:

- **Gradient accumulation** (`accum_steps`) — large effective batch on one GPU;
  step counts are in *optimizer* steps regardless of the accumulation factor.
- **Checkpoint resume** — `save_checkpoint`/`load_checkpoint` persist model +
  optimizer + scheduler + AMP scaler + RNG + step, so `--resume saved/<run>/last.pt`
  continues bit-for-bit.
- **Best checkpoint + early stopping** (`monitor`/`mode`/`patience`/`save_best`)
  — track a val metric, keep `best.pt`, stop when it plateaus.
- **Gradient checkpointing** (`model.grad_checkpoint: true`) — recompute block
  activations in the backward pass. ~**3.7× less** activation memory on an
  8-layer/dim-512/1k-context GPT (5087 → 1380 MB), enabling longer context /
  deeper models for ~30% extra compute.
- **AMP** (`amp: true`) mixed precision on CUDA.

```bash
python examples/train_gpt.py --config configs/gpt_char_tiny.yaml
python examples/train_gpt.py --config configs/gpt_char_tiny.yaml --resume saved/gpt_char_tiny/last.pt
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
       mha  1783680        2.16      1.82     6.20     593944.14       824.56
      sdpa  1783680        2.16      1.84     6.29     745674.18       609.25
     flash  1783680        2.16      1.84     6.29     765725.11       609.25
       gqa  1636224        2.16      1.83     6.25     569726.59       607.56
       mqa  1537920        2.15      1.81     6.12     649212.75       564.94       *
    linear  1783680        2.41      2.24     9.40     395539.02      1939.77
     local  1783680        2.12      1.82     6.15     419378.65       824.56
local_flex  1783680        2.12      1.82     6.15     390446.30       824.56
       mla  1760640        2.18      1.86     6.44     317852.11       896.24
```

`mha == sdpa == flash` quality (a fairness check — same math, different kernels);
`gqa`/`mqa` cut params and memory at near-equal quality (`mqa` is Pareto-optimal
here, lowest memory at best ppl); `local`/`local_flex` edge out full attention on
this tiny char-LM; `linear` is the clearest quality cost. Run in `bfloat16` on an
H100. Writes `saved/<name>/quality.{csv,md}`. GPT char-LM today; the harness takes
a `build_model`/`loss_fn`, so ViT (accuracy) and BERT (MLM) can be added later.

### Long context

Efficient attention only earns its keep at long context. This trains the same
GPT at growing context lengths and reports where each variant still runs, at
what quality and memory:

```bash
python examples/run_longctx_bench.py --config configs/longctx_gpt.yaml
```

Peak memory (MB) per context length — `mha`/`local`/`sink` materialize the dense
S×S score matrix so memory grows quadratically, while the fused/sparse variants
stay roughly linear. (Small model — dim 128, 2 layers — so all variants still fit
on an 80 GB H100 here; the memory *gap* is the point, ~14× at 4096.)

```
variant       512    1024    2048     4096      val_ppl@4096
mha           490    1552    5599    21385      10.57
local         490    1552    5599    21385      10.23   (banded mask: no savings)
sink          490    1552    5599    21385      10.57   (sinks+window, dense kernel)
sdpa          241     414     760     1451      10.57
flash         241     414     760     1451      10.57
mqa           229     390     711     1354      10.77
local_flex    249     430     792     1515      10.23   (best quality)
linear        842    1614    3160     6250      12.58   (cheap FLOPs, weak quality)
```

`sdpa`/`flash` deliver `mha`'s exact quality (10.57) at **~14× less memory** at
4096; `local_flex` gives the best perplexity (10.23) via true block-sparsity at
similar memory. `mha`/`local`/`sink` blow up quadratically and would OOM first on
a real-scale model. `flash` and `sdpa` are memory-identical here (both fused, no
score matrix). Writes `saved/<name>/longctx.{csv,md}`.

> **Note on `sink`:** StreamingLLM's memory win comes from *evicting* tokens
> outside the sink+window from the KV cache during decoding. This repo's `sink`
> implements the attention *pattern* (a banded mask + sink columns) on the
> shared dense kernel, so in single-pass prefill it costs exactly what `mha`/
> `local` do — the table above shows that honestly (it tracks `mha` quality
> because with this window the kept set covers most of the short contexts). It's
> an educational variant for the sink concept; a bounded streaming cache would
> need cache eviction in `KVCache`, which is not yet implemented.

### Decode / KV cache

Generation is bottlenecked by the **KV cache**, not prefill. This drives each
variant's incremental decode and reports the actual cached bytes, prefill
latency, decode throughput, and peak memory:

```bash
python examples/run_decode_bench.py --config configs/decode_gpt.yaml
```

```
variant    params    kv_cache_MB  prefill_ms  decode_tok_s  peak_mem_MB
mha        25305600     48.00        4.65        1251.55       131.29
sdpa       25305600     48.00        2.92        1266.16       131.29
flash      25305600     48.00        3.74        1038.24       131.29   (full K/V, like sdpa)
gqa        23208448     24.00        3.31         933.83       108.35   (½ the KV heads)
mqa        21635584      6.00        4.03        1102.23        86.27   (one KV head)
mla        24920576     13.50        5.01         862.64       102.94   (compressed latent)
```

The cache size follows what each variant stores per token (K/V are cached
*before* the GQA/MQA head-broadcast): `mha`/`sdpa`/`flash` keep full K+V; `gqa`
halves it; `mqa` keeps one KV head; `mla` caches a compressed latent + shared
rope key. **MQA shrinks the cache 8× vs MHA** (one KV head) and **MLA ~3.5×**
while keeping near-MHA quality (see the quality bench). `flash` caches full K/V
(same bytes as `sdpa`); its single-token-step kernel has no edge over `sdpa` at
decode here. Run in `bfloat16` (`dtype: bfloat16`), so all `kv_cache_MB` are half
their fp32 size. Writes `saved/<name>/decode.{csv,md}`.

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
