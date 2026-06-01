# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`transformerlab` — a from-scratch PyTorch package for implementing transformer
architectures and experimenting with efficient attention. Installable `src/`
layout package; import name is `transformerlab` (the repo folder is `transformer`).

The defining design constraint: **attention is a pluggable, registry-selected
component behind one interface**, so the same model can run any attention variant
and be benchmarked head-to-head.

## Commands

```bash
pip install -e ".[dev]"        # editable install + pytest/flake8
pytest -q                      # full suite (~73 tests, CPU, ~5s)
pytest tests/test_attention_equivalence.py -q   # single test file
flake8 src tests examples      # lint (config in .flake8, max-line 120)

python examples/train_gpt.py --config configs/gpt_char_tiny.yaml [--steps N] [--attention NAME] [--precision bf16]
python examples/sample_gpt.py --ckpt saved/gpt_char_tiny --prompt "ROMEO:"
python examples/train_vit.py --config configs/vit_cifar10.yaml [--dataset fake]
python examples/train_seq2seq.py --config configs/seq2seq_copy.yaml
python examples/train_bert.py --config configs/bert_char_tiny.yaml           # masked-LM pretraining
python examples/run_bench.py --config configs/bench_attention.yaml           # speed/memory/FLOPs sweep
python examples/run_quality_bench.py --config configs/quality_gpt.yaml       # val-ppl vs throughput/memory + Pareto
python examples/run_longctx_bench.py --config configs/longctx_gpt.yaml       # quality/memory vs context length
```

`train_vit.py --dataset fake` avoids the CIFAR-10 download (random images) for quick smoke runs.

## Architecture

The dependency direction is `attention → layers → models`; nothing in `layers`
imports `models`. Key pieces:

- **`attention/`** is the heart. `base.py` defines the contract: `AttentionConfig`,
  `MaskType`, `KVCache`, and `AttentionBase` (forward signature
  `(x, *, kv, attn_mask, is_causal, rotary, kv_cache)`). `core.py` provides
  `ProjAttention` — the shared base that owns q/k/v/o projections, head reshaping,
  rotary application, KV-cache update, and GQA/MQA head broadcasting (`repeat_kv`).
  **Most variants only override `_attend(q, k, v, attn_mask, is_causal)`** with
  q/k/v in `[B, H, S, head_dim]`. `sdpa_core` is the explicit reference kernel.
- **Registry** (`attention/registry.py`): variants self-register via
  `@register_attention("name")`. `attention/__init__.py` imports every variant
  module so registration happens on package import. `build_attention(name, cfg)`
  is the only construction path used by models.
- **Mask convention** (matches `F.scaled_dot_product_attention`): boolean mask
  `True` = allowed/keep; float mask = additive. Causal is requested via the
  `is_causal` flag, not a pre-built mask, so backends can fuse it.
- **`layers/block.py`**: `TransformerBlock` is architecture-agnostic and
  parameterized by `BlockConfig` (pre/post-norm, RMS/Layer norm, MLP/SwiGLU,
  optional `cross_attention`). It selects attention by `attention_name`.
  `make_blocks()` builds a stack and accepts a per-layer list of attention names
  (heterogeneous-attention models).
- **`models/`**: `GPT` (causal, RoPE by default, KV-cache `generate`), `VisionTransformer`
  (patch embed + cls/mean pool), `EncoderDecoder` (cross-attention decoder), `BERT`
  (bidirectional encoder + MLM head + token-type/segment embeddings).
  Configs are dataclasses in `models/configs.py`; YAML run files load into
  `RunConfig` (`train/config.py`) and the model dataclasses.
- **`train/trainer.py`**: task-agnostic loop. The task is injected as
  `loss_fn(model, batch) -> (loss, metrics_dict)` — see the `lm_loss` /
  `classification_loss` / `seq2seq_loss` closures in the example scripts.
  Supports gradient accumulation (`accum_steps`; step counts are optimizer
  steps), full-state `save_checkpoint`/`load_checkpoint` (model+opt+sched+scaler
  +RNG+step) for `--resume`, and best-ckpt + early stop (`monitor`/`mode`/
  `patience`/`save_best`). Mixed precision via `amp_dtype`
  (`"fp32"`/`"bf16"`/`"fp16"`; `--precision` on `train_gpt.py`): autocasts
  train+eval, and uses `GradScaler` only for fp16 (bf16 keeps fp32's exponent
  range so it needs no loss scaling — the scaler stays disabled and the
  grad-accum path is unchanged). GPT honors `GPTConfig.grad_checkpoint` (wraps
  blocks in `torch.utils.checkpoint`, training-only) for ~3.7x activation-memory
  savings.
- **`bench/`**: `sweep.py` profiles attention modules (latency/peak-mem/FLOPs);
  `quality.py` trains the same GPT under each variant and reports val perplexity
  vs throughput/memory with a Pareto flag (`mark_pareto`). Both reuse
  `_free_memory` (gc + `empty_cache` + `torch._dynamo.reset()`) between rows so
  the compiled `local_flex` kernel doesn't OOM later rows; `measure_flops`
  degrades to `nan` for ops it can't trace rather than aborting a row.
  `longctx.py` is a thin wrapper over `run_quality_sweep` that sweeps
  `model.max_seq_len` to compare variants as context length grows: `mha`/`local`/
  `sink` materialize the dense S×S scores (quadratic memory; OOM first at scale)
  while `sdpa`/`local_flex`/`mqa`/`flash` stay roughly linear (~14x less memory at
  ctx 4096 in the bundled bf16 bench).
  `decode.py` drives GPT's incremental-decode path (`_forward_cached` + per-layer
  `KVCache`) and reports **actual cached bytes** + decode throughput — the regime
  where `mqa`/`gqa` (fewer cached KV heads) and `mla` (compressed latent) win.

## Conventions & gotchas

- **Math-first comments (required)**: every implementation that realises a
  mathematical operation (attention kernels, feature maps, norms, rotary,
  positional schemes, loss/optim math) MUST carry comments stating the
  underlying equation so the code is readable against the math. Use a Sphinx
  `:math:` docstring for the variant/function summary and inline `#` comments
  tying each tensor op to a symbol — e.g. `# scores = QKᵀ / √d_k` above the
  matmul, `# softmax over keys (last dim)` above the softmax. Name tensors after
  their symbols where practical (`scale = 1/√d`, `phi_q`, `c_KV`). `linear.py`
  and `mla.py` are the reference style. Keep equations correct when refactoring;
  a comment that drifts from the code is a bug.
- **Adding an attention variant**: subclass `ProjAttention`, implement `_attend`
  (call `self._maybe_repeat_kv(k, v)` first to support GQA/MQA), decorate with
  `@register_attention`, and add its import to `attention/__init__.py`. It is then
  usable everywhere via `attention_name`.
- **Equivalence is a tested invariant**: exact variants must match given shared
  weights — `mha == sdpa`, `gqa(num_kv_heads==num_heads) == mha`,
  `local(window>=S) == mha`, `local_flex == local` (see
  `tests/test_attention_equivalence.py`). Preserve this when touching the SDP
  core or projection logic.
- `linear`, `local`, `local_flex`, and `sink` advertise limited mask support via
  `supports_mask` (no arbitrary `FULL` masks). The reference `linear` causal path
  uses `cumsum` (memory-heavy by design); `local` uses a banded mask (correct but
  no sparsity savings); `local_flex` uses `flex_attention` block masks for true
  block-sparsity and falls back to the banded path when flex is unavailable, an
  explicit `attn_mask` is passed, or attention dropout is active.
- `sink` (StreamingLLM attention sinks) keeps the first `extra["sink_size"]`
  tokens *plus* the last `window_size` keys: `make_sink_window_mask` (in
  `core.py`) is `make_sliding_window_mask` unioned with the sink columns
  (causally). It runs on the dense `sdpa_core` kernel, so it costs the same as
  `local` in a single pass — it models the sink *pattern*, not the bounded
  streaming cache (KVCache has no eviction yet). Equivalence invariant:
  `sink(sink_size=0, window>=S) == local` (tested).
- `flash` requires CUDA + fp16/bf16 and falls back to SDPA otherwise; flash-attn
  is an optional dependency, never required.
- `mla` (multi-head latent attention) subclasses `AttentionBase` directly (not
  `ProjAttention`) — it has its own projection structure (KV down/up to a small
  latent `c_KV`, decoupled per-head RoPE with a shared rope key) and carries its
  own RoPE (can't import `layers/` — that's circular). Its latent dims are read
  from `AttentionConfig.extra` (kv_latent_dim/q_latent_dim/nope/rope/v_head_dim);
  GPT forwards `GPTConfig.extra` into the block. It manages position itself, so
  use with `use_rotary=True` (GPT then adds no absolute pos emb) and the model's
  `rotary` arg is ignored. The KV cache stores only `c_KV`+shared `k_R`, not
  per-head K/V — its win shows up in generation memory, not single-pass training.
- This repo was previously a broken fork of victoresque/pytorch-template (MNIST
  demo). All of that was deleted; only `utils/util.py` (now pandas-free,
  `src/transformerlab/utils/`), `.flake8`, and `.gitignore` were carried over.
