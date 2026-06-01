"""Decode / KV-cache benchmark.

The training and long-context benchmarks measure the *prefill* (one forward over
the whole sequence). But the bottleneck for autoregressive *generation* is the
**KV cache**: every new token reads (and grows) the cached keys/values. This
harness drives GPT's incremental-decode path under each attention variant and
reports prefill latency, decode throughput, the actual **cached bytes**, and peak
memory — the regime where MLA's compressed cache (and MQA/GQA's fewer KV heads)
pays off.

What gets cached, per token, by construction (see ProjAttention.forward, which
updates the cache *before* the GQA/MQA head-broadcast):
  mha/sdpa/local/...  -> full K+V  = 2·H·head_dim
  gqa                 -> 2·H_kv·head_dim   (H_kv < H)
  mqa                 -> 2·1·head_dim
  mla                 -> latent c_KV (d_c) + shared k_R (d_rope)   ← smallest
"""
from __future__ import annotations

import time

import torch

from ..attention import KVCache, available_attentions
from ..models import GPT, GPTConfig
from ..utils import set_seed
from .sweep import _free_memory, format_table

DECODE_COLUMNS = ["variant", "params", "kv_cache_MB", "prefill_ms", "decode_tok_s", "peak_mem_MB"]

_DTYPES = {"float32": torch.float32, "fp32": torch.float32,
           "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
           "float16": torch.float16, "fp16": torch.float16}


def _kv_cache_mb(caches):
    # Sum the bytes actually held in every layer's KV cache (the decode footprint).
    total = 0
    for c in caches:
        for t in (c.k, c.v):
            if t is not None:
                total += t.numel() * t.element_size()
    return total / (1024 ** 2)


@torch.no_grad()
def benchmark_decode(variant, *, vocab_size, model_cfg, prompt_len, decode_len, batch,
                     device, num_heads, dtype="float32", warmup=2, seed=123):
    set_seed(seed)
    torch_dtype = _DTYPES[dtype]                              # cache bytes (element_size) scale with this
    kw = {k: v for k, v in model_cfg.items() if k != "attention_name"}
    kw["attention_name"] = variant
    kw["max_seq_len"] = max(kw.get("max_seq_len", 0), prompt_len + decode_len)
    if variant == "gqa" and "num_kv_heads" not in kw:
        kw["num_kv_heads"] = max(1, num_heads // 2)
    if variant in ("local", "local_flex", "sink") and not kw.get("window_size"):
        kw["window_size"] = max(8, (prompt_len + decode_len) // 4)

    # Cast the whole model to the inference dtype so K/V are cached at that width
    # (bf16/fp16 halves kv_cache_MB) and flash/sdpa use their fp16/bf16 kernels.
    model = GPT(GPTConfig(vocab_size=vocab_size, **kw)).to(device=device, dtype=torch_dtype).eval()
    params = sum(p.numel() for p in model.parameters())
    prompt = torch.randint(0, vocab_size, (batch, prompt_len), device=device)

    def decode_run():
        # Prefill the prompt into fresh caches, then generate decode_len tokens greedily.
        caches = [KVCache() for _ in model.blocks]
        logits = model._forward_cached(prompt, caches, is_causal=True)[:, -1]
        for _ in range(decode_len):
            nxt = logits.argmax(-1, keepdim=True)             # greedy next token
            logits = model._forward_cached(nxt, caches, is_causal=False)[:, -1]
        return caches

    for _ in range(warmup):                                   # warm caches/compile
        decode_run()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    # --- timed: prefill separately from the decode loop ---
    caches = [KVCache() for _ in model.blocks]
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = model._forward_cached(prompt, caches, is_causal=True)[:, -1]
    if device.type == "cuda":
        torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    for _ in range(decode_len):
        nxt = logits.argmax(-1, keepdim=True)
        logits = model._forward_cached(nxt, caches, is_causal=False)[:, -1]
    if device.type == "cuda":
        torch.cuda.synchronize()
    decode_s = time.perf_counter() - t1

    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
    return {
        "variant": variant,
        "params": params,
        "kv_cache_MB": _kv_cache_mb(caches),
        "prefill_ms": prefill_ms,
        "decode_tok_s": batch * decode_len / decode_s,
        "peak_mem_MB": peak,
    }


def run_decode_sweep(*, variants=None, model=None, prompt_len=256, decode_len=256, batch=1,
                     vocab_size=256, seed=123, device=None, dtype="float32", **kw):
    variants = variants or available_attentions()
    model = dict(model or {})
    num_heads = model.get("num_heads", 4)
    device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype not in ("float32", "fp32") and device.type != "cuda":
        dtype = "float32"                                    # fp16/bf16 decode only meaningful on CUDA
    rows = []
    for variant in variants:
        try:
            rows.append(benchmark_decode(
                variant, vocab_size=vocab_size, model_cfg=model, prompt_len=prompt_len,
                decode_len=decode_len, batch=batch, device=device, num_heads=num_heads,
                dtype=dtype, seed=seed, **kw,
            ))
        except Exception as exc:  # noqa: BLE001
            rows.append({**{c: float("nan") for c in DECODE_COLUMNS}, "variant": variant})
            print(f"  [skip] {variant}: {exc}")
        finally:
            _free_memory(device)
    return rows


def format_decode_table(rows) -> str:
    return format_table(rows, columns=DECODE_COLUMNS)
