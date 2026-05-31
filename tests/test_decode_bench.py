import math

from transformerlab.bench.decode import DECODE_COLUMNS, run_decode_sweep


def _by_variant(rows):
    return {r["variant"]: r for r in rows}


def test_decode_sweep_finite_rows():
    rows = run_decode_sweep(
        variants=["mha", "mqa", "mla"],
        model={"dim": 64, "n_layers": 2, "num_heads": 4, "dropout": 0.0},
        prompt_len=16, decode_len=16, batch=2, vocab_size=40, device="cpu",
        warmup=0, seed=0,
    )
    assert len(rows) == 3
    for r in rows:
        for col in ["params", "kv_cache_MB", "prefill_ms", "decode_tok_s", "peak_mem_MB"]:
            assert col in r and math.isfinite(r[col])


def test_kv_cache_hierarchy():
    # The whole point: mqa caches fewer KV heads than mha, and mla less still.
    rows = _by_variant(run_decode_sweep(
        variants=["mha", "gqa", "mqa", "mla"],
        model={"dim": 128, "n_layers": 2, "num_heads": 8, "dropout": 0.0},
        prompt_len=32, decode_len=32, batch=1, vocab_size=40, device="cpu",
        warmup=0, seed=0,
    ))
    assert rows["gqa"]["kv_cache_MB"] < rows["mha"]["kv_cache_MB"]
    assert rows["mqa"]["kv_cache_MB"] < rows["gqa"]["kv_cache_MB"]
    assert rows["mla"]["kv_cache_MB"] < rows["mha"]["kv_cache_MB"]


def test_decode_columns_lead_with_cache():
    assert DECODE_COLUMNS[:3] == ["variant", "params", "kv_cache_MB"]
