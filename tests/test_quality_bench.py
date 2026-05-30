import math

from transformerlab.bench.quality import QUALITY_COLUMNS, mark_pareto, run_quality_sweep


def test_quality_sweep_returns_finite_rows():
    rows = run_quality_sweep(
        variants=["mha", "sdpa"],
        model={"max_seq_len": 32, "dim": 48, "n_layers": 2, "num_heads": 4, "dropout": 0.0},
        data={"val_frac": 0.1, "max_eval_batches": 3},
        train_steps=5, batch_size=8, seed=0, device="cpu", num_workers=0,
    )
    assert len(rows) == 2
    for r in rows:
        for col in ["params", "train_loss", "val_loss", "val_ppl", "tokens_per_s", "peak_mem_MB"]:
            assert col in r and math.isfinite(r[col])
        assert r["val_ppl"] > 1.0


def test_mha_sdpa_quality_matches():
    # Equivalent attention + same seed -> near-identical training/eval loss.
    rows = run_quality_sweep(
        variants=["mha", "sdpa"],
        model={"max_seq_len": 32, "dim": 48, "n_layers": 2, "num_heads": 4, "dropout": 0.0},
        data={"val_frac": 0.1, "max_eval_batches": 3},
        train_steps=5, batch_size=8, seed=0, device="cpu", num_workers=0,
    )
    by = {r["variant"]: r for r in rows}
    assert abs(by["mha"]["val_loss"] - by["sdpa"]["val_loss"]) < 1e-3


def test_mark_pareto_flags_frontier():
    rows = [
        {"variant": "a", "val_ppl": 10.0, "peak_mem_MB": 100.0},   # dominated by c
        {"variant": "b", "val_ppl": 8.0, "peak_mem_MB": 200.0},    # frontier (best ppl)
        {"variant": "c", "val_ppl": 9.0, "peak_mem_MB": 50.0},     # frontier (best mem)
    ]
    mark_pareto(rows)
    flags = {r["variant"]: r["pareto"] for r in rows}
    assert flags == {"a": False, "b": True, "c": True}


def test_quality_columns_complete():
    assert QUALITY_COLUMNS[0] == "variant" and "pareto" in QUALITY_COLUMNS
