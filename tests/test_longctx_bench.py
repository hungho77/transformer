import math

from transformerlab.bench.longctx import LONGCTX_COLUMNS, run_longctx_sweep


def test_longctx_sweep_tags_and_finite():
    rows = run_longctx_sweep(
        variants=["mha", "local_flex"],
        ctx_lens=[32, 64],
        model={"dim": 48, "n_layers": 2, "num_heads": 4, "dropout": 0.0},
        train_steps=3, batch_size=8, seed=0, device="cpu", num_workers=0,
    )
    # 2 variants x 2 context lengths
    assert len(rows) == 4
    assert {r["ctx_len"] for r in rows} == {32, 64}
    for r in rows:
        assert "variant" in r and r["ctx_len"] in (32, 64)
        assert math.isfinite(r["val_ppl"]) and r["val_ppl"] > 1.0
        assert math.isfinite(r["peak_mem_MB"])


def test_longctx_columns():
    assert LONGCTX_COLUMNS[:2] == ["variant", "ctx_len"]
