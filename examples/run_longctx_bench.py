"""Long-context benchmark: train the same GPT under each attention variant at a
sweep of context lengths and compare val perplexity, throughput, and peak memory.

    python examples/run_longctx_bench.py --config configs/longctx_gpt.yaml
    python examples/run_longctx_bench.py --variants mha local_flex --ctx-lens 512 1024 2048
"""
import argparse
import csv
from pathlib import Path

import yaml

from transformerlab.bench import format_longctx_table, run_longctx_sweep
from transformerlab.bench.longctx import LONGCTX_COLUMNS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--ctx-lens", nargs="*", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    spec = {}
    if args.config:
        spec = yaml.safe_load(Path(args.config).read_text()) or {}
    name = spec.pop("name", "longctx_gpt")
    save_dir = spec.pop("save_dir", "saved")
    if args.variants:
        spec["variants"] = args.variants
    if args.ctx_lens:
        spec["ctx_lens"] = args.ctx_lens
    if args.steps is not None:
        spec["train_steps"] = args.steps
    if args.device:
        spec["device"] = args.device
    if spec.get("device") in (None, "auto"):
        spec.pop("device", None)

    rows = run_longctx_sweep(**spec)
    table = format_longctx_table(rows)
    print(table)

    out_dir = Path(save_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "longctx.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LONGCTX_COLUMNS)
        writer.writeheader()
        writer.writerows({c: r.get(c) for c in LONGCTX_COLUMNS} for r in rows)
    (out_dir / "longctx.md").write_text("```\n" + table + "\n```\n")
    print(f"\nsaved -> {out_dir}/longctx.csv, longctx.md")


if __name__ == "__main__":
    main()
