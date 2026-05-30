"""Quality x efficiency benchmark: train a GPT under each attention variant and
compare validation perplexity against throughput and peak memory.

    python examples/run_quality_bench.py --config configs/quality_gpt.yaml
    python examples/run_quality_bench.py --variants mha sdpa --steps 50
"""
import argparse
import csv
from pathlib import Path

import yaml

from transformerlab.bench import format_quality_table, mark_pareto, run_quality_sweep
from transformerlab.bench.quality import QUALITY_COLUMNS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    spec = {}
    if args.config:
        spec = yaml.safe_load(Path(args.config).read_text()) or {}
    name = spec.pop("name", "quality_gpt")
    save_dir = spec.pop("save_dir", "saved")
    if args.variants:
        spec["variants"] = args.variants
    if args.steps is not None:
        spec["train_steps"] = args.steps
    if args.device:
        spec["device"] = args.device
    if spec.get("device") in (None, "auto"):
        spec.pop("device", None)  # let run_quality_sweep pick cuda/cpu

    rows = mark_pareto(run_quality_sweep(**spec))
    table = format_quality_table(rows)
    print(table)

    out_dir = Path(save_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "quality.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=QUALITY_COLUMNS)
        writer.writeheader()
        writer.writerows({c: r.get(c) for c in QUALITY_COLUMNS} for r in rows)
    (out_dir / "quality.md").write_text("```\n" + table + "\n```\n")
    print(f"\nsaved -> {out_dir}/quality.csv, quality.md")


if __name__ == "__main__":
    main()
