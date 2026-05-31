"""Decode / KV-cache benchmark: drive each attention variant's incremental
generation and compare cached bytes, prefill latency, decode throughput, memory.

    python examples/run_decode_bench.py --config configs/decode_gpt.yaml
    python examples/run_decode_bench.py --variants mha mqa mla --decode-len 512
"""
import argparse
import csv
from pathlib import Path

import yaml

from transformerlab.bench import format_decode_table, run_decode_sweep
from transformerlab.bench.decode import DECODE_COLUMNS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--prompt-len", type=int, default=None)
    ap.add_argument("--decode-len", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    spec = {}
    if args.config:
        spec = yaml.safe_load(Path(args.config).read_text()) or {}
    name = spec.pop("name", "decode_gpt")
    save_dir = spec.pop("save_dir", "saved")
    if args.variants:
        spec["variants"] = args.variants
    if args.prompt_len is not None:
        spec["prompt_len"] = args.prompt_len
    if args.decode_len is not None:
        spec["decode_len"] = args.decode_len
    if args.device:
        spec["device"] = args.device
    if spec.get("device") in (None, "auto"):
        spec.pop("device", None)

    rows = run_decode_sweep(**spec)
    table = format_decode_table(rows)
    print(table)

    out_dir = Path(save_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "decode.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DECODE_COLUMNS)
        writer.writeheader()
        writer.writerows({c: r.get(c) for c in DECODE_COLUMNS} for r in rows)
    (out_dir / "decode.md").write_text("```\n" + table + "\n```\n")
    print(f"\nsaved -> {out_dir}/decode.csv, decode.md")


if __name__ == "__main__":
    main()
