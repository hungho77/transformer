"""Benchmark attention variants and print a comparison table.

    python examples/run_bench.py --config configs/bench_attention.yaml
    python examples/run_bench.py --variants mha sdpa linear --seq-lens 512 1024
"""
import argparse
from pathlib import Path

import yaml

from transformerlab.bench import format_table, run_sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--seq-lens", nargs="*", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    args = ap.parse_args()

    spec = {}
    if args.config:
        spec = yaml.safe_load(Path(args.config).read_text()) or {}
    if args.variants:
        spec["variants"] = args.variants
    if args.seq_lens:
        spec["seq_lens"] = args.seq_lens
    if args.device:
        spec["device"] = args.device
    if args.dtype:
        spec["dtype"] = args.dtype

    rows = run_sweep(**spec)
    print(format_table(rows))


if __name__ == "__main__":
    main()
