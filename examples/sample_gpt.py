"""Generate text from a trained char-level GPT checkpoint.

    python examples/sample_gpt.py --ckpt saved/gpt_char_tiny --prompt "ROMEO:"
"""
import argparse
from pathlib import Path

import torch

from transformerlab.models import GPT, GPTConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="run dir containing last.pt + meta.pt")
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    args = ap.parse_args()

    run_dir = Path(args.ckpt)
    meta = torch.load(run_dir / "meta.pt", weights_only=False)
    state = torch.load(run_dir / "last.pt", weights_only=False)

    model = GPT(GPTConfig(**meta["model_cfg"]))
    model.load_state_dict(state["model"])
    model.eval()

    stoi, itos = meta["stoi"], meta["itos"]
    idx = torch.tensor([[stoi[c] for c in args.prompt]], dtype=torch.long)
    out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=args.top_k)
    print("".join(itos[int(i)] for i in out[0]))


if __name__ == "__main__":
    main()
