# Installation & Environment Setup

This guide covers setting up `transformerlab` for training on a CUDA server.
Tested on: **Python 3.10, CUDA 12.2, NVIDIA H100**.

---

## Requirements

| Requirement | Minimum |
|-------------|---------|
| Python      | 3.10+   |
| CUDA        | 11.8+ (12.x recommended) |
| PyTorch     | 2.2+    |
| Disk        | ~5 GB (PyTorch + venv) |

---

## One-command setup

```bash
bash setup_env.sh            # core (torch + transformerlab)
bash setup_env.sh --dev      # + pytest / flake8
bash setup_env.sh --flash    # + flash-attn (CUDA kernel build, ~15 min)
bash setup_env.sh --flash --dev   # everything
```

The script will:
1. Check Python ≥ 3.10
2. Create `./venv/` (skips if it already exists)
3. Install PyTorch with the CUDA 12.1 wheel (binary-compatible with CUDA 12.2)
4. Install `transformerlab` in editable mode
5. Optionally build `flash-attn`
6. Run a smoke test (prints GPU name + available attention variants)

---

## Manual step-by-step

If you prefer to install by hand:

```bash
# 1. Create and activate the virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install PyTorch for CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install the package (editable + dev tools)
pip install -e ".[dev]"

# 5. (Optional) flash-attn — requires CUDA compiler, takes ~15 min on H100
pip install flash-attn --no-build-isolation

# 6. Verify
python -c "import torch; print(torch.cuda.get_device_name(0))"
python -c "from transformerlab.attention import available_attentions; print(available_attentions())"
```

---

## Activating the environment

```bash
source venv/bin/activate          # activate
deactivate                        # deactivate when done
```

Add this line to your `~/.bashrc` or `~/.zshrc` for convenience:
```bash
alias tlab='source /mnt/data/sftp/data/hunght23/workspace/projects/transformer/venv/bin/activate'
```

---

## Running training

All scripts live in `examples/` and take a `--config` YAML:

```bash
# GPT char-LM on tiny-shakespeare (trains in ~1 min on H100)
python examples/train_gpt.py --config configs/gpt_char_tiny.yaml

# Override attention variant at the CLI (no config edit needed)
python examples/train_gpt.py --config configs/gpt_char_tiny.yaml --attention sdpa
python examples/train_gpt.py --config configs/gpt_char_tiny.yaml --attention gqa
python examples/train_gpt.py --config configs/gpt_char_tiny.yaml --attention mla

# Sample from a trained checkpoint
python examples/sample_gpt.py --ckpt saved/gpt_char_tiny --prompt "ROMEO:"

# Vision Transformer on CIFAR-10 (use --dataset fake to skip download)
python examples/train_vit.py --config configs/vit_cifar10.yaml --dataset fake

# Encoder-decoder copy task
python examples/train_seq2seq.py --config configs/seq2seq_copy.yaml

# BERT masked-LM pretraining
python examples/train_bert.py --config configs/bert_char_tiny.yaml
```

---

## Benchmarks

```bash
# Latency / memory / FLOPs sweep across attention variants × sequence lengths
python examples/run_bench.py --config configs/bench_attention.yaml

# Quality (val-ppl) vs throughput/memory — marks Pareto-optimal variants
python examples/run_quality_bench.py --config configs/quality_gpt.yaml

# Memory scaling vs context length (shows where MHA OOMs)
python examples/run_longctx_bench.py --config configs/longctx_gpt.yaml

# KV-cache size and decode throughput (MLA / MQA / GQA wins here)
python examples/run_decode_bench.py --config configs/decode_gpt.yaml
```

---

## Tests & lint

```bash
pytest -q                                             # ~55 tests, ~4 s on CPU
pytest tests/test_attention_equivalence.py -q        # equivalence checks only
flake8 src tests examples                            # max-line 120 (see .flake8)
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: transformerlab` | Run `pip install -e .` inside the venv |
| `CUDA not available` | Check `nvidia-smi`; make sure the cu121 PyTorch wheel is installed |
| `flash-attn` build fails | Ensure `nvcc` is on `PATH` (`which nvcc`); CUDA dev headers must match the driver |
| `local_flex` falls back silently | `flex_attention` requires PyTorch ≥ 2.5 and CUDA; check `torch.__version__` |
| OOM on long-context bench | Use `sdpa`, `local_flex`, or `mqa`; `mha`/`local` materialise the full score matrix |

---

## Hardware notes (H100 80 GB)

- The cu121 PyTorch wheel enables BF16 Tensor Cores and `torch.compile` on H100.
- `flash-attn` is recommended for maximum throughput at long sequences.
- `local_flex` (via `flex_attention`) gives the best quality/memory trade-off at
  context lengths ≥ 2048 where `mha` OOMs.
- All benchmarks default to `bfloat16` on CUDA (`torch.autocast`).
