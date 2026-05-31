#!/usr/bin/env bash
# setup_env.sh — create venv and install transformerlab for training on this server
# Usage:
#   bash setup_env.sh              # core install (no flash-attn)
#   bash setup_env.sh --flash      # + flash-attn (heavy CUDA build, recommended for H100)
#   bash setup_env.sh --dev        # + pytest / flake8
#   bash setup_env.sh --flash --dev

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_DIR/venv"
PYTHON="${PYTHON:-python3}"
# PyTorch CUDA wheel — cu121 is binary-compatible with CUDA 12.2
TORCH_INDEX="https://download.pytorch.org/whl/cu121"

INSTALL_FLASH=0
INSTALL_DEV=0
for arg in "$@"; do
  case "$arg" in
    --flash) INSTALL_FLASH=1 ;;
    --dev)   INSTALL_DEV=1  ;;
  esac
done

# ── 1. Python version check ───────────────────────────────────────────────────
PY_VER=$("$PYTHON" -c 'import sys; print(sys.version_info[:2])')
if "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then
  echo "Python $PY_VER — OK"
else
  echo "ERROR: Python 3.10+ required (found $PY_VER)" >&2; exit 1
fi

# ── 2. Create venv ────────────────────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
  echo "Reusing existing venv at $VENV_DIR"
else
  echo "Creating venv at $VENV_DIR"
  "$PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet

# ── 3. PyTorch with CUDA ──────────────────────────────────────────────────────
echo "Installing PyTorch (CUDA 12.1 wheel, compatible with CUDA 12.2)..."
pip install torch torchvision --index-url "$TORCH_INDEX" --quiet

# ── 4. Core package ───────────────────────────────────────────────────────────
echo "Installing transformerlab (editable)..."
if [ "$INSTALL_DEV" -eq 1 ]; then
  pip install -e "$REPO_DIR/[dev]" --quiet
else
  pip install -e "$REPO_DIR" --quiet
fi

# ── 5. Optional: flash-attn ───────────────────────────────────────────────────
if [ "$INSTALL_FLASH" -eq 1 ]; then
  echo "Installing flash-attn (this compiles CUDA kernels — may take 10–20 min)..."
  pip install wheel packaging psutil ninja --quiet
  pip install flash-attn --no-build-isolation
fi

# ── 6. Smoke test ─────────────────────────────────────────────────────────────
echo ""
echo "Verifying install..."
python - <<'PY'
import torch, transformerlab
from transformerlab.attention import available_attentions
print(f"  torch        {torch.__version__}")
print(f"  CUDA avail   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU          {torch.cuda.get_device_name(0)}")
print(f"  attentions   {available_attentions()}")
try:
    import flash_attn
    print(f"  flash-attn   {flash_attn.__version__}")
except ImportError:
    print("  flash-attn   not installed (optional)")
PY

echo ""
echo "Setup complete. Activate the environment with:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Quick-start commands:"
echo "  python examples/train_gpt.py --config configs/gpt_char_tiny.yaml"
echo "  python examples/run_bench.py  --config configs/bench_attention.yaml"
