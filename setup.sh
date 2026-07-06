#!/usr/bin/env bash
# setup.sh — Set up the revmax-aaai2027 Python environment.
# Tested on: Apple Silicon Mac (macOS Sonoma), Python 3.9.
# Usage:  bash setup.sh          # uses python3.9
#         bash setup.sh python3  # override interpreter
set -euo pipefail

PY=${1:-python3.9}

echo "=== revmax-aaai2027 environment setup ==="
echo "Python: $($PY --version 2>&1)"

# ── Create venv ────────────────────────────────────────────────────────────────
if [ ! -d venv ]; then
    echo "Creating venv..."
    $PY -m venv venv
else
    echo "venv already exists, skipping creation."
fi

source venv/bin/activate
pip install --upgrade pip --quiet

# ── Install dependencies ───────────────────────────────────────────────────────
echo "Installing requirements.txt (pinned versions)..."
pip install -r requirements.txt --quiet

# ── Verify critical imports ────────────────────────────────────────────────────
echo ""
echo "=== Import verification ==="
python - <<'EOF'
import sys
import torch
import torch_geometric
import networkx

print(f"Python    : {sys.version.split()[0]}")
print(f"torch     : {torch.__version__}")
print(f"MPS avail : {torch.backends.mps.is_available()}")
print(f"CUDA avail: {torch.cuda.is_available()}")
print(f"pyg       : {torch_geometric.__version__}")
print(f"networkx  : {networkx.__version__}")

# Spot check key project modules
import omegaconf, numpy, scipy, pandas, matplotlib, tqdm, gymnasium, stable_baselines3
print(f"omegaconf : {omegaconf.__version__}")
print(f"numpy     : {numpy.__version__}")
print(f"scipy     : {scipy.__version__}")
EOF

# ── Run tests ──────────────────────────────────────────────────────────────────
echo ""
echo "=== pytest ==="
# Run from project root so src/ is on path
pytest tests/ -q --tb=short 2>&1 | tail -5

echo ""
echo "=== Setup OK. ==="
echo ""
echo "IMPORTANT — Apple Silicon MPS notes:"
echo "  Phase 2 REINFORCE runs on CPU (set in experiment scripts) to avoid MPS"
echo "  autograd state corruption under torch.no_grad()."
echo "  If you see MPS errors in OTHER scripts, prefix them with:"
echo "    PYTORCH_ENABLE_MPS_FALLBACK=1 python ..."
echo ""
echo "Activate environment:  source venv/bin/activate"
echo "Quick sanity:          bash scripts/smoke_test.sh"
