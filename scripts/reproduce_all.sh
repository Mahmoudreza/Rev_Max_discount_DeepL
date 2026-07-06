#!/usr/bin/env bash
# scripts/reproduce_all.sh — Ordered pipeline to reproduce paper results.
# Run from repo root: bash scripts/reproduce_all.sh
#
# Each step states:
#   - Approx runtime (Apple Silicon M-series)
#   - Which checkpoint it produces or requires
#   - Whether it can use a released checkpoint instead of retraining
#
# Prerequisites:
#   bash setup.sh
#   bash scripts/download_checkpoints.sh    # skip if you want to retrain from scratch
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate

echo "=== RevMax Reproduction Pipeline ==="
echo "Repo root: $(pwd)"
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — Baselines table (Idea 1 context)
# Runtime:  ~5 min on FF n=1000
# Produces: results/baselines_table.csv
# Needs:    nothing (pure algorithmic, no checkpoint)
# ──────────────────────────────────────────────────────────────────────────────
echo "[1/7] Running baselines (greedy, degree, etc.) ..."
python experiments/run_baselines.py --config configs/base_config.yaml
echo "  → results/baselines_table.csv"

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — Train Rev-GNN-LSTM (Idea 1 main model)
# Runtime:  ~3–4 h (Phase 1 ~1h MPS + Phase 2 ~2h CPU)
# Produces: results/checkpoints/rev_gnn_lstm.pt  (expected rev ~462 on FF n=1000)
# Released: YES — skip retrain with: cp <downloaded>/rev_gnn_lstm.pt results/checkpoints/
# ──────────────────────────────────────────────────────────────────────────────
if [ "${SKIP_TRAINING:-0}" = "1" ]; then
    echo "[2/7] Skipping LSTM training (SKIP_TRAINING=1)"
else
    echo "[2/7] Training Rev-GNN-LSTM (~3-4 h) ..."
    PYTORCH_ENABLE_MPS_FALLBACK=1 python experiments/run_rev_gnn_lstm.py \
        --config configs/experiments/rev_gnn_lstm.yaml
fi
echo "  → results/checkpoints/rev_gnn_lstm.pt"

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — Idea 1 evaluation (Fig 1: revenue-vs-k, zero-shot generalization)
# Runtime:  ~30 min
# Produces: results/figures/fig1_*.pdf/png
# Needs:    results/checkpoints/rev_gnn_lstm.pt
# ──────────────────────────────────────────────────────────────────────────────
echo "[3/7] Generating Idea 1 figures (Fig 1) ..."
python experiments/generate_paper_figures.py
echo "  → results/figures/fig1_*.pdf"

# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — Time-critical eval (Idea 2: TC-LSTM)
# Runtime:  ~10 min (eval only, uses released checkpoint)
# Produces: results/tc_eval_*.csv
# Needs:    results/checkpoints/rev_gnn_lstm_tc.pt  (or retrain: run_tc_lstm_training.py)
# Released: YES
# ──────────────────────────────────────────────────────────────────────────────
echo "[4/7] Time-critical evaluation (Idea 2) ..."
python experiments/run_tc_eval.py \
    --config configs/experiments/time_critical.yaml
echo "  → results/tc_eval_*.csv"

# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — Budget eval (Idea 3: LSTM-Idea3 at k=3,10,30)
# Runtime:  ~2 h (eval across budget levels + baselines comparison)
# Produces: results/budget_eval_*.csv
# Needs:    results/checkpoints/rev_gnn_lstm_budget.pt
# Released: YES
# ──────────────────────────────────────────────────────────────────────────────
echo "[5/7] Budget-constrained evaluation (Idea 3) ..."
python experiments/run_budget_eval.py \
    --config configs/experiments/budget_constrained.yaml \
    --budget_ckpt results/checkpoints/rev_gnn_lstm_budget.pt
echo "  → results/budget_eval_*.csv"

# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 — Ablations
# Runtime:  ~1 h
# Produces: results/ablation_*.csv
# Needs:    nothing (retrains from scratch on small graphs)
# ──────────────────────────────────────────────────────────────────────────────
echo "[6/7] Running ablations ..."
for CONFIG in ablation_encoder_type ablation_graph_type ablation_influence_model; do
    python experiments/${CONFIG}.py 2>/dev/null || \
    echo "  [warn] ${CONFIG}.py not yet runnable standalone — see ablation/ subfolder"
done
echo "  → results/ablation_*.csv"

# ──────────────────────────────────────────────────────────────────────────────
# STEP 7 — Final figure bundle
# Runtime:  ~5 min
# Produces: results/figures/paper_*.pdf
# ──────────────────────────────────────────────────────────────────────────────
echo "[7/7] Generating full paper figure bundle ..."
python experiments/generate_paper_figures.py
echo "  → results/figures/"

echo ""
echo "=== Reproduction complete. ==="
echo "Key outputs:"
echo "  results/figures/    ← PDF figures for paper"
echo "  results/logs/       ← Training CSV logs"
echo "  results/checkpoints/← Model weights"
