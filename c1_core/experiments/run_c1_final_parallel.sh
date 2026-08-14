#!/usr/bin/env bash
# run_c1_final_parallel.sh
# ========================
# Launch 5 eval workers in parallel (one per network) on GPU server.
# Each writes results/logs/c1_final_shard{N}.json.
# After all finish, merge into results/logs/c1_final.json.
#
# Usage (from repo root):
#   bash c1_core/experiments/run_c1_final_parallel.sh [--device cuda] [--config PATH]
#
# GPU assignment: if CUDA_VISIBLE_DEVICES has multiple GPUs, workers
# are round-robin assigned. Defaults to CPU if no GPU present.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

CONFIG="${CONFIG:-configs/experiments/rev_gnn_lstm.yaml}"
DEVICE="${DEVICE:-cpu}"
LOG_DIR="results/logs"
SCRIPT="c1_core/experiments/eval_c1_final.py"

# Parse CLI overrides
for arg in "$@"; do
  case $arg in
    --device=*) DEVICE="${arg#*=}" ;;
    --config=*) CONFIG="${arg#*=}" ;;
    --device)   shift; DEVICE="$1" ;;
    --config)   shift; CONFIG="$1" ;;
  esac
done

mkdir -p "$LOG_DIR"
echo "=== C1 Final Eval — parallel (5 shards) ==="
echo "    config=$CONFIG  device=$DEVICE"
echo "    log → $LOG_DIR/c1_final_shard{0..4}.json"
echo ""

# Detect GPU count
N_GPU=0
if command -v nvidia-smi &>/dev/null; then
  N_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
fi
echo "    GPUs detected: $N_GPU"
echo ""

PIDS=()
for SHARD in 0 1 2 3 4; do
  # Assign GPU round-robin if available
  if [ "$N_GPU" -gt 0 ]; then
    GPU_ID=$(( SHARD % N_GPU ))
    DEV="cuda:$GPU_ID"
  else
    DEV="$DEVICE"
  fi

  LOG_FILE="$LOG_DIR/c1_final_shard${SHARD}.log"
  echo "  Shard $SHARD → device=$DEV  log=$LOG_FILE"

  CUDA_VISIBLE_DEVICES=${GPU_ID:-""} \
  python -u "$SCRIPT" \
    --shard "$SHARD" \
    --config "$CONFIG" \
    --device "$DEV" \
    > "$LOG_FILE" 2>&1 &

  PIDS+=($!)
done

echo ""
echo "  PIDs: ${PIDS[*]}"
echo "  Waiting for all shards to finish..."

# Wait and report exit codes
ALL_OK=1
for i in "${!PIDS[@]}"; do
  PID="${PIDS[$i]}"
  if wait "$PID"; then
    echo "  Shard $i DONE  (PID $PID)"
  else
    echo "  Shard $i FAILED (PID $PID) — check $LOG_DIR/c1_final_shard${i}.log"
    ALL_OK=0
  fi
done

if [ "$ALL_OK" -eq 1 ]; then
  echo ""
  echo "=== All shards complete. Merging... ==="
  python -u "$SCRIPT" --merge --config "$CONFIG" 2>&1 | tee "$LOG_DIR/c1_final_merge.log"
  echo ""
  echo "✓  results/logs/c1_final.json written."
  echo "   Commit with:"
  echo "   git add results/logs/c1_final.json && git commit -m 'results: C1 final eval with sha attribution'"
else
  echo ""
  echo "✗  One or more shards failed. Fix errors before merging."
  echo "   Re-run individual shards:"
  echo "   python $SCRIPT --shard N --config $CONFIG"
  exit 1
fi
