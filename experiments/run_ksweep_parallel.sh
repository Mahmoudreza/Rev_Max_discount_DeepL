#!/usr/bin/env bash
# run_ksweep_parallel.sh — Launch parallel ksweep workers, then merge + verify.
#
# Splits work into 6 processes (4 GPUs assumed, cycled):
#   polblogs (large) → split: k=5,10,15 on GPU 0 | k=20,30,40 on GPU 1
#   FF_1000          → GPU 2
#   Rice_FB          → GPU 3
#   Modular_FF       → GPU 0
#   FF_2000          → GPU 1
#
# Each worker writes results/logs/budget_sweep_{NET}_{k-range}.json
# OMP_NUM_THREADS=4 per worker to avoid CPU thread oversubscription.
#
# Usage (from repo root, after git pull):
#   bash experiments/run_ksweep_parallel.sh
#
# After all pids finish:
#   python -u experiments/merge_ksweep_shards.py 2>&1
#   # Then supply --reference <FF_1000_k40_OURS_sequential_val> for verify

set -euo pipefail
export OMP_NUM_THREADS=4

PIDS=()

echo "[launch] polblogs k=5,10,15 → GPU 0, /tmp/ksweep_polblogs_lo.log"
CUDA_VISIBLE_DEVICES=0 nohup python -u \
  experiments/eval_all_methods_ksweep.py \
  --networks polblogs --k-values 5 10 15 \
  > /tmp/ksweep_polblogs_lo.log 2>&1 &
PIDS+=($!)

echo "[launch] polblogs k=20,30,40 → GPU 1, /tmp/ksweep_polblogs_hi.log"
CUDA_VISIBLE_DEVICES=1 nohup python -u \
  experiments/eval_all_methods_ksweep.py \
  --networks polblogs --k-values 20 30 40 \
  > /tmp/ksweep_polblogs_hi.log 2>&1 &
PIDS+=($!)

echo "[launch] FF_1000 → GPU 2, /tmp/ksweep_FF_1000.log"
CUDA_VISIBLE_DEVICES=2 nohup python -u \
  experiments/eval_all_methods_ksweep.py \
  --networks FF_1000 \
  > /tmp/ksweep_FF_1000.log 2>&1 &
PIDS+=($!)

echo "[launch] Rice_FB → GPU 3, /tmp/ksweep_Rice_FB.log"
CUDA_VISIBLE_DEVICES=3 nohup python -u \
  experiments/eval_all_methods_ksweep.py \
  --networks Rice_FB \
  > /tmp/ksweep_Rice_FB.log 2>&1 &
PIDS+=($!)

echo "[launch] Modular_FF → GPU 0, /tmp/ksweep_Modular_FF.log"
CUDA_VISIBLE_DEVICES=0 nohup python -u \
  experiments/eval_all_methods_ksweep.py \
  --networks Modular_FF \
  > /tmp/ksweep_Modular_FF.log 2>&1 &
PIDS+=($!)

echo "[launch] FF_2000 → GPU 1, /tmp/ksweep_FF_2000.log"
CUDA_VISIBLE_DEVICES=1 nohup python -u \
  experiments/eval_all_methods_ksweep.py \
  --networks FF_2000 \
  > /tmp/ksweep_FF_2000.log 2>&1 &
PIDS+=($!)

echo ""
echo "[pids] ${PIDS[*]}"
echo "Monitor: tail -5 /tmp/ksweep_*.log"
echo ""
echo "Waiting for all 6 workers..."
FAILED=0
for pid in "${PIDS[@]}"; do
    if wait "$pid"; then
        echo "  pid=$pid OK"
    else
        echo "  pid=$pid FAILED (exit $?)"
        FAILED=$((FAILED + 1))
    fi
done

if [ "$FAILED" -gt 0 ]; then
    echo "ABORT: $FAILED worker(s) failed. Fix and re-run missing shards, then merge."
    exit 1
fi

echo ""
echo "All workers complete. Running merge..."
python -u experiments/merge_ksweep_shards.py 2>&1

echo ""
echo "NEXT: run verify-parallel step with sequential reference:"
echo "  python -u experiments/merge_ksweep_shards.py --reference <FF_1000_k40_OURS>"
