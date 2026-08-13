#!/usr/bin/env bash
# run_ie_ksweep_parallel.sh — Parallel IE+Budget k-sweep (CPU-only, no GPU).
#
# 6 workers (polblogs split into lo/hi to avoid wall-clock bottleneck):
#   polblogs k=5,10,15    → /tmp/ie_polblogs_lo.log
#   polblogs k=20,30,40   → /tmp/ie_polblogs_hi.log
#   FF_1000               → /tmp/ie_FF_1000.log
#   Rice_FB               → /tmp/ie_Rice_FB.log
#   Modular_FF            → /tmp/ie_Modular_FF.log
#   FF_2000               → /tmp/ie_FF_2000.log
#
# Usage (from repo root):
#   bash experiments/run_ie_ksweep_parallel.sh

set -euo pipefail
export OMP_NUM_THREADS=4   # IE is CPU-bound Monte-Carlo; avoid thread oversubscription

PIDS=()

echo "[IE-launch] polblogs k=5,10,15 → /tmp/ie_polblogs_lo.log"
nohup python -u experiments/eval_ie_budget_ksweep.py \
  --networks polblogs --k-values 5 10 15 \
  > /tmp/ie_polblogs_lo.log 2>&1 &
PIDS+=($!)

echo "[IE-launch] polblogs k=20,30,40 → /tmp/ie_polblogs_hi.log"
nohup python -u experiments/eval_ie_budget_ksweep.py \
  --networks polblogs --k-values 20 30 40 \
  > /tmp/ie_polblogs_hi.log 2>&1 &
PIDS+=($!)

echo "[IE-launch] FF_1000 → /tmp/ie_FF_1000.log"
nohup python -u experiments/eval_ie_budget_ksweep.py \
  --networks FF_1000 \
  > /tmp/ie_FF_1000.log 2>&1 &
PIDS+=($!)

echo "[IE-launch] Rice_FB → /tmp/ie_Rice_FB.log"
nohup python -u experiments/eval_ie_budget_ksweep.py \
  --networks Rice_FB \
  > /tmp/ie_Rice_FB.log 2>&1 &
PIDS+=($!)

echo "[IE-launch] Modular_FF → /tmp/ie_Modular_FF.log"
nohup python -u experiments/eval_ie_budget_ksweep.py \
  --networks Modular_FF \
  > /tmp/ie_Modular_FF.log 2>&1 &
PIDS+=($!)

echo "[IE-launch] FF_2000 → /tmp/ie_FF_2000.log"
nohup python -u experiments/eval_ie_budget_ksweep.py \
  --networks FF_2000 \
  > /tmp/ie_FF_2000.log 2>&1 &
PIDS+=($!)

echo ""
echo "[IE-launch] All 6 workers launched: ${PIDS[*]}"
echo "Waiting for all to finish..."

for pid in "${PIDS[@]}"; do
    wait "$pid" && echo "[IE-done] PID $pid OK" || echo "[IE-FAIL] PID $pid FAILED"
done

echo ""
echo "[IE-sweep] All workers finished. Run merge:"
echo "  python -u experiments/merge_ie_shards.py"
