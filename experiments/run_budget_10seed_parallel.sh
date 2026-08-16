#!/usr/bin/env bash
# run_budget_10seed_parallel.sh — Block A: 10-seed budget sweep, parallelised by network.
# Usage: bash experiments/run_budget_10seed_parallel.sh [GPU_COUNT]
set -euo pipefail
export OMP_NUM_THREADS=4
GPUS=${1:-4}

PIDS=()
G=0

launch() {
    local NET=$1; local LOG=/tmp/budget10s_${NET}.log
    CUDA_VISIBLE_DEVICES=$((G % GPUS)) nohup python -u \
        experiments/budget_sweep_10seed.py --networks "$NET" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
    echo "[launch] $NET GPU=$((G % GPUS))  log=$LOG"
    G=$((G+1))
}

for NET in polblogs FF_1000 Rice_FB Modular_FF FF_2000; do
    launch "$NET"
done

echo "Pids: ${PIDS[*]}"
FAILED=0
for pid in "${PIDS[@]}"; do
    wait "$pid" && echo "  pid=$pid OK" || { echo "  pid=$pid FAILED"; FAILED=$((FAILED+1)); }
done
[ $FAILED -gt 0 ] && { echo "ABORT: $FAILED workers failed"; exit 1; }

echo "All done. Running merge..."
python -u experiments/merge_budget_10seed.py 2>&1
