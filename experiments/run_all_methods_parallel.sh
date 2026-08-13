#!/usr/bin/env bash
# run_all_methods_parallel.sh — evaluate all 9 methods on all 5 networks in parallel
# One worker per network, all run concurrently.
# Usage:  bash experiments/run_all_methods_parallel.sh [GPU_ID]
#   GPU_ID: which CUDA device to use (default 0; all workers share it — models are tiny)
#           If the server has ≥5 GPUs pass "multi" to assign GPU i to worker i.
set -e
GPU=${1:-0}
MULTI=0
[[ "$1" == "multi" ]] && MULTI=1

cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
LOGDIR=/tmp/all_methods_par

mkdir -p $LOGDIR results/logs results/checkpoints

NETWORKS="polblogs FF_1000 Rice_FB Modular_FF FF_2000"
PIDS=()
idx=0
for NET in $NETWORKS; do
    if [[ $MULTI -eq 1 ]]; then
        DEV=$idx
    else
        DEV=$GPU
    fi
    LOG=$LOGDIR/${NET}.log
    echo "[launcher] starting ${NET} → GPU=${DEV}  log=${LOG}"
    CUDA_VISIBLE_DEVICES=$DEV nohup $PY -u experiments/eval_all_methods_ksweep.py \
        --networks $NET > $LOG 2>&1 &
    PIDS+=($!)
    idx=$((idx+1))
done

echo "[launcher] workers: ${PIDS[*]}"
echo "[launcher] monitor: tail -f $LOGDIR/*.log"
echo "[launcher] waiting for all workers..."

# Wait and track exit codes
FAIL=0
for pid in "${PIDS[@]}"; do
    wait $pid
    CODE=$?
    if [[ $CODE -ne 0 ]]; then
        echo "[launcher] WORKER PID=$pid FAILED (exit $CODE)" >&2
        FAIL=1
    fi
done

if [[ $FAIL -eq 1 ]]; then
    echo "[launcher] FAILED — check logs in $LOGDIR" >&2
    exit 1
fi

echo "[launcher] all workers done — merging shards..."
$PY -u experiments/merge_all_methods_shards.py
