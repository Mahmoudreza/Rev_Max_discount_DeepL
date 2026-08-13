#!/usr/bin/env bash
# run_c1_ffba_eval_parallel.sh — parallel C1 FFBA evaluation (5 workers, one per network)
# Usage: bash experiments/run_c1_ffba_eval_parallel.sh [GPU_ID]
#        GPU_ID: CUDA device (default 0). Pass 'multi' to assign GPU i to worker i.
set -e
GPU=${1:-0}
MULTI=0
[[ "$1" == "multi" ]] && MULTI=1

cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
LOGDIR=/tmp/c1_ffba_eval_par

mkdir -p $LOGDIR results/logs results/checkpoints

NETWORKS="FF_1000 FF_2000 Modular_FF Rice_FB polblogs"
PIDS=()
idx=0
for NET in $NETWORKS; do
    [[ $MULTI -eq 1 ]] && DEV=$idx || DEV=$GPU
    LOG=$LOGDIR/${NET}.log
    echo "[launcher] ${NET} → GPU=${DEV}  log=${LOG}"
    CUDA_VISIBLE_DEVICES=$DEV nohup $PY -u experiments/eval_c1_ffba.py \
        --networks $NET > $LOG 2>&1 &
    PIDS+=($!)
    idx=$((idx+1))
done

echo "[launcher] PIDs: ${PIDS[*]}"
echo "[launcher] monitor: tail -f $LOGDIR/*.log"
echo "[launcher] waiting..."

FAIL=0
for pid in "${PIDS[@]}"; do
    wait $pid; CODE=$?
    [[ $CODE -ne 0 ]] && { echo "[launcher] WORKER PID=$pid FAILED (exit $CODE)" >&2; FAIL=1; }
done
[[ $FAIL -eq 1 ]] && { echo "[launcher] FAILED — check $LOGDIR" >&2; exit 1; }

echo "[launcher] all done — merging shards..."
$PY -u experiments/merge_c1_ffba_shards.py
