#!/usr/bin/env bash
# run_c1_ffba_parallel.sh — launch both C1 FFBA arms on GPU server
# Usage:  bash experiments/run_c1_ffba_parallel.sh [GPU_U1] [GPU_U2]
# Default: GPU 0 for 50_50, GPU 1 for 2to1
# Requires: ≥2 GPUs (or set same GPU for both, slower)
set -e
GPU_U1=${1:-0}
GPU_U2=${2:-1}

cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

echo "[launcher] GPU_U1=$GPU_U1  GPU_U2=$GPU_U2"
echo "[launcher] project: $(pwd)"

# Delete any existing P1/P2 epoch-1 checkpoints so assert won't fire on rerun
# (only if explicitly re-launching; comment out if resuming)
# rm -f results/checkpoints/c1_ffba_50_50_p1_ep1.pt results/checkpoints/c1_ffba_2to1_p1_ep1.pt

mkdir -p results/logs results/checkpoints

CUDA_VISIBLE_DEVICES=$GPU_U1 nohup $PY -u experiments/run_c1_ffba_training.py 50_50 \
    > /tmp/c1_ffba_50_50.log 2>&1 &
PID_U1=$!

CUDA_VISIBLE_DEVICES=$GPU_U2 nohup $PY -u experiments/run_c1_ffba_training.py 2to1 \
    > /tmp/c1_ffba_2to1.log 2>&1 &
PID_U2=$!

echo "[launcher] ARM-U1 (50_50) PID=$PID_U1  GPU=$GPU_U1"
echo "[launcher] ARM-U2 (2to1)  PID=$PID_U2  GPU=$GPU_U2"
echo "[launcher] Logs: /tmp/c1_ffba_50_50.log  /tmp/c1_ffba_2to1.log"
echo "[launcher] Monitor: tail -f /tmp/c1_ffba_50_50.log /tmp/c1_ffba_2to1.log"
