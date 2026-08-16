#!/usr/bin/env bash
# restart_bcd.sh — restart only Blocks B, C, D2 (A and D1 already running)
set -euo pipefail
PYTHON=$(command -v python 2>/dev/null || command -v python3)
echo "Python: $PYTHON"
NGPU=${1:-4}
_GPU_COUNTER_FILE=$(mktemp)
echo 0 > "$_GPU_COUNTER_FILE"
nxt_gpu() {
    local c; c=$(cat "$_GPU_COUNTER_FILE")
    echo $((c % NGPU))
    echo $((c + 1)) > "$_GPU_COUNTER_FILE"
}

echo "[B] Ablation unconstrained"
GPU=$(nxt_gpu)
CUDA_VISIBLE_DEVICES=$GPU nohup "$PYTHON" -u experiments/ablation_unc_10seed.py --gpu 0 > /tmp/blockB.log 2>&1 &
echo "  pid=$! GPU=$GPU"

echo "[C] Controls"
for NET in polblogs FF_1000 Rice_FB Modular_FF FF_2000; do
    GPU=$(nxt_gpu)
    CUDA_VISIBLE_DEVICES=$GPU nohup "$PYTHON" -u experiments/controls_10seed.py --networks "$NET" --gpu 0 > /tmp/blockC_${NET}.log 2>&1 &
    echo "  $NET pid=$! GPU=$GPU"
done

echo "[D2] Off-graph Cal-DP (CPU)"
nohup "$PYTHON" -u experiments/caldp_offgraph.py > /tmp/blockD2.log 2>&1 &
echo "  pid=$! CPU"

echo "Monitor: for f in /tmp/blockB.log /tmp/blockC_*.log /tmp/blockD2.log; do echo \"==> \$f\"; tail -2 \$f 2>/dev/null; done"
