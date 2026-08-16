#!/usr/bin/env bash
set -euo pipefail
PYTHON=$(command -v python 2>/dev/null || command -v python3)
pkill -f caldp_offgraph.py 2>/dev/null || true
pkill -f ablation_unc_10seed.py 2>/dev/null || true
sleep 2
nohup "$PYTHON" -u experiments/caldp_offgraph.py --networks FF_1000 Rice_FB Modular_FF FF_2000 > /tmp/blockD2.log 2>&1 &
echo "D2 pid=$!"
CUDA_VISIBLE_DEVICES=0 nohup "$PYTHON" -u experiments/ablation_unc_10seed.py --gpu 0 >> /tmp/blockB.log 2>&1 &
echo "blockB pid=$!"
echo "Monitor: tail -3 /tmp/blockB.log /tmp/blockD2.log"
