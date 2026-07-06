#!/bin/bash
cd /Users/reza/Desktop/revmax-aaai2027
nohup venv/bin/python experiments/run_rev_gnn_im_rl.py --config configs/experiments/rev_gnn_im_rl.yaml > /tmp/revmax_full_train.log 2>&1 &
echo "Training launched with PID: $!"
echo "Check progress: tail -f /tmp/revmax_full_train.log"
