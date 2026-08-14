#!/usr/bin/env bash
# Runs per-network Cal-DP obs workers in parallel.
# polblogs is split into two shards (k=5,10,15 and k=20,30,40).
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=/tmp
OMP=4

echo "[$(date +%H:%M:%S)] Launching workers..."

OMP_NUM_THREADS=$OMP nohup python -u experiments/resweep_caldp_obs.py \
    --networks polblogs --k-values 5 10 15 \
    --out results/logs/caldp_obs_polblogs_lo.json \
    > $LOG/caldp_obs_polblogs_lo.log 2>&1 &
PIDS=($!)

OMP_NUM_THREADS=$OMP nohup python -u experiments/resweep_caldp_obs.py \
    --networks polblogs --k-values 20 30 40 \
    --out results/logs/caldp_obs_polblogs_hi.json \
    > $LOG/caldp_obs_polblogs_hi.log 2>&1 &
PIDS+=($!)

for NET in FF_1000 Rice_FB Modular_FF FF_2000; do
    OMP_NUM_THREADS=$OMP nohup python -u experiments/resweep_caldp_obs.py \
        --networks $NET \
        > $LOG/caldp_obs_${NET}.log 2>&1 &
    PIDS+=($!)
done

echo "PIDs: ${PIDS[*]}"
echo "Logs: $LOG/caldp_obs_*.log"

# Wait for all workers
for PID in "${PIDS[@]}"; do
    wait $PID && echo "  $PID: done" || echo "  $PID: FAILED"
done

echo "[$(date +%H:%M:%S)] All workers finished. Merging polblogs shards..."

# Merge polblogs lo+hi into one shard before running merge script
python - << 'PYEOF'
import json, os
lo = json.load(open("results/logs/caldp_obs_polblogs_lo.json"))
hi = json.load(open("results/logs/caldp_obs_polblogs_hi.json"))
lo["results"].update(hi["results"])
with open("results/logs/caldp_obs_polblogs.json", "w") as f:
    json.dump(lo, f, indent=2)
print("  polblogs merged -> results/logs/caldp_obs_polblogs.json")
PYEOF

echo "[$(date +%H:%M:%S)] Running merge script..."
python -u experiments/merge_caldp_obs.py
echo "Done."
