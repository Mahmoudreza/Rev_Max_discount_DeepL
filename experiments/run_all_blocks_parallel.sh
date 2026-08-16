#!/usr/bin/env bash
# run_all_blocks_parallel.sh — launch ALL blocks (A–F) in parallel.
# All blocks write to different output files — no shared-write conflicts.
#
# Usage (from repo root):
#   bash experiments/run_all_blocks_parallel.sh [N_GPUS]
#   e.g.  bash experiments/run_all_blocks_parallel.sh 4
#
# Dependency: Block A merge runs AFTER all 5 Block-A shards finish.
#             Everything else is fully independent.
#
# Estimated wall-clock with 4 A100s: ~5-7h (bottleneck: Block D1 adaptation).
# GPU memory per process: ~2-4 GB (eval), ~6-8 GB (D1 training).

set -euo pipefail
export OMP_NUM_THREADS=4
NGPU=${1:-4}
G=0   # round-robin GPU counter

PIDS_A=()   # Block A eval workers (need merge after)
PIDS_ALL=() # everything else

nxt_gpu() { echo $((G % NGPU)); G=$((G+1)); }

echo "============================================================"
echo " Parallel block launcher  GPUs=0...$((NGPU-1))"
echo "============================================================"

# ── BLOCK A: 10-seed budget sweep (5 network shards) ─────────────────────────
echo "[A] Budget sweep 10-seed"
for NET in polblogs FF_1000 Rice_FB Modular_FF FF_2000; do
    GPU=$(nxt_gpu)
    LOG=/tmp/blockA_${NET}.log
    echo "    $NET → GPU $GPU  $LOG"
    CUDA_VISIBLE_DEVICES=$GPU nohup python -u \
        experiments/budget_sweep_10seed.py --networks "$NET" --gpu "$GPU" \
        > "$LOG" 2>&1 &
    PIDS_A+=($!)
done

# ── BLOCK B: ordering + Phase-1 ablation (unconstrained, 10 seeds) ───────────
echo "[B] Ablation unconstrained 10-seed"
GPU=$(nxt_gpu); LOG=/tmp/blockB.log
CUDA_VISIBLE_DEVICES=$GPU nohup python -u \
    experiments/ablation_unc_10seed.py --gpu "$GPU" \
    > "$LOG" 2>&1 &
PIDS_ALL+=($!)
echo "    → GPU $GPU  $LOG"

# ── BLOCK C: controls (5 network shards) ─────────────────────────────────────
echo "[C] Controls (C1 floor-greedy, C2 myopic-caldp, C3 pagerank)"
for NET in polblogs FF_1000 Rice_FB Modular_FF FF_2000; do
    GPU=$(nxt_gpu)
    LOG=/tmp/blockC_${NET}.log
    echo "    $NET → GPU $GPU  $LOG"
    CUDA_VISIBLE_DEVICES=$GPU nohup python -u \
        experiments/controls_10seed.py --networks "$NET" --gpu "$GPU" \
        > "$LOG" 2>&1 &
    PIDS_ALL+=($!)
done

# ── BLOCK D2: off-graph Cal-DP + B4 fill fractions (CPU-heavy, no GNN) ───────
echo "[D2] Off-graph Cal-DP + B4 fill fractions"
LOG=/tmp/blockD2.log
nohup python -u \
    experiments/caldp_offgraph.py \
    > "$LOG" 2>&1 &
PIDS_ALL+=($!)
echo "    → CPU  $LOG"

# ── BLOCK E: misspecification (FF-1000 + Rice-FB) ────────────────────────────
echo "[E] Misspecification"
GPU=$(nxt_gpu); LOG=/tmp/blockE.log
CUDA_VISIBLE_DEVICES=$GPU nohup python -u \
    experiments/misspec_eval.py --gpu "$GPU" \
    > "$LOG" 2>&1 &
PIDS_ALL+=($!)
echo "    → GPU $GPU  $LOG"

# ── BLOCK D1: adaptation fine-tuning (1 network per GPU, training) ───────────
echo "[D1] Policy adaptation (fine-tuning, ~5-8h)"
for NET in polblogs FF_1000 Rice_FB Modular_FF FF_2000; do
    GPU=$(nxt_gpu)
    LOG=/tmp/blockD1_${NET}.log
    echo "    $NET → GPU $GPU  $LOG"
    CUDA_VISIBLE_DEVICES=$GPU nohup python -u \
        experiments/adapt_policy.py --networks "$NET" --gpu "$GPU" \
        > "$LOG" 2>&1 &
    PIDS_ALL+=($!)
done

echo ""
echo "All workers launched."
echo "Monitor with:"
echo "  for f in /tmp/block*.log /tmp/blockD*.log; do echo \"==> \$f\"; tail -3 \$f 2>/dev/null; done"
echo ""

# ── Wait for Block A shards, then merge ──────────────────────────────────────
echo "Waiting for Block A shards..."
A_FAILED=0
for pid in "${PIDS_A[@]}"; do
    wait "$pid" && echo "  BlockA pid=$pid OK" || \
        { echo "  BlockA pid=$pid FAILED"; A_FAILED=$((A_FAILED+1)); }
done
if [ "$A_FAILED" -eq 0 ]; then
    echo "Running Block A merge + paired tests..."
    python -u experiments/merge_budget_10seed.py 2>&1 | tee /tmp/blockA_merge.log
else
    echo "WARNING: $A_FAILED Block A shards failed — skipping merge. Fix and re-run merge manually:"
    echo "  python -u experiments/merge_budget_10seed.py"
fi

# ── Wait for all remaining workers ───────────────────────────────────────────
echo ""
echo "Waiting for remaining workers (B, C, D, E)..."
FAILED=0
for pid in "${PIDS_ALL[@]}"; do
    wait "$pid" && echo "  pid=$pid OK" || \
        { echo "  pid=$pid FAILED"; FAILED=$((FAILED+1)); }
done

echo ""
echo "============================================================"
echo " DONE.  Failed workers: $FAILED"
echo " Quick result summary:"
echo "   Block A: tail -20 /tmp/blockA_merge.log"
echo "   Block B: tail -30 /tmp/blockB.log"
echo "   Block C: for f in /tmp/blockC_*.log; do tail -5 \$f; done"
echo "   Block D2: tail -20 /tmp/blockD2.log"
echo "   Block E: tail -30 /tmp/blockE.log"
echo "   Block D1: for f in /tmp/blockD1_*.log; do tail -5 \$f; done"
echo "============================================================"
