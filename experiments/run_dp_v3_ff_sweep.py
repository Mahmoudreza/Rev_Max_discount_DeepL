"""experiments/run_dp_v3_ff_sweep.py — Real DP v1/v2/v3 sweep on FF n=1000.

Matches seeds, graph, and budget config from dp_upgrade_eval (dp_upgrade.yaml).
v2/v3 use planner-only seeding (no Phase-1 free-seed warm-start).

Pre-committed decision rule:
    v3 PROMOTED iff revenue(v3, k=40) >= 420  AND  v3 > v2 by >= 10
    Otherwise paper keeps DP-Calibrated v1.

Usage:
    cd revmax-aaai2027 && source venv/bin/activate
    python experiments/run_dp_v3_ff_sweep.py         # or run in background (see below)

Background (nice):
    nice -n 10 python -u experiments/run_dp_v3_ff_sweep.py \
        > /tmp/dp_v3_ff_sweep.log 2>&1 &

Output:
    results/logs/dp_v3_ff_sweep.json
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.env.graph_generators import generate_forest_fire
from src.env.budget_revenue_env import BudgetEnvConfig
from src.evaluation.dp_calibrated import dp_calibrated_budget as run_v1
from src.evaluation.dp_calibrated_v2 import dp_calibrated_v2_budget as run_v2
from src.evaluation.dp_calibrated_v3 import dp_calibrated_v3_budget as run_v3

# ── Constants matching dp_upgrade.yaml / eval ──────────────────────────────────
SEED        = 42
N_GRAPH     = 1000
P_FF        = 0.37
PB_FF       = 0.32
C           = 0.3
WEIGHT_HIGH = 2.0
N_TRIALS    = 3
N_SIMS      = 30
DELTA       = 0.05
N_CLASSES   = 5
K_LIST      = [10, 15, 20, 30, 40]

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_DIR = ROOT / "results" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "dp_v3_ff_sweep.log")),
    ],
)
log = logging.getLogger(__name__)


def main() -> None:
    log.info("=" * 70)
    log.info("DP v1/v2/v3 REAL SWEEP — FF n=%d, k=%s", N_GRAPH, K_LIST)
    log.info("c=%.2f  weight_high=%.1f  seed=%d  n_trials=%d  n_sims=%d  delta=%.3f",
             C, WEIGHT_HIGH, SEED, N_TRIALS, N_SIMS, DELTA)
    log.info("v2/v3: planner-only seeding (no free-seed warm-start)")
    log.info("=" * 70)

    # Build Forest Fire graph (same graph as dp_upgrade_eval)
    graph = generate_forest_fire(N_GRAPH, P_FF, PB_FF, seed=SEED)
    log.info("Graph: n=%d  m=%d", graph.number_of_nodes(), graph.number_of_edges())

    cfg = BudgetEnvConfig(
        budget_B=1.0,
        production_cost=C,
        weight_high=WEIGHT_HIGH,
        seed=SEED,
    )

    common = dict(
        cfg=cfg,
        c=C,
        n_trials=N_TRIALS,
        n_sims=N_SIMS,
        delta=DELTA,
        n_classes=N_CLASSES,
    )

    results_all: dict = {}

    for k in K_LIST:
        B = round(k * C, 6)
        log.info("")
        log.info("── k=%d  B=%.3f ──────────────────────────────────────────", k, B)

        t0 = time.time()
        r_v1 = run_v1(graph, B=B, **common)
        t1 = time.time()

        r_v2 = run_v2(graph, B=B, **common)
        t2 = time.time()

        r_v3 = run_v3(graph, B=B, **common)
        t3 = time.time()

        m1 = r_v1["revenue"]["mean"]
        m2 = r_v2["revenue"]["mean"]
        m3 = r_v3["revenue"]["mean"]

        log.info("  v1=%.3f  (%.1fs)", m1, t1 - t0)
        log.info("  v2=%.3f  (%.1fs)", m2, t2 - t1)
        log.info("  v3=%.3f  (%.1fs)", m3, t3 - t2)

        # Gate checks
        gate_v3_ge_v1 = None
        gate_v3_gt_v2_10 = None

        if k in (10, 30):
            gate_v3_ge_v1 = bool(m3 >= m1 - 1e-6)
            status = "PASS ✓" if gate_v3_ge_v1 else "FAIL ✗"
            log.info("  Gate v3>=v1 @ k=%d: %s  (delta=%.3f)", k, status, m3 - m1)

        if k == 40:
            # Pre-committed rule: v3 >= 420 AND v3 > v2 + 10
            gate_v3_ge_420  = bool(m3 >= 420.0)
            gate_v3_gt_v2_10 = bool(m3 > m2 + 10.0)
            promoted = gate_v3_ge_420 and gate_v3_gt_v2_10
            log.info("  Gate v3>=420 @ k=40: %s  (v3=%.3f)",
                     "PASS ✓" if gate_v3_ge_420 else "FAIL ✗", m3)
            log.info("  Gate v3>v2+10 @ k=40: %s  (v3-v2=%.3f)",
                     "PASS ✓" if gate_v3_gt_v2_10 else "FAIL ✗", m3 - m2)
            log.info("  PROMOTION DECISION: %s",
                     "v3 PROMOTED → paper uses DP-Cal-v3" if promoted
                     else "v3 NOT PROMOTED → paper keeps DP-Cal-v1")

        results_all[f"k={k}"] = {
            "B": B,
            "v1": r_v1["revenue"],
            "v2": r_v2["revenue"],
            "v3": r_v3["revenue"],
            "gate_v3_ge_v1":    gate_v3_ge_v1,
            "gate_v3_gt_v2_10": gate_v3_gt_v2_10,
        }

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = LOG_DIR / "dp_v3_ff_sweep.json"
    with open(out_path, "w") as f:
        json.dump(results_all, f, indent=2)
    log.info("")
    log.info("Saved → %s", out_path)

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("FINAL SUMMARY (pre-committed gate, FF n=%d)", N_GRAPH)
    log.info("=" * 70)

    for k in K_LIST:
        key = f"k={k}"
        m1 = results_all[key]["v1"]["mean"]
        m2 = results_all[key]["v2"]["mean"]
        m3 = results_all[key]["v3"]["mean"]
        log.info("  k=%2d: v1=%.2f  v2=%.2f  v3=%.2f  (v3-v1=%+.2f  v3-v2=%+.2f)",
                 k, m1, m2, m3, m3 - m1, m3 - m2)

    # Overall promotion
    k40 = results_all.get("k=40", {})
    r3_40 = k40.get("v3", {}).get("mean", 0.0)
    r2_40 = k40.get("v2", {}).get("mean", 0.0)
    promoted = (r3_40 >= 420.0) and (r3_40 > r2_40 + 10.0)
    log.info("")
    log.info("VERDICT: %s",
             "v3 PROMOTED — paper uses DP-Calibrated-v3 as Idea-3 DP baseline"
             if promoted
             else "v3 NOT PROMOTED — paper keeps DP-Calibrated-v1 as Idea-3 DP baseline")
    log.info("=" * 70)

    sys.exit(0 if promoted else 1)


if __name__ == "__main__":
    main()
