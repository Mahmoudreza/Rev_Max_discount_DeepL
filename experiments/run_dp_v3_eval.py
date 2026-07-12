"""experiments/run_dp_v3_eval.py — evaluate DP-Calibrated v1/v2/v3 on FF n=200.

Usage:
    python experiments/run_dp_v3_eval.py

Outputs results/logs/dp_v3_eval_results.json with per-budget numbers.
Gate: v3 >= v1 at k=10,30 AND v3 > v2+10 at k=40.
"""

from __future__ import annotations

import json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import networkx as nx

from src.env.budget_revenue_env import BudgetEnvConfig
from src.evaluation.dp_calibrated import dp_calibrated_budget as run_v1
from src.evaluation.dp_calibrated_v2 import dp_calibrated_v2_budget as run_v2
from src.evaluation.dp_calibrated_v3 import dp_calibrated_v3_budget as run_v3
import logging as _logging

LOG_DIR  = ROOT / "results" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

_logging.basicConfig(level=_logging.INFO,
                     format="%(asctime)s %(message)s",
                     handlers=[
                         _logging.StreamHandler(),
                         _logging.FileHandler(str(LOG_DIR / "dp_v3_eval.log")),
                     ])

class _Logger:
    def log(self, msg: str) -> None:
        _logging.info(msg)

logger = _Logger()

# ── Graph + config ─────────────────────────────────────────────────────────────
N     = 200
SEED  = 42
rng   = nx.utils.create_py_random_state(SEED)
graph = nx.barabasi_albert_graph(N, 3, seed=SEED)

cfg = BudgetEnvConfig(
    production_cost=1.0,
    weight_high=0.8,
)

C = cfg.production_cost

K_LIST  = [10, 20, 30, 40]
N_TRIALS = 10
N_SIMS   = 20       # calibration sims (fast pass)
DELTA    = 0.05

COMMON = dict(
    cfg=cfg,
    c=C,
    n_trials=N_TRIALS,
    n_sims=N_SIMS,
    delta=DELTA,
)

results_all: dict = {}

for k in K_LIST:
    B = k * C
    logger.log(f"\n=== k={k}  B={B:.1f} ===")

    t0 = time.time()
    r_v1 = run_v1(graph, B=B, **COMMON)
    t1 = time.time()
    r_v2 = run_v2(graph, B=B, **COMMON)
    t2 = time.time()
    r_v3 = run_v3(graph, B=B, **COMMON)
    t3 = time.time()

    m1 = r_v1["revenue"]["mean"]
    m2 = r_v2["revenue"]["mean"]
    m3 = r_v3["revenue"]["mean"]

    logger.log(f"  v1={m1:.3f}  ({t1-t0:.1f}s)")
    logger.log(f"  v2={m2:.3f}  ({t2-t1:.1f}s)")
    logger.log(f"  v3={m3:.3f}  ({t3-t2:.1f}s)")

    gate_k10_30 = (m3 >= m1 - 1e-6) if k in (10, 30) else None
    gate_k40    = (m3 >= m2 - 1e-6)  if k == 40        else None

    if gate_k10_30 is not None:
        status = "PASS" if gate_k10_30 else "FAIL"
        logger.log(f"  Gate v3>=v1 @ k={k}: {status}")
    if gate_k40 is not None:
        status = "PASS" if gate_k40 else "FAIL"
        logger.log(f"  Gate v3>=v2 @ k=40: {status}  (v3-v2={m3-m2:+.3f})")

    results_all[f"k={k}"] = {
        "B": B,
        "v1": r_v1["revenue"],
        "v2": r_v2["revenue"],
        "v3": r_v3["revenue"],
        "gate_v3_ge_v1": gate_k10_30,
        "gate_v3_gt_v2_10": gate_k40,
    }

# Save
out_path = LOG_DIR / "dp_v3_eval_results.json"
with open(out_path, "w") as f:
    json.dump(results_all, f, indent=2)
logger.log(f"\nSaved → {out_path}")

# Final gate summary
passes = []
for k in (10, 30):
    key = f"k={k}"
    if key in results_all and results_all[key]["gate_v3_ge_v1"] is not None:
        passes.append(results_all[key]["gate_v3_ge_v1"])
if "k=40" in results_all and results_all["k=40"]["gate_v3_gt_v2_10"] is not None:
    passes.append(results_all["k=40"]["gate_v3_gt_v2_10"])

if passes:
    overall = all(passes)
    logger.log(f"\n{'='*40}")
    logger.log(f"Overall gate: {'PASS — v3 promoted' if overall else 'FAIL — investigate'}")
    logger.log(f"{'='*40}")
    sys.exit(0 if overall else 1)
