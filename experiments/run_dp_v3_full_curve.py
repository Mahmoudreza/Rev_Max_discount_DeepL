"""experiments/run_dp_v3_full_curve.py — Complete DP v1/v2/v3 curve.

Fills in the missing small-k points for FF n=1000 (k=[1,2,3,5,8]) and runs
the full sweep on Rice-FB (k=[1,2,3,5,8,10,15,20,30,40]).

Combines with existing dp_v3_ff_sweep.json (k=[10,15,20,30,40]) to give
the complete revenue-vs-budget curve needed for the paper figures.

Same seeds, c, weight_high as dp_upgrade.yaml / dp_upgrade_rice.yaml.

Usage:
    cd revmax-aaai2027 && source venv/bin/activate
    nice -n 10 python -u experiments/run_dp_v3_full_curve.py \
        > /tmp/dp_v3_full_curve.log 2>&1 &

Outputs:
    results/logs/dp_v3_ff_small_k.json      — FF n=1000 k=[1,2,3,5,8]
    results/logs/dp_v3_rice_fb_sweep.json   — Rice-FB full sweep
    results/logs/dp_v3_full_curve_merged.json — FF merged + Rice-FB
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import networkx as nx

from src.env.graph_generators import generate_forest_fire
from src.env.budget_revenue_env import BudgetEnvConfig
from src.evaluation.dp_calibrated import dp_calibrated_budget as run_v1
from src.evaluation.dp_calibrated_v2 import dp_calibrated_v2_budget as run_v2
from src.evaluation.dp_calibrated_v3 import dp_calibrated_v3_budget as run_v3

# ── Shared constants ───────────────────────────────────────────────────────────
SEED        = 42
C           = 0.3
WEIGHT_HIGH = 2.0
N_TRIALS    = 3
N_SIMS      = 30
DELTA       = 0.05
N_CLASSES   = 5

# Forest Fire (FF n=1000)
N_FF   = 1000
P_FF   = 0.37
PB_FF  = 0.32
K_FF_SMALL = [1, 2, 3, 5, 8]   # missing small-k points

# Rice-FB
RICE_PKL   = ROOT / "data" / "processed" / "rice_facebook.pkl"
K_RICE_ALL = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40]   # full sweep

LOG_DIR = ROOT / "results" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "dp_v3_full_curve.log")),
    ],
)
log = logging.getLogger(__name__)


def _sweep(graph, k_list, cfg, label: str) -> dict:
    """Run v1/v2/v3 sweep over k_list on graph. Returns {k=N: {...}} dict."""
    common = dict(
        cfg=cfg, c=C, n_trials=N_TRIALS, n_sims=N_SIMS,
        delta=DELTA, n_classes=N_CLASSES,
    )
    results = {}
    for k in k_list:
        B = round(k * C, 6)
        log.info("  [%s] k=%d  B=%.3f", label, k, B)

        t0 = time.time()
        r1 = run_v1(graph, B=B, **common)
        t1 = time.time()
        r2 = run_v2(graph, B=B, **common)
        t2 = time.time()
        r3 = run_v3(graph, B=B, **common)
        t3 = time.time()

        m1, m2, m3 = r1["revenue"]["mean"], r2["revenue"]["mean"], r3["revenue"]["mean"]
        log.info("    v1=%.3f(%.1fs)  v2=%.3f(%.1fs)  v3=%.3f(%.1fs)",
                 m1, t1-t0, m2, t2-t1, m3, t3-t2)

        results[f"k={k}"] = {
            "B": B, "k": k,
            "v1": r1["revenue"], "v2": r2["revenue"], "v3": r3["revenue"],
        }
    return results


def _load_rice() -> nx.Graph:
    """Load and return undirected Rice-FB graph."""
    with open(RICE_PKL, "rb") as f:
        g = pickle.load(f)
    if g.is_directed():
        g = g.to_undirected()
    return g


def main() -> None:
    log.info("=" * 70)
    log.info("DP v1/v2/v3 FULL CURVE — FF small-k + Rice-FB sweep")
    log.info("c=%.2f  weight_high=%.1f  seed=%d  n_trials=%d  n_sims=%d",
             C, WEIGHT_HIGH, SEED, N_TRIALS, N_SIMS)
    log.info("v2/v3: planner-only seeding (no free-seed warm-start)")
    log.info("=" * 70)

    cfg_ff = BudgetEnvConfig(
        budget_B=1.0, production_cost=C, weight_high=WEIGHT_HIGH, seed=SEED,
    )

    # ── Part 1: FF n=1000 small-k ─────────────────────────────────────────────
    log.info("")
    log.info("── Part 1: FF n=1000  k=%s", K_FF_SMALL)
    ff_graph = generate_forest_fire(N_FF, P_FF, PB_FF, seed=SEED)
    log.info("  Graph: n=%d  m=%d", ff_graph.number_of_nodes(), ff_graph.number_of_edges())

    ff_small = _sweep(ff_graph, K_FF_SMALL, cfg_ff, "FF")
    ff_out = LOG_DIR / "dp_v3_ff_small_k.json"
    with open(ff_out, "w") as f:
        json.dump(ff_small, f, indent=2)
    log.info("  FF small-k saved → %s", ff_out)

    # ── Part 2: Rice-FB full sweep ────────────────────────────────────────────
    log.info("")
    log.info("── Part 2: Rice-FB  k=%s", K_RICE_ALL)
    rice_graph = _load_rice()
    log.info("  Graph: n=%d  m=%d", rice_graph.number_of_nodes(), rice_graph.number_of_edges())

    cfg_rice = BudgetEnvConfig(
        budget_B=1.0, production_cost=C, weight_high=WEIGHT_HIGH, seed=SEED,
    )
    rice_all = _sweep(rice_graph, K_RICE_ALL, cfg_rice, "Rice-FB")
    rice_out = LOG_DIR / "dp_v3_rice_fb_sweep.json"
    with open(rice_out, "w") as f:
        json.dump(rice_all, f, indent=2)
    log.info("  Rice-FB saved → %s", rice_out)

    # ── Part 3: Merge FF (small-k + existing large-k) ─────────────────────────
    log.info("")
    log.info("── Part 3: Merging FF full curve")
    existing_ff_path = LOG_DIR / "dp_v3_ff_sweep.json"
    ff_large = {}
    if existing_ff_path.exists():
        with open(existing_ff_path) as f:
            ff_large = json.load(f)
        log.info("  Loaded existing FF large-k from %s", existing_ff_path)
    else:
        log.info("  [WARN] dp_v3_ff_sweep.json not found — merged FF will be small-k only")

    ff_merged = {}
    all_k_sorted = sorted(
        list(ff_small.keys()) + list(ff_large.keys()),
        key=lambda x: int(x.split("=")[1])
    )
    for kk in all_k_sorted:
        if kk in ff_small:
            ff_merged[kk] = ff_small[kk]
        elif kk in ff_large:
            ff_merged[kk] = ff_large[kk]

    merged_out = LOG_DIR / "dp_v3_full_curve_merged.json"
    with open(merged_out, "w") as f:
        json.dump({"ff_n1000": ff_merged, "rice_fb": rice_all}, f, indent=2)
    log.info("  Merged saved → %s", merged_out)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("SUMMARY — FF n=1000 (all k)")
    log.info("%-6s | %-9s | %-9s | %-9s", "k", "v1", "v2", "v3")
    log.info("-" * 42)
    for kk in sorted(ff_merged.keys(), key=lambda x: int(x.split("=")[1])):
        d = ff_merged[kk]
        v1m = d.get("v1", {}).get("mean", float("nan"))
        v2m = d.get("v2", {}).get("mean", float("nan"))
        v3m = d.get("v3", {}).get("mean", float("nan"))
        log.info("%-7s | %9.2f | %9.2f | %9.2f", kk, v1m, v2m, v3m)

    log.info("")
    log.info("SUMMARY — Rice-FB")
    log.info("%-6s | %-9s | %-9s | %-9s", "k", "v1", "v2", "v3")
    log.info("-" * 42)
    for kk in sorted(rice_all.keys(), key=lambda x: int(x.split("=")[1])):
        d = rice_all[kk]
        v1m = d.get("v1", {}).get("mean", float("nan"))
        v2m = d.get("v2", {}).get("mean", float("nan"))
        v3m = d.get("v3", {}).get("mean", float("nan"))
        log.info("%-7s | %9.2f | %9.2f | %9.2f", kk, v1m, v2m, v3m)

    log.info("=" * 70)
    log.info("Done. Full curve data at: %s", merged_out)


if __name__ == "__main__":
    main()
