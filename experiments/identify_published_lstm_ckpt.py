"""experiments/identify_published_lstm_ckpt.py — Identify which LSTM budget
checkpoint matches the published paper results.

Evaluates rev_gnn_lstm_budget_v1.pt (Jul-4) and rev_gnn_lstm_budget_v2.pt (Jul-6)
at the two fingerprint points:
  k=3  on FF n=1000
  k=10 on Rice-FB n=443
n_trials=3, seed=42 (matches published dp_upgrade_eval seeds).

Published target numbers (from paper):
  FF   k=3  → 332 ± noise
  Rice k=10 → 68.6 ± noise

Decision rule (absolute difference ≤ MATCH_TOL from target mean):
  If v1.pt matches both → v1.pt IS the published checkpoint
  If v2.pt matches both → v2.pt IS the published checkpoint
  If neither → published checkpoint confirmed lost

Context: On 2026-07-12, the Welford-bug retrain (PID 61149) overwrote
rev_gnn_lstm_budget.pt before a pre-retrain backup was made.
rev_gnn_lstm_budget_v1_welford_bug.pt has the same hash as the partially
retrained checkpoint (ep 10) — it is NOT the original buggy checkpoint.

Usage:
    cd revmax-aaai2027 && source venv/bin/activate
    python experiments/identify_published_lstm_ckpt.py

    # Optionally, also test the retrained checkpoint once PID 61149 completes:
    python experiments/identify_published_lstm_ckpt.py --include_retrained

Output:
    Printed verdict to stdout.
    results/logs/lstm_ckpt_fingerprint.json
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.utils.helpers import load_config_with_base, set_seed, get_device
from src.env.graph_generators import generate_forest_fire
from src.evaluation.budget_baselines import evaluate_budget_aware_policy

# ── Constants (must match dp_upgrade_eval / dp_upgrade_eval_rice seeds) ───────
SEED        = 42
N_FF        = 1000
P_FF        = 0.37
PB_FF       = 0.32
C           = 0.3
WEIGHT_HIGH = 2.0
N_TRIALS    = 3

RICE_PKL = ROOT / "data" / "processed" / "rice_facebook.pkl"

# Fingerprint points
FINGERPRINT_FF_K   = 3
FINGERPRINT_RICE_K = 10

# Published targets (mean revenue)
TARGET_FF   = 332.0
TARGET_RICE =  68.6
MATCH_TOL   =  15.0   # ± tolerance; noise over 3 trials is expected

CFG_DP = ROOT / "configs" / "experiments" / "dp_upgrade.yaml"

CKPTS = {
    "v1": ROOT / "results" / "checkpoints" / "rev_gnn_lstm_budget_v1.pt",
    "v2": ROOT / "results" / "checkpoints" / "rev_gnn_lstm_budget_v2.pt",
}

LOG_DIR = ROOT / "results" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _load_lstm(ckpt: Path, device: torch.device, label: str):
    """Load budget-aware LSTM policy from checkpoint.

    Args:
        ckpt:   Checkpoint path.
        device: Target device.
        label:  Name for logging.

    Returns:
        SequentialJointPolicy in eval mode.
    """
    from src.models.encoders.graphsage import GraphSAGEEncoder
    from src.models.encoders.sequence_models import EpisodeLSTM
    from src.models.policies.sequential_joint_policy import SequentialJointPolicy

    cfg = load_config_with_base(str(CFG_DP))
    sd = torch.load(str(ckpt), map_location=device, weights_only=True)
    ih_key = next((k for k in sd if "weight_ih_l0" in k), None)
    lstm_hidden = sd[ih_key].shape[0] // 4 if ih_key else 128
    lstm_n_layers = int(cfg.sequence_model.lstm_n_layers)
    gnn_hidden = int(cfg.encoder.hidden_dim)

    enc  = GraphSAGEEncoder(21, gnn_hidden, int(cfg.encoder.n_layers), 0.0)
    lstm = EpisodeLSTM(gnn_hidden, lstm_hidden, lstm_n_layers)
    pol  = SequentialJointPolicy(enc, lstm, gnn_dim=gnn_hidden, context_dim=lstm_hidden)
    pol.load_state_dict(sd, strict=False)
    pol.to(device).eval()

    import hashlib
    sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()[:12]
    mtime = datetime.fromtimestamp(ckpt.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    log.info("  %s → %s | sha256=%s | mtime=%s | lstm_hidden=%d",
             label, ckpt.name, sha, mtime, lstm_hidden)
    return pol


def _load_rice():
    """Load undirected Rice-FB graph."""
    import networkx as nx
    with open(RICE_PKL, "rb") as f:
        g = pickle.load(f)
    return g.to_undirected() if g.is_directed() else g


def _eval(pol, graph, k: int, device) -> dict:
    """Run evaluate_budget_aware_policy at given k.

    Args:
        pol:    Policy to evaluate.
        graph:  NetworkX graph.
        k:      Number of offers.
        device: torch.device.

    Returns:
        Result dict from evaluate_budget_aware_policy.
    """
    B = round(k * C, 6)
    return evaluate_budget_aware_policy(
        pol, graph, B=B, c=C, device=device,
        n_trials=N_TRIALS, weight_high=WEIGHT_HIGH,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include_retrained", action="store_true",
                        help="Also test rev_gnn_lstm_budget.pt (PID-61149 retrain, "
                             "only useful after retrain completes at ~17:00 2026-07-12).")
    args = parser.parse_args()

    if args.include_retrained:
        CKPTS["retrained"] = ROOT / "results" / "checkpoints" / "rev_gnn_lstm_budget.pt"

    set_seed(SEED)
    device = get_device()

    log.info("=" * 68)
    log.info("LSTM Checkpoint Fingerprint Identification")
    log.info("Fingerprint: FF k=%d (target=%.1f) + Rice k=%d (target=%.1f)",
             FINGERPRINT_FF_K, TARGET_FF, FINGERPRINT_RICE_K, TARGET_RICE)
    log.info("Tolerance: ±%.1f", MATCH_TOL)
    log.info("=" * 68)

    ff_graph   = generate_forest_fire(N_FF, P_FF, PB_FF, seed=SEED)
    rice_graph = _load_rice()
    log.info("FF:   n=%d m=%d", ff_graph.number_of_nodes(), ff_graph.number_of_edges())
    log.info("Rice: n=%d m=%d", rice_graph.number_of_nodes(), rice_graph.number_of_edges())

    results: dict = {}
    for label, ckpt in CKPTS.items():
        if not ckpt.exists():
            log.warning("  MISSING: %s — skipping.", ckpt)
            continue

        log.info("")
        log.info("── %s ──────────────────────────────────────────────────", label)
        pol = _load_lstm(ckpt, device, label)

        r_ff   = _eval(pol, ff_graph,   FINGERPRINT_FF_K,   device)
        r_rice = _eval(pol, rice_graph, FINGERPRINT_RICE_K, device)

        m_ff   = r_ff["revenue"]["mean"]
        m_rice = r_rice["revenue"]["mean"]
        s_ff   = r_ff["revenue"]["std"]
        s_rice = r_rice["revenue"]["std"]

        match_ff   = abs(m_ff   - TARGET_FF)   <= MATCH_TOL
        match_rice = abs(m_rice - TARGET_RICE) <= MATCH_TOL
        both_match = match_ff and match_rice

        log.info("  FF   k=%d: rev=%.2f±%.2f  target=%.1f  diff=%.1f  %s",
                 FINGERPRINT_FF_K, m_ff, s_ff, TARGET_FF,
                 abs(m_ff - TARGET_FF), "✓ MATCH" if match_ff else "✗ no match")
        log.info("  Rice k=%d: rev=%.2f±%.2f  target=%.1f  diff=%.1f  %s",
                 FINGERPRINT_RICE_K, m_rice, s_rice, TARGET_RICE,
                 abs(m_rice - TARGET_RICE), "✓ MATCH" if match_rice else "✗ no match")
        log.info("  → %s IS the published checkpoint" % label if both_match
                 else "  → %s does NOT match published numbers" % label)

        results[label] = {
            "ff_k3":    {"mean": m_ff,   "std": s_ff,   "target": TARGET_FF,
                         "diff": abs(m_ff - TARGET_FF), "match": match_ff},
            "rice_k10": {"mean": m_rice, "std": s_rice, "target": TARGET_RICE,
                         "diff": abs(m_rice - TARGET_RICE), "match": match_rice},
            "both_match": both_match,
        }

    # ── Verdict ───────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 68)
    log.info("VERDICT")
    log.info("=" * 68)
    matches = [lbl for lbl, r in results.items() if r.get("both_match")]
    if matches:
        for m in matches:
            log.info("  ✓ %s.pt IS the published checkpoint →"
                     " relabel, update checkpoints/README.md, ship in GitHub release",
                     m)
    else:
        log.info("  ✗ Neither checkpoint matches published numbers (±%.1f tolerance).", MATCH_TOL)
        log.info("    Published checkpoint is CONFIRMED LOST.")
        log.info("    Update CLAUDE.md Session State accordingly.")
    log.info("=" * 68)

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "fingerprint": {
            "ff_k": FINGERPRINT_FF_K, "rice_k": FINGERPRINT_RICE_K,
            "target_ff": TARGET_FF, "target_rice": TARGET_RICE,
            "match_tol": MATCH_TOL,
        },
        "results": results,
        "verdict": matches if matches else "CONFIRMED_LOST",
        "timestamp": datetime.now().isoformat(),
    }
    out_path = LOG_DIR / "lstm_ckpt_fingerprint.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info("Saved → %s", out_path)


if __name__ == "__main__":
    main()
