"""experiments/run_gate_b_transformer_eval.py — Gate B v2: Transformer-Idea3 vs LSTM-Idea3.

Evaluates budget-aware policies on TWO networks:
  • FF n=1000  (Forest Fire, same seed / graph as dp_upgrade_eval)
  • Rice-FB n=443

k sweep: [1, 2, 3, 5, 8, 10, 15, 20, 30, 40]  (10 values, matches dp_upgrade_eval)
n_trials=3, seed=42, SKIP enforcement, accounting identity checks.

LSTM versions:
  --lstm_v   v1        → results/checkpoints/rev_gnn_lstm_budget_v1.pt  (Jul-4, clean original)
  --lstm_v   retrained → results/checkpoints/rev_gnn_lstm_budget.pt     (Welford-fixed retrain)
  --lstm_v   both      → evaluate both; default after retrain completes

Gate B criterion (pre-committed):
  PASS iff TFM-Idea3 > LSTM-Idea3 at ≥4 of 10 k-values on EITHER network.

⚠ NOTE on LSTM retrain (PID 61149):
  The Welford-fixed retrain started 2026-07-12 14:49 (200 epochs, ~7 min/10ep).
  Run this script with --lstm_v v1 first (immediate) and re-run with --lstm_v both
  once PID 61149 completes (expected ~2026-07-12 17:00).

Usage:
    cd revmax-aaai2027 && source venv/bin/activate

    # Immediate run (LSTM-v1 only):
    python -u experiments/run_gate_b_transformer_eval.py --lstm_v v1 \\
        > /tmp/gate_b_v2_v1.log 2>&1 &

    # After retrain completes (both LSTM versions):
    python -u experiments/run_gate_b_transformer_eval.py --lstm_v both \\
        > /tmp/gate_b_v2_both.log 2>&1 &

Outputs:
    results/logs/gate_b_eval_v2.json        (overwritten on re-run)
    results/logs/gate_b_eval_v2_<ts>.json   (timestamped archive)

NOTE: gate_b_eval.json from 2026-07-12 session is VOID — weak criterion
      (single k=40, FF only). This v2 script supersedes it.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.utils.helpers import load_config_with_base, set_seed, ensure_dir, get_device
from src.env.graph_generators import generate_forest_fire
from src.evaluation.budget_baselines import evaluate_budget_aware_policy

# ── Constants ──────────────────────────────────────────────────────────────────
SEED         = 42
N_GRAPH      = 1000
P_FF         = 0.37
PB_FF        = 0.32
C            = 0.3
WEIGHT_HIGH  = 2.0
N_TRIALS     = 3
K_LIST       = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40]

RICE_PKL     = ROOT / "data" / "processed" / "rice_facebook.pkl"
LSTM_V1_CKPT = ROOT / "results" / "checkpoints" / "rev_gnn_lstm_budget_v1.pt"
LSTM_RT_CKPT = ROOT / "results" / "checkpoints" / "rev_gnn_lstm_budget.pt"
TFM_CKPT     = ROOT / "results" / "checkpoints" / "rev_gnn_transformer_budget.pt"
CFG_BUDGET   = ROOT / "configs" / "experiments" / "budget_constrained.yaml"
CFG_TFM      = ROOT / "configs" / "experiments" / "rev_gnn_transformer_300ep.yaml"
CFG_DP       = ROOT / "configs" / "experiments" / "dp_upgrade.yaml"

# Gate B criterion
GATE_B_MIN_WINS = 4   # TFM must beat LSTM at >= this many k-values
GATE_B_N_K      = len(K_LIST)  # 10

BUDGET_ACCT_EPS = 1e-4   # accounting identity tolerance

LOG_DIR = ROOT / "results" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
ensure_dir(str(ROOT / "results" / "checkpoints"))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "gate_b_eval_v2.log")),
    ],
)
log = logging.getLogger(__name__)


# ── Model loaders ──────────────────────────────────────────────────────────────

def _load_lstm(ckpt: Path, device: torch.device, label: str) -> object:
    """Load LSTM-Idea3 (21-dim budget-aware SequentialJointPolicy).

    Args:
        ckpt:   Path to LSTM budget checkpoint.
        device: torch.device.
        label:  Human label for logging.

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

    mtime = datetime.fromtimestamp(ckpt.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    log.info("  %s → %s (mtime %s)  lstm_hidden=%d  params=%s",
             label, ckpt.name, mtime, lstm_hidden,
             f"{sum(p.numel() for p in pol.parameters()):,}")
    return pol


def _load_transformer(ckpt: Path, device: torch.device) -> object:
    """Load Transformer-Idea3 (21-dim budget-aware TransformerJointPolicy).

    Args:
        ckpt:   Path to TFM budget checkpoint.
        device: torch.device.

    Returns:
        TransformerJointPolicy in eval mode.
    """
    from omegaconf import OmegaConf
    from src.models.encoders.graphsage import GraphSAGEEncoder
    from src.models.encoders.episode_transformer import EpisodeTransformerSliding
    from src.models.policies.transformer_joint_policy import TransformerJointPolicy

    cfg_budget = load_config_with_base(str(CFG_BUDGET))
    cfg_tfm    = load_config_with_base(str(CFG_TFM))
    cfg = OmegaConf.merge(
        cfg_budget,
        OmegaConf.create({"transformer": OmegaConf.to_container(cfg_tfm.transformer)}),
    )

    enc = GraphSAGEEncoder(
        21, int(cfg.encoder.hidden_dim),
        int(cfg.encoder.n_layers), float(cfg.encoder.dropout),
    )
    tfm = EpisodeTransformerSliding.from_config(cfg.transformer)
    pol = TransformerJointPolicy(
        enc, tfm,
        gnn_dim=int(cfg.encoder.hidden_dim),
        context_dim=tfm.context_dim,
    )
    sd = torch.load(str(ckpt), map_location=device, weights_only=True)
    pol.load_state_dict(sd, strict=False)
    pol.to(device).eval()

    mtime = datetime.fromtimestamp(ckpt.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    log.info("  Transformer-Idea3 → %s (mtime %s)  window=%d  params=%s",
             ckpt.name, mtime, tfm.window,
             f"{sum(p.numel() for p in pol.parameters()):,}")
    return pol


def _load_rice() -> "nx.Graph":
    """Load undirected Rice-FB graph from pickle."""
    import networkx as nx
    with open(RICE_PKL, "rb") as f:
        g = pickle.load(f)
    return g.to_undirected() if g.is_directed() else g


# ── Accounting check ──────────────────────────────────────────────────────────

def _check_accounting(result: dict, B: float, label: str, k: int) -> None:
    """Fail loudly if budget accounting identity is violated.

    Checks bankrupt_mean == 0 (SKIP enforcement means no episode overspends).

    Args:
        result: Return value of evaluate_budget_aware_policy.
        B:      Budget for this k.
        label:  Policy name for error messages.
        k:      k value for error messages.
    """
    bkr = result.get("bankrupt", {}).get("mean", 0.0)
    if bkr > 0.0:
        log.error("ACCOUNTING VIOLATION: %s k=%d B=%.3f  bankrupt_rate=%.1f%% "
                  "(SKIP enforcement should prevent this)", label, k, B, bkr * 100)
        # Do NOT raise — log and continue so we still get the eval result.


# ── Evaluation loop ──────────────────────────────────────────────────────────

def _eval_network(
    graph,
    graph_label: str,
    lstm_policies: dict,   # {label: policy}
    tfm_policy,
    device: torch.device,
) -> dict:
    """Run k-sweep on one graph for all LSTM versions + TFM.

    Args:
        graph:       NetworkX graph.
        graph_label: "FF n=1000" or "Rice-FB n=443".
        lstm_policies: Dict mapping label → policy.
        tfm_policy:  TransformerJointPolicy.
        device:      torch.device.

    Returns:
        Dict with per-k per-method results and win counts.
    """
    results_k: dict = {}

    for k in K_LIST:
        B = round(k * C, 6)
        log.info("  [%s] k=%d  B=%.3f", graph_label, k, B)
        eval_kw = dict(B=B, c=C, device=device, n_trials=N_TRIALS,
                       weight_high=WEIGHT_HIGH)

        k_res: dict = {"B": B}

        # ── TFM ──────────────────────────────────────────────────────────────
        t0 = time.time()
        r_tfm = evaluate_budget_aware_policy(tfm_policy, graph, **eval_kw)
        dt_tfm = time.time() - t0
        m_tfm = r_tfm["revenue"]["mean"]
        _check_accounting(r_tfm, B, "TFM", k)
        k_res["tfm"] = {
            "revenue": r_tfm["revenue"],
            "bankrupt_mean": r_tfm.get("bankrupt", {}).get("mean", 0.0),
        }
        log.info("    TFM   rev=%.3f  bkr=%.0f%%  (%.1fs)",
                 m_tfm, k_res["tfm"]["bankrupt_mean"] * 100, dt_tfm)

        # ── LSTM versions ─────────────────────────────────────────────────────
        k_res["lstm"] = {}
        for lstm_label, lstm_pol in lstm_policies.items():
            t0 = time.time()
            r_lstm = evaluate_budget_aware_policy(lstm_pol, graph, **eval_kw)
            dt_lstm = time.time() - t0
            m_lstm = r_lstm["revenue"]["mean"]
            _check_accounting(r_lstm, B, f"LSTM-{lstm_label}", k)
            k_res["lstm"][lstm_label] = {
                "revenue": r_lstm["revenue"],
                "bankrupt_mean": r_lstm.get("bankrupt", {}).get("mean", 0.0),
            }
            delta = m_tfm - m_lstm
            log.info("    LSTM-%s rev=%.3f  bkr=%.0f%%  (%.1fs)  delta=%+.3f",
                     lstm_label, m_lstm,
                     k_res["lstm"][lstm_label]["bankrupt_mean"] * 100,
                     dt_lstm, delta)

        results_k[f"k={k}"] = k_res

    # ── Win counts ────────────────────────────────────────────────────────────
    win_counts: dict = {}
    for lstm_label in lstm_policies:
        wins = sum(
            1 for k in K_LIST
            if results_k[f"k={k}"]["tfm"]["revenue"]["mean"]
            > results_k[f"k={k}"]["lstm"][lstm_label]["revenue"]["mean"]
        )
        win_counts[lstm_label] = wins
        log.info("  Win count TFM > LSTM-%s: %d / %d",
                 lstm_label, wins, GATE_B_N_K)

    return {"k_results": results_k, "win_counts": win_counts}


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Gate B v2: TFM vs LSTM budget eval")
    parser.add_argument(
        "--lstm_v", choices=["v1", "retrained", "both"], default="v1",
        help="Which LSTM checkpoint(s) to evaluate. "
             "v1=rev_gnn_lstm_budget_v1.pt (Jul-4 original, clean); "
             "retrained=rev_gnn_lstm_budget.pt (Welford-fixed retrain, needs PID 61149 done); "
             "both=evaluate both (only after retrain completes).",
    )
    args = parser.parse_args()

    set_seed(SEED)
    device = get_device()

    log.info("=" * 72)
    log.info("GATE B v2 — Transformer-Idea3 vs LSTM-Idea3 (budget-aware)")
    log.info("Networks: FF n=%d  +  Rice-FB n=443", N_GRAPH)
    log.info("k=%s  n_trials=%d  seed=%d  c=%.2f", K_LIST, N_TRIALS, SEED, C)
    log.info("SKIP enforcement: ON (bankrupt_mean must == 0)")
    log.info("Criterion: TFM > LSTM at ≥%d / %d k-values on EITHER network",
             GATE_B_MIN_WINS, GATE_B_N_K)
    log.info("LSTM version: %s", args.lstm_v)
    log.info("NOTE: gate_b_eval.json (2026-07-12) is VOID — weak criterion.")
    log.info("=" * 72)

    # ── Verify checkpoints ────────────────────────────────────────────────────
    required = [TFM_CKPT]
    if args.lstm_v in ("v1", "both"):
        required.append(LSTM_V1_CKPT)
    if args.lstm_v in ("retrained", "both"):
        required.append(LSTM_RT_CKPT)

    for ckpt in required:
        if not ckpt.exists():
            log.error("MISSING checkpoint: %s — abort.", ckpt)
            sys.exit(2)

    # Warn if retrained checkpoint is the same as v1_welford_bug (mid-retrain)
    if args.lstm_v in ("retrained", "both") and LSTM_RT_CKPT.exists():
        import hashlib
        def _sha256(p: Path) -> str:
            h = hashlib.sha256()
            h.update(p.read_bytes())
            return h.hexdigest()

        welford_bug = ROOT / "results" / "checkpoints" / "rev_gnn_lstm_budget_v1_welford_bug.pt"
        if welford_bug.exists() and _sha256(LSTM_RT_CKPT) == _sha256(welford_bug):
            log.warning("WARNING: rev_gnn_lstm_budget.pt hash matches v1_welford_bug.pt "
                        "(retrain ~ep 10). PID 61149 may still be running. "
                        "Proceeding with partial retrain checkpoint — results are preliminary.")

    # ── Load TFM ─────────────────────────────────────────────────────────────
    log.info("")
    log.info("Loading Transformer-Idea3 ...")
    tfm_policy = _load_transformer(TFM_CKPT, device)

    # ── Load LSTM(s) ──────────────────────────────────────────────────────────
    lstm_policies: dict = {}
    if args.lstm_v in ("v1", "both"):
        log.info("Loading LSTM-v1 (Jul-4 original) ...")
        lstm_policies["v1"] = _load_lstm(LSTM_V1_CKPT, device, "LSTM-v1")

    if args.lstm_v in ("retrained", "both"):
        log.info("Loading LSTM-retrained (Welford-fixed) ...")
        lstm_policies["retrained"] = _load_lstm(LSTM_RT_CKPT, device, "LSTM-retrained")

    # ── Build graphs ──────────────────────────────────────────────────────────
    log.info("")
    log.info("Building graphs ...")
    ff_graph = generate_forest_fire(N_GRAPH, P_FF, PB_FF, seed=SEED)
    log.info("  FF: n=%d m=%d", ff_graph.number_of_nodes(), ff_graph.number_of_edges())
    rice_graph = _load_rice()
    log.info("  Rice-FB: n=%d m=%d",
             rice_graph.number_of_nodes(), rice_graph.number_of_edges())

    # ── Evaluate on each network ───────────────────────────────────────────────
    all_results: dict = {}

    log.info("")
    log.info("── Evaluating on FF n=%d ───────────────────────────────────", N_GRAPH)
    all_results["ff_n1000"] = _eval_network(
        ff_graph, f"FF n={N_GRAPH}", lstm_policies, tfm_policy, device)

    log.info("")
    log.info("── Evaluating on Rice-FB n=443 ─────────────────────────────")
    all_results["rice_fb"] = _eval_network(
        rice_graph, "Rice-FB n=443", lstm_policies, tfm_policy, device)

    # ── Gate B decision ───────────────────────────────────────────────────────
    gate_results: dict = {}
    overall_pass = False

    for lstm_label in lstm_policies:
        ff_wins   = all_results["ff_n1000"]["win_counts"][lstm_label]
        rice_wins = all_results["rice_fb"]["win_counts"][lstm_label]
        passes_ff   = ff_wins   >= GATE_B_MIN_WINS
        passes_rice = rice_wins >= GATE_B_MIN_WINS
        passes = passes_ff or passes_rice

        gate_results[lstm_label] = {
            "ff_wins":      ff_wins,
            "rice_wins":    rice_wins,
            "ff_pass":      passes_ff,
            "rice_pass":    passes_rice,
            "gate_b_pass":  passes,
        }
        if passes:
            overall_pass = True

    # ── Save ──────────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "gate_b_v2": True,
        "void_note": "gate_b_eval.json (2026-07-12, single k=40 FF-only) is VOID.",
        "criterion": f"TFM > LSTM at >= {GATE_B_MIN_WINS} / {GATE_B_N_K} k-values on EITHER network",
        "lstm_v": args.lstm_v,
        "overall_pass": overall_pass,
        "gate_by_lstm": gate_results,
        "networks": all_results,
        "timestamp": ts,
    }
    out_path = LOG_DIR / "gate_b_eval_v2.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    # timestamped archive
    arch_path = LOG_DIR / f"gate_b_eval_v2_{ts}.json"
    with open(arch_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info("")
    log.info("Saved → %s", out_path)
    log.info("Archive → %s", arch_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 72)
    log.info("GATE B v2 SUMMARY")
    log.info("=" * 72)
    log.info("Criterion: TFM > LSTM at >= %d / %d k on EITHER network",
             GATE_B_MIN_WINS, GATE_B_N_K)
    log.info("")

    for lstm_label in lstm_policies:
        g = gate_results[lstm_label]
        log.info("  vs LSTM-%s:", lstm_label)
        log.info("    FF n=1000  wins: %d/%d  %s",
                 g["ff_wins"], GATE_B_N_K,
                 "PASS ✓" if g["ff_pass"] else "fail")
        log.info("    Rice-FB    wins: %d/%d  %s",
                 g["rice_wins"], GATE_B_N_K,
                 "PASS ✓" if g["rice_pass"] else "fail")
        log.info("    → Gate B: %s",
                 "PASS ✓" if g["gate_b_pass"] else "FAIL ✗")
        log.info("")

    # k-by-k table for primary LSTM
    primary_lstm = list(lstm_policies.keys())[0]
    log.info("k-by-k breakdown (primary lstm=%s):", primary_lstm)
    log.info("%-5s | %-4s | %9s | %9s | %9s | %9s | %7s",
             "k", "B", "LSTM-FF", "TFM-FF", "LSTM-Rice", "TFM-Rice", "TFM wins")
    log.info("-" * 73)
    for k in K_LIST:
        key = f"k={k}"
        B   = round(k * C, 6)
        ff_l  = all_results["ff_n1000"]["k_results"][key]["lstm"][primary_lstm]["revenue"]["mean"]
        ff_t  = all_results["ff_n1000"]["k_results"][key]["tfm"]["revenue"]["mean"]
        ri_l  = all_results["rice_fb"]["k_results"][key]["lstm"][primary_lstm]["revenue"]["mean"]
        ri_t  = all_results["rice_fb"]["k_results"][key]["tfm"]["revenue"]["mean"]
        ff_w  = "✓" if ff_t > ff_l else " "
        ri_w  = "✓" if ri_t > ri_l else " "
        log.info("k=%-3d | %.2f | %9.3f | %9.3f | %9.3f | %9.3f | FF:%s Rice:%s",
                 k, B, ff_l, ff_t, ri_l, ri_t, ff_w, ri_w)

    log.info("=" * 72)
    log.info("OVERALL GATE B: %s",
             "PASS ✓ (paper includes Transformer-Idea3)" if overall_pass
             else "FAIL ✗ (more budget training needed or TFM underperforms)")
    log.info("=" * 72)

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
