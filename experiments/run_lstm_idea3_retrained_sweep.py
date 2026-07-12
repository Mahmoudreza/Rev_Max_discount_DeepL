"""experiments/run_lstm_idea3_retrained_sweep.py — Idea-3 LSTM retrained standard sweep.

Evaluates the Welford-fixed retrain checkpoint (rev_gnn_lstm_budget.pt)
on the standard Idea-3 protocol:
  k = [1,2,3,5,8,10,15,20,30,40], B = k * 0.3, c = 0.3
  FF n=1000  AND  Rice-FB n=443
  n_trials=3, seed=42, SKIP enforcement, per-episode accounting check.

Saves: results/logs/lstm_idea3_retrained_sweep.json  (NEW file, do not overwrite
       existing budget_eval_c0.3.json or dp_upgrade_eval_rice_lstm.json).

Also loads "published" numbers from budget_eval_c0.3.json and
dp_upgrade_eval_rice_lstm.json for side-by-side comparison.

Pre-committed decision rule:
  If retrained >= published at BOTH flag points:
      k=3  FF  (published=327.9, threshold>=295, +/-10%)
      k=10 Rice (published=68.6, threshold>=61.7, +/-10%)
  → paper uses retrained checkpoint everywhere; published archived.
  Else → paper keeps published numbers; retrained archived with note.

Usage:
    cd revmax-aaai2027 && source venv/bin/activate
    nice -n 10 python -u experiments/run_lstm_idea3_retrained_sweep.py \
        > /tmp/lstm_retrained_sweep.log 2>&1 &
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.utils.helpers import load_config_with_base, set_seed, get_device
from src.env.graph_generators import generate_forest_fire
from src.evaluation.budget_baselines import evaluate_budget_aware_policy

# ── Constants ─────────────────────────────────────────────────────────────────
SEED        = 42
N_FF        = 1000
P_FF        = 0.37
PB_FF       = 0.32
C           = 0.3
WEIGHT_HIGH = 2.0
N_TRIALS    = 3
K_LIST      = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40]

RICE_PKL     = ROOT / "data" / "processed" / "rice_facebook.pkl"
CKPT         = ROOT / "results" / "checkpoints" / "rev_gnn_lstm_budget.pt"
CFG_DP       = ROOT / "configs" / "experiments" / "dp_upgrade.yaml"
LOG_DIR      = ROOT / "results" / "logs"
OUT_JSON     = LOG_DIR / "lstm_idea3_retrained_sweep.json"

# Flag points for decision rule
FLAG_FF_K    = 3
FLAG_RICE_K  = 10
PUB_FF_K3    = 327.9     # from budget_eval_c0.3.json
PUB_RICE_K10 = 68.57     # from dp_upgrade_eval_rice_lstm.json
THRESH_FF    = PUB_FF_K3   * 0.90   # 295.1
THRESH_RICE  = PUB_RICE_K10 * 0.90  # 61.7

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "lstm_idea3_retrained_sweep.log")),
    ],
)
log = logging.getLogger(__name__)


def _load_policy(ckpt: Path, device: torch.device):
    """Load LSTM-Idea3 budget policy from checkpoint.

    Args:
        ckpt:   Path to checkpoint.
        device: Target device.

    Returns:
        SequentialJointPolicy in eval mode.
    """
    from src.models.encoders.graphsage import GraphSAGEEncoder
    from src.models.encoders.sequence_models import EpisodeLSTM
    from src.models.policies.sequential_joint_policy import SequentialJointPolicy

    import hashlib
    sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    mtime = datetime.fromtimestamp(ckpt.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    log.info("  Checkpoint: %s", ckpt.name)
    log.info("  SHA256:     %s", sha)
    log.info("  mtime:      %s", mtime)

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

    n_params = sum(p.numel() for p in pol.parameters())
    log.info("  Params: %s  lstm_hidden=%d", f"{n_params:,}", lstm_hidden)
    return pol, sha


def _load_rice():
    """Load undirected Rice-FB graph."""
    import networkx as nx
    with open(RICE_PKL, "rb") as f:
        g = pickle.load(f)
    return g.to_undirected() if g.is_directed() else g


def _load_published() -> tuple[dict, dict]:
    """Load published LSTM-Idea3 numbers for side-by-side comparison.

    Returns:
        Tuple of (ff_pub_dict, rice_pub_dict) mapping k-label → mean revenue.
    """
    ff_pub = {}
    rice_pub = {}

    ff_path = LOG_DIR / "budget_eval_c0.3.json"
    if ff_path.exists():
        d = json.load(open(ff_path))
        for k in K_LIST:
            dd = d.get(f"k={k}", {}).get("LSTM-Idea3", {})
            if isinstance(dd, dict):
                rev = dd.get("revenue", {}).get("mean", dd.get("mean"))
            else:
                rev = dd
            ff_pub[k] = rev

    rice_path = LOG_DIR / "dp_upgrade_eval_rice_lstm.json"
    if rice_path.exists():
        d = json.load(open(rice_path))
        for k in K_LIST:
            dd = d.get(f"k={k}", {}).get("LSTM-Idea3", {})
            if isinstance(dd, dict):
                rev = dd.get("revenue", {}).get("mean", dd.get("mean"))
            else:
                rev = dd
            rice_pub[k] = rev

    return ff_pub, rice_pub


def _sweep(pol, graph, label: str, device) -> dict:
    """Run k-sweep on graph with SKIP enforcement and accounting checks.

    Args:
        pol:    Policy.
        graph:  NetworkX graph.
        label:  Graph label for logging.
        device: torch.device.

    Returns:
        Dict {k: {"mean": float, "std": float, "B": float, "bankrupt_mean": float}}.
    """
    results = {}
    for k in K_LIST:
        B = round(k * C, 6)
        t0 = time.time()
        r = evaluate_budget_aware_policy(
            pol, graph, B=B, c=C, device=device,
            n_trials=N_TRIALS, weight_high=WEIGHT_HIGH,
        )
        dt = time.time() - t0
        m = r["revenue"]["mean"]
        s = r["revenue"].get("std", 0.0)
        bkr = r.get("bankrupt", {}).get("mean", 0.0)

        if bkr > 0.0:
            log.error("ACCOUNTING VIOLATION [%s] k=%d B=%.3f bkr=%.1f%% — "
                      "SKIP enforcement should prevent overspend.",
                      label, k, B, bkr * 100)

        log.info("  [%s] k=%-2d B=%.2f  rev=%8.3f ± %6.3f  bkr=%.0f%%  %.1fs",
                 label, k, B, m, s, bkr * 100, dt)
        results[k] = {
            "k": k, "B": B,
            "mean": m, "std": s,
            "bankrupt_mean": bkr,
        }
    return results


def main() -> None:
    set_seed(SEED)
    device = get_device()

    log.info("=" * 70)
    log.info("LSTM-Idea3 Retrained Standard Sweep")
    log.info("Checkpoint: %s", CKPT.name)
    log.info("k=%s  n_trials=%d  seed=%d  c=%.2f", K_LIST, N_TRIALS, SEED, C)
    log.info("Flag points: FF k=%d (pub=%.1f, thr=%.1f) | Rice k=%d (pub=%.1f, thr=%.1f)",
             FLAG_FF_K, PUB_FF_K3, THRESH_FF,
             FLAG_RICE_K, PUB_RICE_K10, THRESH_RICE)
    log.info("=" * 70)

    if not CKPT.exists():
        log.error("Checkpoint not found: %s", CKPT)
        sys.exit(2)

    pol, sha = _load_policy(CKPT, device)
    ff_pub, rice_pub   = _load_published()
    ff_graph   = generate_forest_fire(N_FF, P_FF, PB_FF, seed=SEED)
    rice_graph = _load_rice()
    log.info("FF: n=%d m=%d  Rice: n=%d m=%d",
             ff_graph.number_of_nodes(), ff_graph.number_of_edges(),
             rice_graph.number_of_nodes(), rice_graph.number_of_edges())

    # ── Sweep ─────────────────────────────────────────────────────────────────
    log.info("")
    log.info("── FF n=1000 ─────────────────────────────────────────────────")
    ff_res = _sweep(pol, ff_graph, "FF", device)

    log.info("")
    log.info("── Rice-FB n=443 ─────────────────────────────────────────────")
    rice_res = _sweep(pol, rice_graph, "Rice", device)

    # ── Side-by-side table ────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 80)
    log.info("SIDE-BY-SIDE: LSTM-Idea3 retrained vs published")
    log.info("%-5s | %10s | %10s | %7s || %10s | %10s | %7s",
             "k", "FF-ret", "FF-pub", "delta", "Rice-ret", "Rice-pub", "delta")
    log.info("-" * 80)
    for k in K_LIST:
        fr = ff_res[k]["mean"]
        fp = ff_pub.get(k)
        rr = rice_res[k]["mean"]
        rp = rice_pub.get(k)
        fd = f"{fr-fp:+.1f}" if fp is not None else "n/a"
        rd = f"{rr-rp:+.1f}" if rp is not None else "n/a"
        flag_ff   = "◄FLAG" if k == FLAG_FF_K else ""
        flag_rice = "◄FLAG" if k == FLAG_RICE_K else ""
        log.info("k=%-3d | %10.3f | %10.1f | %7s %s || %10.3f | %10.2f | %7s %s",
                 k, fr, fp if fp is not None else -1, fd, flag_ff,
                 rr, rp if rp is not None else -1, rd, flag_rice)
    log.info("=" * 80)

    # ── Decision rule ─────────────────────────────────────────────────────────
    ret_ff   = ff_res[FLAG_FF_K]["mean"]
    ret_rice = rice_res[FLAG_RICE_K]["mean"]
    passes_ff   = ret_ff   >= THRESH_FF
    passes_rice = ret_rice >= THRESH_RICE
    retrained_wins = passes_ff and passes_rice

    log.info("")
    log.info("DECISION RULE (pre-committed):")
    log.info("  FF   k=%d: retrained=%.3f  threshold=%.1f  %s",
             FLAG_FF_K, ret_ff, THRESH_FF,
             "PASS ✓" if passes_ff else "FAIL ✗")
    log.info("  Rice k=%d: retrained=%.3f  threshold=%.1f  %s",
             FLAG_RICE_K, ret_rice, THRESH_RICE,
             "PASS ✓" if passes_rice else "FAIL ✗")
    log.info("")
    if retrained_wins:
        log.info("→ VERDICT: retrained >= published at BOTH flag points.")
        log.info("  PAPER uses RETRAINED checkpoint. Published numbers archived.")
        log.info("  Checkpoint: %s (sha256=%s)", CKPT.name, sha[:16])
    else:
        log.info("→ VERDICT: retrained REGRESSES at ≥1 flag point.")
        log.info("  PAPER keeps PUBLISHED numbers.")
        log.info("  Retrained checkpoint archived with note.")

    # ── Save ──────────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "checkpoint": str(CKPT.name),
        "sha256": sha,
        "mtime": datetime.fromtimestamp(CKPT.stat().st_mtime).isoformat(),
        "config": {
            "seed": SEED, "n_ff": N_FF, "c": C, "weight_high": WEIGHT_HIGH,
            "n_trials": N_TRIALS, "k_list": K_LIST,
        },
        "ff_n1000": ff_res,
        "rice_fb": rice_res,
        "decision": {
            "flag_ff_k": FLAG_FF_K, "pub_ff": PUB_FF_K3, "thresh_ff": THRESH_FF,
            "flag_rice_k": FLAG_RICE_K, "pub_rice": PUB_RICE_K10, "thresh_rice": THRESH_RICE,
            "ret_ff": ret_ff, "ret_rice": ret_rice,
            "passes_ff": passes_ff, "passes_rice": passes_rice,
            "retrained_wins": retrained_wins,
            "paper_model": "retrained" if retrained_wins else "published",
        },
        "timestamp": ts,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    # timestamped archive
    arch = LOG_DIR / f"lstm_idea3_retrained_sweep_{ts}.json"
    with open(arch, "w") as f:
        json.dump(out, f, indent=2)

    log.info("")
    log.info("Saved → %s", OUT_JSON)
    log.info("Archive → %s", arch)


if __name__ == "__main__":
    main()
