#!/usr/bin/env python3
"""experiments/run_phase2_only.py — Phase 2 REINFORCE from ph1end checkpoint.

Loads rev_gnn_lstm_unified_ph1end.pt (21-dim), runs Phase 2 REINFORCE
(200 epochs, per-bucket Welford advantage), saves best → rev_gnn_lstm_unified.pt.

Requires PYTORCH_ENABLE_MPS_FALLBACK=1 to be set in env (for Dirichlet sampling on MPS).
"""

from __future__ import annotations
import json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import phase2_reinforce and helpers from the training script
from experiments.run_budget_unified_training import (
    phase2_reinforce,
    _build_policy_21dim,
    _to_edge_index,
    FF_P, FF_PB, C, CKPT_DIR, LOG_DIR, TRAIN_SIZES, PH2_EPOCHS,
)
from src.env.graph_generators import generate_forest_fire
from src.utils.features import compute_static_features, build_graph_feature_cache

PH1_CKPT = os.path.join(CKPT_DIR, "rev_gnn_lstm_unified_ph1end.pt")
OUT_CKPT  = os.path.join(CKPT_DIR, "rev_gnn_lstm_unified.pt")


def main():
    # Verify MPS fallback is enabled
    fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    if torch.backends.mps.is_available() and fallback != "1":
        print("WARNING: PYTORCH_ENABLE_MPS_FALLBACK=1 not set. Dirichlet sampling "
              "will fail on MPS. Set it or use CPU.")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 70)
    print("run_phase2_only.py — Phase 2 REINFORCE from ph1end checkpoint")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"PYTORCH_ENABLE_MPS_FALLBACK={fallback}")

    # ── Load Phase 1 checkpoint ───────────────────────────────────────────────
    if not os.path.exists(PH1_CKPT):
        print(f"ERROR: {PH1_CKPT} not found. Run run_budget_unified_training.py first.")
        sys.exit(1)

    import hashlib
    sha = hashlib.sha256(open(PH1_CKPT, "rb").read()).hexdigest()
    print(f"\nLoading Phase 1 checkpoint: {PH1_CKPT}")
    print(f"SHA256: {sha[:16]}...")

    policy = _build_policy_21dim().to(device)
    sd = torch.load(PH1_CKPT, map_location=device)
    policy.load_state_dict(sd, strict=True)
    print(f"Policy loaded: {sum(p.numel() for p in policy.parameters())} params")

    # ── Rebuild training graphs + caches ─────────────────────────────────────
    print(f"\nGenerating {len(TRAIN_SIZES)} training graphs...")
    training_data = []
    for i, n in enumerate(TRAIN_SIZES):
        g = generate_forest_fire(n, p=FF_P, pb=FF_PB, seed=i)
        static  = compute_static_features(g)
        cache   = build_graph_feature_cache(g, static)
        ei      = _to_edge_index(g, device)
        training_data.append({
            "graph": g, "cache": cache,
            "edge_index": ei, "n": g.number_of_nodes(),
        })
        print(f"  g{i}: n={g.number_of_nodes()} m={g.number_of_edges()}")

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    t0 = time.time()
    ph2_log = phase2_reinforce(
        policy, training_data, n_epochs=PH2_EPOCHS,
        device=device, save_every=25
    )

    # ── Ensure best checkpoint exists ─────────────────────────────────────────
    if not os.path.exists(OUT_CKPT):
        torch.save(policy.state_dict(), OUT_CKPT)
        print(f"[final] No best checkpoint selected; saving final → {OUT_CKPT}")

    final_sha = hashlib.sha256(open(OUT_CKPT, "rb").read()).hexdigest()
    print(f"\n[final] {OUT_CKPT}  SHA256={final_sha[:16]}...")

    # ── Save log ──────────────────────────────────────────────────────────────
    log_path = os.path.join(LOG_DIR, "phase2_only_log.json")
    with open(log_path, "w") as f:
        json.dump({
            "ph1_ckpt": PH1_CKPT, "ph1_sha256": sha,
            "phase2": ph2_log["phase2"],
            "best_epoch": ph2_log["best_epoch"],
            "best_min_mean": ph2_log["best_min_mean"],
            "final_sha256": final_sha,
            "duration_s": time.time() - t0,
        }, f, indent=2)
    print(f"[log] → {log_path}")
    print(f"\nDone. Total wall time: {(time.time()-t0)/60:.1f} min")
    print("Run experiments/run_unified_sweep.py to evaluate.")


if __name__ == "__main__":
    main()
