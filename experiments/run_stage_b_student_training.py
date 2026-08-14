"""experiments/run_stage_b_student_training.py — Stage B: student imitation gates (R1).

Stage B spec:
  Train TWO students, identical architecture (SequentialJointPolicy, 21-dim,
  zero-init budget column, warm-start from rev_gnn_lstm.pt), identical
  Phase-1 recipe (150 epochs, CE + 0.3*MSE), on n=200 graphs:

    Student-M: mixed-expert trajectories (current scheme) — the control.
    Student-R: rollout-expert trajectories from Stage A cache.

  NO Phase-2 RL yet.
  Evaluate both imitation-only at k={1,3,10}, n=200, 3 seeds, SKIP enforcement.

GATE R1: Student-R >= Student-M at >= 2 of 3 k values.
  FAIL → stop. Report both tables + one-line verdict; future work.
  PASS → Stage C.

Usage:
  python experiments/run_stage_b_student_training.py --student M   # Student-M
  python experiments/run_stage_b_student_training.py --student R   # Student-R
  python experiments/run_stage_b_student_training.py --eval_gate   # run gate R1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim

from src.env.graph_generators import generate_forest_fire
from src.env.budget_revenue_env import BudgetEnvConfig, BudgetRevenueEnv
from src.evaluation.budget_baselines import greedy_discount_budget
from src.evaluation.dp_calibrated_v2 import dp_calibrated_v2_budget
from src.evaluation.dp_calibrated_v3 import dp_calibrated_v3_budget
from src.training.mixed_expert_trajectories import generate_budget_expert_trajectory
from src.evaluation.rollout_expert import (
    RolloutExpertConfig,
    generate_rollout_expert_trajectory,
)
import random

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
from src.utils.budget_features import compute_budget_node_features_fast
from src.utils.features import compute_static_features, build_graph_feature_cache
from src.evaluation.budget_baselines import evaluate_budget_aware_policy

# ── Constants ──────────────────────────────────────────────────────────────────
N_NODES   = 200
P_FF, PB_FF = 0.37, 0.32
FF_SEED   = 42
C         = 0.3
K_VALUES  = [1, 3, 10]
N_TRIALS  = 3
N_EPOCHS  = 150      # Phase 1 only
LR        = 1e-3
CE_WEIGHT = 1.0
MSE_WEIGHT = 0.3
WARM_START_CKPT = "results/checkpoints/rev_gnn_lstm.pt"

CKPT_M = "results/checkpoints/stage_b_student_M.pt"
CKPT_R = "results/checkpoints/stage_b_student_R.pt"
GATE_R1_RESULTS = "results/logs/stage_b_gate_r1.json"
os.makedirs("results/checkpoints", exist_ok=True)
os.makedirs("results/logs", exist_ok=True)


# ── Model loading ──────────────────────────────────────────────────────────────

def _build_policy(device: torch.device):
    """Build SequentialJointPolicy(GraphSAGE 21-dim + EpisodeLSTM)."""
    from src.models.encoders.graphsage import GraphSAGEEncoder
    from src.models.encoders.sequence_models import EpisodeLSTM
    from src.models.policies.sequential_joint_policy import SequentialJointPolicy

    encoder = GraphSAGEEncoder(in_dim=21, hidden_dim=64)
    lstm    = EpisodeLSTM(graph_dim=64, lstm_hidden=64)
    return SequentialJointPolicy(encoder, lstm, gnn_dim=64, context_dim=64).to(device)


def _load_student(device: torch.device):
    """Build model, optionally warm-start from rev_gnn_lstm.pt (in_dim 20→21)."""
    model = _build_policy(device)

    if not os.path.exists(WARM_START_CKPT):
        print(f"  [WARN] warm-start {WARM_START_CKPT} not found; random init.")
        return model

    try:
        loaded = torch.load(WARM_START_CKPT, map_location=device)
        # Support both raw state-dict and {"state_dict": ...} wrappers
        if isinstance(loaded, dict) and "state_dict" in loaded:
            loaded = loaded["state_dict"]
        model_state = model.state_dict()
        patched = 0
        for key, val in loaded.items():
            if key not in model_state:
                continue
            if val.shape == model_state[key].shape:
                model_state[key] = val
            elif key == "encoder.input_proj.weight":
                # (64, 20) → (64, 21): copy old weights, zero-init budget col
                new_w = torch.zeros_like(model_state[key])
                new_w[:, :val.shape[1]] = val
                model_state[key] = new_w
                patched += 1
        model.load_state_dict(model_state)
        print(f"  Warm-start loaded from {WARM_START_CKPT}  (patched {patched} weight(s))")
    except Exception as e:
        print(f"  [WARN] warm-start load failed ({e}); random init.")
    return model


# ── Trajectory source ──────────────────────────────────────────────────────────

def _get_trajectory(
    graph,
    k: int,
    seed: int,
    source: str,
    base_cfg: BudgetEnvConfig,
) -> List[dict]:
    """Get one trajectory from either 'M' (mixed-expert) or 'R' (rollout-expert)."""
    if source == "M":
        return generate_budget_expert_trajectory(graph, k=k, c=C, seed=seed)
    elif source == "R":
        expert_cfg = RolloutExpertConfig(c=C)
        return generate_rollout_expert_trajectory(
            graph, base_cfg, k=k, c=C, seed=seed,
            expert_cfg=expert_cfg, force_rebuild=False,
        )
    else:
        raise ValueError(f"Unknown source '{source}'")


# ── Training loop (Phase 1: imitation) ────────────────────────────────────────

def train_student(
    student_id: str,
    graph,
    base_cfg: BudgetEnvConfig,
    device: torch.device,
) -> None:
    """Train one student via imitation learning (Phase 1 only, 150 epochs)."""
    model = _load_student(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    ce_loss_fn  = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()

    n = graph.number_of_nodes()
    static_feats = compute_static_features(graph)
    feat_cache   = build_graph_feature_cache(graph, static_feats)

    edges_list   = list(graph.edges())
    src = [u for u, v in edges_list] + [v for u, v in edges_list]
    dst = [v for u, v in edges_list] + [u for u, v in edges_list]
    edge_index = torch.tensor([src, dst], dtype=torch.long).to(device)

    ckpt_out = CKPT_M if student_id == "M" else CKPT_R
    best_loss = float("inf")

    print(f"\n  Training Student-{student_id} ({N_EPOCHS} epochs, source={'mixed' if student_id=='M' else 'rollout'})")

    for epoch in range(N_EPOCHS):
        epoch_losses = []

        # Sample trajectories: use all k-values and 3 seeds each
        for k in K_VALUES:
            for seed in range(N_TRIALS):
                traj = _get_trajectory(graph, k, seed, student_id, base_cfg)
                if not traj:
                    continue

                B_0 = float(k * C)
                env_for_feat = BudgetRevenueEnv(graph, BudgetEnvConfig(
                    budget_B=B_0, production_cost=C, seed=seed, weight_high=2.0,
                ))
                env_for_feat.reset()

                model.reset_episode(device)
                model.train()

                step_loss = 0.0
                for step_dict in traj:
                    node_idx_exp = step_dict["node_idx"]
                    disc_exp     = step_dict["discount"]

                    feats = compute_budget_node_features_fast(
                        feat_cache, env_for_feat.S, env_for_feat.offered,
                        env_for_feat.t, k=n, env=env_for_feat,
                    )
                    x    = torch.tensor(feats, dtype=torch.float32).to(device)
                    mask = torch.zeros(n, dtype=torch.bool, device=device)
                    for idx in env_for_feat.available_nodes:
                        mask[idx] = True

                    if mask.sum() == 0:
                        break

                    # forward returns (masked_scores, node_embeddings, context, graph_emb)
                    masked_scores, h_emb, context, _ = model(x, edge_index, mask)

                    # CE: expert node's LOCAL index within the available subset
                    avail_idx     = mask.nonzero(as_tuple=True)[0]
                    local_matches = (avail_idx == node_idx_exp).nonzero(as_tuple=True)[0]
                    if len(local_matches) == 0:
                        break  # expert node no longer available (guard)
                    local_target  = local_matches[0:1]        # (1,) local rank
                    masked_logits = masked_scores[mask]        # (n_avail,)
                    loss_ce = ce_loss_fn(masked_logits.unsqueeze(0), local_target)

                    # MSE: Beta mean vs expert discount (both shape [1])
                    combined_exp = torch.cat([h_emb[node_idx_exp], context])
                    disc_dist    = model.get_discount_distribution(combined_exp)
                    disc_t       = torch.tensor([disc_exp], dtype=torch.float32, device=device)
                    loss_mse     = mse_loss_fn(disc_dist.mean.unsqueeze(0), disc_t)

                    loss = CE_WEIGHT * loss_ce + MSE_WEIGHT * loss_mse
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    step_loss += loss.item()

                    # Advance env state
                    try:
                        env_for_feat.step(node_idx_exp, disc_exp)
                    except Exception:
                        break

                    model.update_sequence_state(disc_exp, step_dict["accepted"], step_dict["price"])

                if traj:
                    epoch_losses.append(step_loss / len(traj))

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model.state_dict(), ckpt_out)

        if epoch % 25 == 0 or epoch == N_EPOCHS - 1:
            print(f"    epoch {epoch:3d}/{N_EPOCHS}: loss={mean_loss:.4f}  best={best_loss:.4f}")

    print(f"  Student-{student_id} best checkpoint saved to {ckpt_out}  (loss={best_loss:.4f})")


# ── Evaluation ─────────────────────────────────────────────────────────────────

def _eval_student(ckpt_path: str, graph, device: torch.device) -> dict:
    """Evaluate a student checkpoint over k=[1,3,10], 3 seeds each."""
    if not os.path.exists(ckpt_path):
        return {"error": f"ckpt not found: {ckpt_path}"}

    model = _build_policy(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    results = {}
    for k in K_VALUES:
        B_0 = float(k * C)
        res = evaluate_budget_aware_policy(
            model, graph, B=B_0, c=C, device=device, n_trials=N_TRIALS,
        )
        results[k] = {"mean": res["revenue"]["mean"], "std": res["revenue"]["std"],
                      "all": res["revenue"]["all"]}
    return results


# ── Gate R1 ────────────────────────────────────────────────────────────────────

def run_gate_r1(graph, device: torch.device) -> None:
    print("\n" + "=" * 64)
    print("GATE R1 — Student-R vs Student-M")

    res_M = _eval_student(CKPT_M, graph, device)
    res_R = _eval_student(CKPT_R, graph, device)

    print(f"\n  {'k':>4}  {'Student-M':>10}  {'Student-R':>10}  {'Δ%':>8}  {'Pass?':>6}")
    print("  " + "-" * 50)

    gate_row = {}
    for k in K_VALUES:
        m_rev = res_M[k]["mean"] if k in res_M else 0.0
        r_rev = res_R[k]["mean"] if k in res_R else 0.0
        pct   = (r_rev - m_rev) / (m_rev + 1e-9) * 100.0
        gate_pass = r_rev >= m_rev
        print(f"  {k:>4}  {m_rev:>10.2f}  {r_rev:>10.2f}  {pct:>+7.1f}%  "
              f"{'PASS' if gate_pass else 'FAIL':>6}")
        gate_row[k] = {"M": m_rev, "R": r_rev, "pct": pct, "pass": gate_pass}

    n_pass  = sum(1 for row in gate_row.values() if row["pass"])
    r1_pass = n_pass >= 2
    print(f"\n  R1: {'PASS ✓ — proceed to Stage C' if r1_pass else 'FAIL ✗ — stop pilot, future work'}")
    print(f"  ({n_pass}/3 k-values where Student-R >= Student-M)")

    output = {
        "verdict": "PASS" if r1_pass else "FAIL",
        "n_pass":  n_pass,
        "per_k":   gate_row,
        "student_M_results": res_M,
        "student_R_results": res_R,
    }
    with open(GATE_R1_RESULTS, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to {GATE_R1_RESULTS}")
    sys.exit(0 if r1_pass else 1)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student",  choices=["M", "R"], default=None,
                        help="Train Student-M (mixed) or Student-R (rollout)")
    parser.add_argument("--eval_gate", action="store_true",
                        help="Run Gate R1 evaluation (needs both checkpoints)")
    args = parser.parse_args()

    device = get_device()
    set_seed(42)

    graph = generate_forest_fire(N_NODES, p=P_FF, pb=PB_FF, seed=FF_SEED)
    base_cfg = BudgetEnvConfig(budget_B=3.0, production_cost=C, seed=0, weight_high=2.0)

    print(f"Stage B — Student imitation gate")
    print(f"  Graph: FF n={N_NODES}, device={device}")

    if args.eval_gate:
        run_gate_r1(graph, device)
    elif args.student in ("M", "R"):
        train_student(args.student, graph, base_cfg, device)
    else:
        print("Run with --student M, --student R, or --eval_gate")
        sys.exit(1)


if __name__ == "__main__":
    main()
