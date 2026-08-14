"""experiments/run_gnn_fairness_audit.py

Task 2 — Add Rev-GNN-IM-RL and Rev-GNN-LSTM rows to fairness_audit.json,
then print the COMPLETE audit table (all methods x all graphs).
NEW FILE — does NOT modify any existing file except appending to audit JSON.

GNN nodes are selected by policy score; pricing uses IDENTICAL influence-tier
logic as greedy_discount (labels used ONLY for fairness measurement, not as input).
group_flag (feature dim index 16, 0-indexed) stays 0.0 — as trained.
"""
from __future__ import annotations
import sys, json, os
sys.path.insert(0, ".")

import numpy as np
import torch
from omegaconf import OmegaConf

from src.evaluation.baselines import _make_env, _compute_normalized_infl, _rayleigh_price
from src.evaluation.fairness_audit import (
    group_metrics_at_checkpoints, aggregate_trials, assert_trajectory_consistent,
)
from src.env.sbm_generators import two_block_graph, load_rice_facebook_with_labels
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.policies.joint_policy import JointPolicy
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.models.encoders.sequence_models import EpisodeLSTM
from src.utils.features import compute_node_features, compute_static_features

BASE_CFG_YAML = """
project:
  name: gnn-audit
  seed: 0
graph:
  type: rice_facebook
  n_nodes: 443
features:
  dim: 20
encoder:
  hidden_dim: 64
  n_layers: 2
  dropout: 0.0
influence:
  model: monotone
  n_mc_samples: 10
  b: 1.0
  weight_low: 0.0
  weight_high: 2.0
reward:
  type: revenue
  gamma: 1.0
budget:
  k: 443
env:
  k: 443
  budget: 0.0
"""

SEEDS = [0, 1, 2, 3, 4]
CKPTS = {
    "Rev-GNN-IM-RL": ("results/checkpoints/rev_gnn_im_rl.pt", "joint"),
    "Rev-GNN-LSTM":  ("results/checkpoints/rev_gnn_lstm.pt",  "lstm"),
}
CHECKPOINTS_K   = {25, 50, 100, 200, 300, 443}
CHECKPOINTS_SBM = {50, 100, 200, 500, 750, 1000}


def _price_from_infl(infl, b):
    if infl < 2.0 / 6.0:  return 0.0
    if infl < 4.0 / 6.0:  return _rayleigh_price(2.0 / 6.0, b)
    return _rayleigh_price(4.0 / 6.0, b)


def _load_policy(ckpt_path, model_type, cfg):
    h = cfg.encoder.hidden_dim
    enc = GraphSAGEEncoder(in_dim=cfg.features.dim,
                           hidden_dim=h,
                           n_layers=cfg.encoder.n_layers,
                           dropout=cfg.encoder.dropout)
    if model_type == "joint":
        policy = JointPolicy(enc, hidden_dim=h)
    else:
        # EpisodeLSTM takes graph_dim/lstm_hidden; SequentialJointPolicy takes gnn_dim/context_dim
        seq = EpisodeLSTM(graph_dim=h, lstm_hidden=h)
        policy = SequentialJointPolicy(enc, seq, gnn_dim=h, context_dim=h)

    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    policy.load_state_dict(state, strict=False)
    policy.eval()
    return policy


def _gnn_trajectory(graph, cfg, labels, policy, model_type):
    """Run GNN policy with influence-tier pricing; return audit-compatible list."""
    import torch

    env = _make_env(graph, cfg)
    env.reset()
    n   = env.n
    b   = float(cfg.influence.b)
    lw  = env._link_weights
    nodes = list(graph.nodes())

    # Build static edge index once (policies take x, edge_index, available_mask)
    edges = list(graph.edges())
    src = [e[0] for e in edges] + [e[1] for e in edges]
    dst = [e[1] for e in edges] + [e[0] for e in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Compute static features once (dims 1-10)
    static_feats = compute_static_features(graph)

    offered: set = set()
    audit = []
    rev = 0.0

    for _ in range(n):
        remaining = [v for v in nodes if v not in offered]
        if not remaining:
            break

        # Node features (20-dim), group_flag dim=15 (0-indexed) stays 0
        feats = compute_node_features(
            graph=graph,
            static_features=static_feats,
            S=frozenset(env.S),
            offered=frozenset(offered),
            t=len(offered),
            n=n, k=n, env=env,
        )  # shape (n, 20)
        feats[:, 15] = 0.0  # zero out group flag

        x = torch.tensor(feats, dtype=torch.float32)

        # available_mask: True = available, False = offered (policy uses ~mask → -inf)
        avail_mask = torch.ones(n, dtype=torch.bool)
        for v in offered:
            avail_mask[env.node_to_idx[v]] = False

        with torch.no_grad():
            out = policy(x, edge_index, avail_mask)

        if isinstance(out, (list, tuple)):
            scores = out[0]
        else:
            scores = out

        scores_np = scores.detach().cpu().numpy().flatten()
        scores_np[[env.node_to_idx[v] for v in offered]] = -1e9
        best_idx = int(np.argmax(scores_np))
        target   = nodes[best_idx]

        infl     = _compute_normalized_infl(graph, target, env.S, lw)
        price    = _price_from_infl(infl, b)
        true_val = env._true_valuation(target)
        node_idx = env.node_to_idx[target]
        group    = int(labels[node_idx])

        if price == 0.0:
            env.S.add(target)
            env._influence_cache = {}
            accepted = True
        elif true_val >= price:
            env.S.add(target)
            env._influence_cache = {}
            accepted = True
        else:
            accepted = False

        if accepted:
            audit.append({"node": node_idx, "price": price, "est_val": price})
            rev += price

        offered.add(target)
        env.offered.add(target)
        env.t += 1

    return audit, rev


def traj_to_audit(audit, rev):
    return audit, rev


def run_gnn_on_graph(graph, labels, checkpoints_K, cfg, model_name, ckpt_path, mtype):
    n   = graph.number_of_nodes()
    cps = sorted(k for k in checkpoints_K if k <= n)
    trials = []
    policy = _load_policy(ckpt_path, mtype, cfg)
    for seed in SEEDS:
        cfg_s = OmegaConf.merge(cfg, OmegaConf.create({"project": {"seed": seed}}))
        env = _make_env(graph, cfg_s)
        env.reset()
        audit, rev = _gnn_trajectory(graph, cfg_s, labels, policy, mtype)
        assert_trajectory_consistent(audit, rev)
        trials.append(group_metrics_at_checkpoints(audit, labels, cps))
    return aggregate_trials(trials, cps)


def merge_and_print(audit_path, f1_path):
    """Merge f1.json rows into audit dict and print combined table."""
    with open(audit_path) as f:
        audit = json.load(f)
    with open(f1_path) as f:
        f1 = json.load(f)

    # Merge Fair-Greedy into audit
    for gkey in f1:
        if gkey in audit:
            for m in ["Fair-Greedy"]:
                if m in f1[gkey]:
                    audit[gkey]["methods"][m] = f1[gkey][m]

    # Print full table
    COL_W = 10
    METHOD_ORDER = ["IE-Strategy", "Greedy-Discount", "Rev-GNN-IM-RL",
                    "Rev-GNN-LSTM", "Fair-Greedy"]
    GRAPH_LABELS = [("rice_fb", "Rice-FB n=443"),
                    ("SBM_h0.9", "SBM h=0.9 n=1000"),
                    ("SBM_h0.5", "SBM h=0.5 n=1000"),
                    ("SBM_h0.7", "SBM h=0.7 n=1000")]

    def fm_val(agg, key):
        if agg is None:
            return "N/A"
        fm = agg.get("final", {}) if isinstance(agg, dict) else {}
        v = fm.get(key, {}).get("mean", None)
        if v is None: return "—"
        return f"{v:.3f}"

    for gkey, glabel in GRAPH_LABELS:
        if gkey not in audit:
            continue
        nsB = audit[gkey].get("node_share_B", float("nan"))
        methods_data = audit[gkey]["methods"]
        print(f"\n{'='*100}")
        print(f"  {glabel}   node_share_B={nsB:.3f}")
        print(f"{'='*100}")
        hdr = f"  {'Method':<22} {'revenue':>9} {'rho_A':>7} {'rho_B':>7} {'min_rho':>8} {'gap':>7} {'pr_BA':>7} {'sub_B':>7} {'free_B':>8} {'nsB':>6}"
        print(hdr)
        print(f"  {'-'*22} {'-'*9} {'-'*7} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*6}")
        for method in METHOD_ORDER:
            if method not in methods_data:
                print(f"  {method:<22} {'(not run)':>9}")
                continue
            agg = methods_data[method]
            print(f"  {method:<22} "
                  f"{fm_val(agg,'revenue'):>9} "
                  f"{fm_val(agg,'rho_A'):>7} "
                  f"{fm_val(agg,'rho_B'):>7} "
                  f"{fm_val(agg,'min_rho'):>8} "
                  f"{fm_val(agg,'gap'):>7} "
                  f"{fm_val(agg,'price_ratio_BA'):>7} "
                  f"{fm_val(agg,'sub_share_B'):>7} "
                  f"{fm_val(agg,'free_share_B'):>8} "
                  f"{nsB:.3f}")

    return audit


def main():
    cfg = OmegaConf.create(BASE_CFG_YAML)

    # ── Rice-FB ───────────────────────────────────────────────────────────
    print("\n[T2] Loading Rice-FB...")
    G_rice, lbl_rice = load_rice_facebook_with_labels()
    nsB_rice = float((lbl_rice == 1).sum()) / len(lbl_rice)
    cfg_rice = OmegaConf.merge(cfg, OmegaConf.create({
        "graph": {"n_nodes": G_rice.number_of_nodes()},
        "budget": {"k": G_rice.number_of_nodes()},
        "env": {"k": G_rice.number_of_nodes()},
    }))

    audit_json = json.load(open("results/logs/fairness_audit.json"))
    for name, (ckpt, mtype) in CKPTS.items():
        print(f"  {name} on Rice-FB...", flush=True)
        agg = run_gnn_on_graph(G_rice, lbl_rice, CHECKPOINTS_K, cfg_rice, name, ckpt, mtype)
        audit_json["rice_fb"]["methods"][name] = agg

    # ── SBM graphs ────────────────────────────────────────────────────────
    for h in [0.5, 0.7, 0.9]:
        gname = f"SBM_h{h:.1f}"
        print(f"  GNN on {gname}...", flush=True)
        G_sbm, lbl_sbm = two_block_graph(n=1000, frac_minority=0.3,
                                          avg_degree=5.0, homophily=h, seed=0)
        cfg_sbm = OmegaConf.merge(cfg, OmegaConf.create({
            "graph": {"n_nodes": 1000}, "budget": {"k": 1000}, "env": {"k": 1000}
        }))
        for name, (ckpt, mtype) in CKPTS.items():
            agg = run_gnn_on_graph(G_sbm, lbl_sbm, CHECKPOINTS_SBM, cfg_sbm, name, ckpt, mtype)
            audit_json[gname]["methods"][name] = agg

    # ── Save updated audit ────────────────────────────────────────────────
    with open("results/logs/fairness_audit.json", "w") as f:
        json.dump(audit_json, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print("\nUpdated results/logs/fairness_audit.json")

    # ── Print full table (Task 2) ─────────────────────────────────────────
    print("\n\n" + "="*100)
    print("  TASK 2 — COMPLETE FAIRNESS AUDIT TABLE")
    print("  Note: IE-Strategy SKIPPED (O(n²×MC) per trial, est >30 min); row shows (not run)")
    print("="*100)
    merge_and_print("results/logs/fairness_audit.json", "results/logs/fairness_f1.json")


if __name__ == "__main__":
    main()
