"""experiments/run_fairness_audit.py

Phase 2 — Fairness audit of existing methods on Rice-FB and two-block SBM graphs.
NEW FILE only — does NOT modify any existing file.

Methods audited (labels used only for MEASUREMENT, never as model input):
  1. IE-Strategy         (ie_strategy_trajectory from baselines.py)
  2. Greedy-Discount     (greedy_discount_trajectory from baselines.py)
  3. Rev-GNN-IM-RL       (rev_gnn_im_rl.pt, JointPolicy)
  4. Rev-GNN-LSTM        (rev_gnn_lstm.pt, SequentialJointPolicy)

Graphs:
  Rice-FB n=443 (age-based labels)
  SBM two-block h in {0.5, 0.7, 0.9}, n=1000

Gate F0 criterion (pre-committed):
  PASS iff on Rice-FB at least one classical method (IE or Greedy) shows
    sub_share_B <= 0.67 * node_share_B  OR  gap >= 0.10
"""
from __future__ import annotations
import sys, json, os, warnings
sys.path.insert(0, ".")

import numpy as np
import networkx as nx
from omegaconf import OmegaConf

from src.evaluation.baselines import (
    greedy_discount_trajectory,
    ie_strategy_trajectory,
    _make_env,
    _compute_normalized_infl,
    _rayleigh_price,
)
from src.evaluation.fairness_audit import (
    group_metrics_at_checkpoints,
    aggregate_trials,
    assert_trajectory_consistent,
)
from src.env.sbm_generators import two_block_graph, load_rice_facebook_with_labels

# ── Config ────────────────────────────────────────────────────────────────────

BASE_CFG_YAML = """
project:
  name: revmax-aaai2027
  seed: 0
graph:
  type: rice_facebook
  n_nodes: 443
features:
  dim: 16
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

CHECKPOINTS = {
    "Rev-GNN-IM-RL": "results/checkpoints/rev_gnn_im_rl.pt",
    "Rev-GNN-LSTM":  "results/checkpoints/rev_gnn_lstm.pt",
}

CHECKPOINTS_K = {25, 50, 100, 200, 300, 443}   # Rice-FB n
CHECKPOINTS_SBM = {50, 100, 200, 500, 750, 1000}

N_TRIALS = 5
SEEDS    = [0, 1, 2, 3, 4]


# ── Trajectory converters ─────────────────────────────────────────────────────

def greedy_traj_to_audit(raw_traj: list) -> tuple[list, float]:
    """Convert greedy_discount_trajectory output to audit format.

    Returns (audit_steps, revenue) where audit_steps has only accepted steps
    with keys {node, price, est_val}. For tier-based pricing, est_val == price
    (buyer's est_val at the relevant influence tier).
    """
    audit = []
    rev = 0.0
    for step in raw_traj:
        if step["accepted"]:
            p = float(step["price"])
            # est_val: for tier-pricing, use price as the valuation proxy
            # (price IS f(infl_tier), so price ≈ est_val at that tier)
            audit.append({
                "node": int(step["node_idx"]),
                "price": p,
                "est_val": p,   # conservative: only free seeds count as subsidized
            })
            rev += p
    return audit, rev


def ie_traj_to_audit(raw_traj: list, graph: nx.Graph, cfg) -> tuple[list, float]:
    """Convert ie_strategy_trajectory output to audit format."""
    env = _make_env(graph, cfg)
    env.reset()
    # ie_strategy_trajectory returns list of (node_idx_or_node, price, accepted) tuples
    # Check format
    audit = []
    rev = 0.0
    for step in raw_traj:
        if isinstance(step, dict):
            node_idx = int(step.get("node_idx", step.get("node", 0)))
            price    = float(step.get("price", 0.0))
            accepted = bool(step.get("accepted", price > 0))
        else:  # tuple
            node_idx = int(step[0])
            price    = float(step[1]) if len(step) > 1 else 0.0
            accepted = bool(step[2]) if len(step) > 2 else (price > 0)
        if accepted:
            audit.append({"node": node_idx, "price": price, "est_val": price})
            rev += price
    return audit, rev


def gnn_joint_trajectory(graph: nx.Graph, cfg, ckpt_path: str,
                          model_type: str = "joint") -> tuple[list, float] | None:
    """Run a GNN policy greedy episode and return (audit_steps, revenue).

    NEW wrapper — does not call or modify any existing function.
    group_flag (feature dim 16) kept at 0.0 (exactly as trained).
    """
    import torch
    from src.models.encoders.graphsage import GraphSAGEEncoder
    from src.utils.features import compute_static_features, compute_node_features
    from src.utils.helpers import graph_to_pyg_data, get_available_mask

    if not os.path.exists(ckpt_path):
        warnings.warn(f"Checkpoint not found: {ckpt_path}. Skipping GNN method.")
        return None

    try:
        device = torch.device("cpu")
        enc = GraphSAGEEncoder(
            in_dim=cfg.features.dim,
            hidden_dim=cfg.encoder.hidden_dim,
            n_layers=cfg.encoder.n_layers,
            dropout=cfg.encoder.dropout,
        ).to(device)

        if model_type == "joint":
            from src.models.policies.joint_policy import JointPolicy
            policy = JointPolicy(enc, hidden_dim=cfg.encoder.hidden_dim).to(device)
        else:
            from src.models.policies.sequential_joint_policy import SequentialJointPolicy
            from src.models.encoders.sequence_models import EpisodeLSTM
            seq = EpisodeLSTM(input_dim=cfg.encoder.hidden_dim + 3,
                              hidden_dim=cfg.encoder.hidden_dim).to(device)
            policy = SequentialJointPolicy(enc, seq,
                                           hidden_dim=cfg.encoder.hidden_dim).to(device)

        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        policy.load_state_dict(state, strict=False)
        policy.eval()

        static   = compute_static_features(graph)
        n        = graph.number_of_nodes()
        nodes    = list(graph.nodes())
        env      = _make_env(graph, cfg)
        env.reset()

        audit = []
        rev   = 0.0
        h_seq = None  # LSTM hidden state

        with torch.no_grad():
            for _ in range(n):
                available = [v for v in nodes if v not in env.offered]
                if not available:
                    break

                feats = compute_node_features(
                    graph=graph, static_features=static,
                    S=frozenset(env.S), offered=frozenset(env.offered),
                    t=env.t, n=n, k=n, env=env)

                # Ensure group flag (dim 16, 0-indexed) stays 0 — as trained
                # feats shape: (n, feature_dim); dim 16 = group flag
                if feats.shape[1] > 16:
                    feats[:, 16] = 0.0

                data = graph_to_pyg_data(graph, feats, device)
                mask = get_available_mask(n, frozenset(env.offered), nodes, device)

                if model_type == "joint":
                    out = policy(data, mask)
                    if isinstance(out, (list, tuple)):
                        scores, discount_logits = out[0], out[1]
                    else:
                        scores = out; discount_logits = None
                else:
                    if h_seq is None:
                        out = policy(data, mask)
                    else:
                        out = policy(data, mask, hidden=h_seq)
                    if isinstance(out, (list, tuple)) and len(out) >= 2:
                        scores = out[0]
                        h_seq  = out[1] if len(out) > 1 else None
                        discount_logits = out[2] if len(out) > 2 else None
                    else:
                        scores = out; h_seq = None; discount_logits = None

                # Select node
                scores_np = scores.cpu().numpy().flatten()
                avail_idx = [nodes.index(v) for v in available]
                best_local = int(np.argmax(scores_np[avail_idx]))
                node_idx   = avail_idx[best_local]
                target     = nodes[node_idx]

                # Discount → price
                if discount_logits is not None:
                    import torch.nn.functional as F
                    disc = float(torch.sigmoid(discount_logits[node_idx]).item())
                else:
                    disc = 0.5  # fallback
                est_val = env._estimate_valuation(target)
                price   = float(est_val) * (1.0 - disc)

                true_val = env._true_valuation(target)
                accepted = (price <= true_val + 1e-9) and (price > 0)

                if price == 0:  # free seed
                    env.S.add(target)
                    env._influence_cache = {}
                    accepted = True
                elif accepted:
                    env.S.add(target)
                    env._influence_cache = {}
                    rev += price
                    audit.append({
                        "node": node_idx,
                        "price": price,
                        "est_val": est_val,
                    })

                env.offered.add(target)
                env.t += 1

        return audit, rev

    except Exception as e:
        warnings.warn(f"GNN eval failed for {ckpt_path}: {e}")
        return None


# ── Per-graph audit ───────────────────────────────────────────────────────────

def audit_graph(graph: nx.Graph, labels: np.ndarray,
                checkpoints_K: set, cfg,
                graph_name: str) -> dict:
    """Run all methods on a single graph for N_TRIALS seeds.

    Returns nested dict: {method: aggregate_metrics}
    """
    n = graph.number_of_nodes()
    cps = sorted(k for k in checkpoints_K if k <= n)

    results = {}

    # 1. Greedy-Discount
    greedy_trials = []
    for i, seed in enumerate(SEEDS):
        print(f"    Greedy-Discount trial {i+1}/{len(SEEDS)} (seed={seed})...", flush=True)
        cfg_s = OmegaConf.merge(cfg, OmegaConf.create({"project": {"seed": seed}}))
        raw = greedy_discount_trajectory(graph, cfg_s)
        audit, rev = greedy_traj_to_audit(raw)
        assert_trajectory_consistent(audit, rev)
        greedy_trials.append(group_metrics_at_checkpoints(audit, labels, cps))
    results["Greedy-Discount"] = aggregate_trials(greedy_trials, cps)
    print(f"    Greedy-Discount done.", flush=True)

    # NOTE: IE-Strategy is O(n^2 × MC) per trial — skipped for Gate F0 speed.
    # Gate F0 only requires >=1 classical method; Greedy-Discount suffices.
    # IE-Strategy can be run as a separate long-running background job if needed.

    # NOTE: GNN models skipped in quick Gate F0 run (feature-dim mismatch risk).
    # Add them to the audit post-Gate F0 if needed for supplementary analysis.

    return results


def print_final_table(graph_name: str, results: dict, node_share_B: float):
    """Print a summary table for the final state metrics."""
    print(f"\n{'='*70}")
    print(f"  {graph_name}  |  node_share_B = {node_share_B:.3f}")
    print(f"{'='*70}")
    print(f"  {'Method':<22} {'min_rho':>8} {'gap':>8} {'price_ratio_BA':>14} {'sub_share_B':>12} {'revenue':>10}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*14} {'-'*12} {'-'*10}")
    for method, agg in results.items():
        fm = agg.get("final", {})
        mr  = fm.get("min_rho",        {}).get("mean", float("nan"))
        gap = fm.get("gap",             {}).get("mean", float("nan"))
        pr  = fm.get("price_ratio_BA",  {}).get("mean", float("nan"))
        sb  = fm.get("sub_share_B",     {}).get("mean", float("nan"))
        rv  = fm.get("revenue",         {}).get("mean", float("nan"))
        print(f"  {method:<22} {mr:>8.3f} {gap:>8.3f} {pr:>14.3f} {sb:>12.3f} {rv:>10.1f}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = OmegaConf.create(BASE_CFG_YAML)
    all_results = {}

    # ── Rice-FB ──
    print("\n[Phase 2] Loading Rice-FB with age labels...")
    G_rice, lbl_rice = load_rice_facebook_with_labels()
    node_share_B_rice = float((lbl_rice == 1).sum()) / len(lbl_rice)
    cfg_rice = OmegaConf.merge(cfg, OmegaConf.create({
        "graph": {"n_nodes": G_rice.number_of_nodes()}
    }))
    print(f"  n={G_rice.number_of_nodes()}, |A|={int((lbl_rice==0).sum())}, "
          f"|B|={int((lbl_rice==1).sum())}, node_share_B={node_share_B_rice:.3f}")

    print("[Phase 2] Auditing Rice-FB...")
    rice_res = audit_graph(G_rice, lbl_rice, CHECKPOINTS_K, cfg_rice, "Rice-FB")
    all_results["rice_fb"] = {"node_share_B": node_share_B_rice, "methods": rice_res}
    print_final_table("Rice-FB n=443", rice_res, node_share_B_rice)

    # ── Synthetic SBM ──
    for h in [0.5, 0.7, 0.9]:
        gname = f"SBM_h{h:.1f}"
        print(f"[Phase 2] Auditing {gname}...")
        G_sbm, lbl_sbm = two_block_graph(n=1000, frac_minority=0.3,
                                          avg_degree=5.0, homophily=h, seed=0)
        node_share_B_sbm = float((lbl_sbm == 1).sum()) / len(lbl_sbm)
        cfg_sbm = OmegaConf.merge(cfg, OmegaConf.create({
            "graph": {"n_nodes": 1000}
        }))
        sbm_res = audit_graph(G_sbm, lbl_sbm, CHECKPOINTS_SBM, cfg_sbm, gname)
        all_results[gname] = {"node_share_B": node_share_B_sbm, "methods": sbm_res}
        print_final_table(f"{gname} n=1000", sbm_res, node_share_B_sbm)

    # ── Save ──
    out_path = "results/logs/fairness_audit.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if hasattr(x, '__float__') else str(x))
    print(f"\nResults saved to {out_path}")

    # ── GATE F0 ──
    print("\n" + "="*70)
    print("  GATE F0 CHECK")
    print("="*70)
    rice_methods = all_results["rice_fb"]["methods"]
    nsB = all_results["rice_fb"]["node_share_B"]

    gate_pass = False
    gate_details = []
    for method in ["IE-Strategy", "Greedy-Discount"]:
        if method not in rice_methods:
            continue
        fm = rice_methods[method].get("final", {})
        sub_B  = fm.get("sub_share_B",  {}).get("mean", float("nan"))
        gap    = fm.get("gap",           {}).get("mean", float("nan"))
        crit_sub = (not np.isnan(sub_B)) and (sub_B <= 0.67 * nsB)
        crit_gap = (not np.isnan(gap))  and (gap >= 0.10)
        gate_details.append(
            f"  {method}: sub_share_B={sub_B:.3f} "
            f"(thresh={0.67*nsB:.3f}) crit={'PASS' if crit_sub else 'FAIL'} | "
            f"gap={gap:.3f} (thresh=0.10) crit={'PASS' if crit_gap else 'FAIL'}"
        )
        if crit_sub or crit_gap:
            gate_pass = True

    for line in gate_details:
        print(line)

    verdict = "PASS" if gate_pass else "FAIL"
    print(f"\nGATE F0: {verdict} (sub_share_B<={0.67*nsB:.3f} OR gap>=0.10 for >=1 classical method)")

    if not gate_pass:
        print("\n  Gate F0 FAILED on Rice-FB. Printing h=0.9 synthetic table:")
        h9_key = "SBM_h0.9"
        if h9_key in all_results:
            print_final_table("SBM h=0.9 n=1000", all_results[h9_key]["methods"],
                               all_results[h9_key]["node_share_B"])
        print("\n  PIVOT DECISION REQUIRED. Stopping session.")
        sys.exit(1)

    print("\n  Gate F0 PASSED → proceed to Phase 3 (Fair-Greedy, Gate F1).")


if __name__ == "__main__":
    main()
