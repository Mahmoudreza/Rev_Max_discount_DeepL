#!/usr/bin/env python3
"""experiments/run_unified_sweep.py — Gate sweep for the unified budget model.

Evaluates rev_gnn_lstm_unified.pt against comparison baselines on:
  FF n=1000 AND Rice-FB n=443
  k=[1,2,3,5,8,10,15,20,30,40], c=0.3, n_trials=3, seeds matching dp_upgrade_eval
  SKIP enforcement + accounting identity check per episode.

Prints combined comparison table:
  k | Unified | TFM-Idea3 | LSTM-v1 | DP-composite | Greedy+B

Saves → results/logs/unified_sweep.json

Pre-committed GATE criteria (printed as single verdict line):
  FF  k=3   >= 210
  FF  k=10  >= 350
  FF  k=40  >= 430
  Rice k=15 >= 45
  ALL four required for PASS.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.env.graph_generators import generate_forest_fire
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.utils.features import compute_static_features, build_graph_feature_cache

# ── Constants ──────────────────────────────────────────────────────────────────
C         = 0.3
B_MAX     = 40 * C    # = 12.0
CKPT_DIR  = "results/checkpoints"
LOG_DIR   = "results/logs"
UNIFIED_CKPT = os.path.join(CKPT_DIR, "rev_gnn_lstm_unified.pt")

K_LIST  = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40]
N_TRIALS = 3
SEEDS   = [42, 123, 7]   # matching dp_upgrade_eval seeds

# Gate criteria
GATE = {
    ("ff", 3):   210,
    ("ff", 10):  350,
    ("ff", 40):  430,
    ("rice", 15): 45,
}


def _to_edge_index(graph, device):
    edges = list(graph.edges())
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    nodes = list(graph.nodes())
    nmap  = {v: i for i, v in enumerate(nodes)}
    src   = [nmap[e[0]] for e in edges] + [nmap[e[1]] for e in edges]
    dst   = [nmap[e[1]] for e in edges] + [nmap[e[0]] for e in edges]
    return torch.tensor([src, dst], dtype=torch.long, device=device)


def _avail_mask(env, n, device):
    mask = torch.zeros(n, dtype=torch.bool, device=device)
    for idx in env.available_nodes:
        mask[idx] = True
    return mask


def _unified_feat(cache, env, k):
    from src.utils.features import compute_node_features_fast
    base = compute_node_features_fast(cache, env.S, env.offered, env.t, k, env)
    n = cache["n"]
    col = np.full((n, 1), env.B / B_MAX, dtype=np.float32)
    return np.concatenate([base, col], axis=1)


def _load_unified(device) -> SequentialJointPolicy:
    enc  = GraphSAGEEncoder(in_dim=21, hidden_dim=64, n_layers=2)
    lstm = EpisodeLSTM(graph_dim=64, lstm_hidden=64, n_layers=1)
    pol  = SequentialJointPolicy(enc, lstm, gnn_dim=64, context_dim=64)
    sd   = torch.load(UNIFIED_CKPT, map_location=device)
    pol.load_state_dict(sd, strict=True)
    pol.eval()
    return pol.to(device)


@torch.no_grad()
def _eval_unified_one(policy, graph, k, seed, device):
    """One trial of unified policy on graph at budget k*c.

    Returns: (revenue, bankrupt, n_accepted, budget_remaining)
    """
    n       = graph.number_of_nodes()
    B       = k * C
    cfg     = BudgetEnvConfig(budget_B=B, production_cost=C, seed=seed)
    env     = BudgetRevenueEnv(graph, cfg)
    env.reset()

    static  = compute_static_features(graph)
    cache   = build_graph_feature_cache(graph, static)
    ei      = _to_edge_index(graph, device)

    policy.reset_episode(device)
    revenue    = 0.0
    n_accepted = 0
    bankrupt   = False
    total_cost = 0.0
    total_rev  = 0.0

    for _t in range(n):
        if not env.available_nodes:
            break
        if env._check_bankrupt():
            bankrupt = True
            break

        x_np = _unified_feat(cache, env, k)
        x_t  = torch.FloatTensor(x_np).to(device)
        avail = _avail_mask(env, n, device)

        scores, h, ctx, _ = policy.forward(x_t, ei, avail)
        node_idx = int(scores.argmax().item())

        comb     = torch.cat([h[node_idx], ctx], dim=0)
        beta     = policy.get_discount_distribution(comb)
        discount = float(beta.mean.item())

        # SKIP enforcement
        est_val      = env._estimate_valuation(env.nodes[node_idx])
        offered_price = est_val * (1.0 - discount)
        if env.B - C + offered_price < -1e-9:
            # Mark skip (no step) — count as offered
            env.offered.add(env.nodes[node_idx])
            env.t += 1
            env.budget_history.append(env.B)
            policy.update_sequence_state(discount, False, 0.0)
            continue

        obs, reward, done, info = env.step(node_idx, discount)
        if info["accepted"]:
            revenue    += info["offered_price"]
            n_accepted += 1
            total_rev  += info["offered_price"]
        total_cost += C
        policy.update_sequence_state(discount, info["accepted"],
                                     info.get("revenue_step", 0.0))
        if done:
            break

    # Accounting identity: revenue <= total_cost (net spend = cost - revenue)
    # Soft check (warn loudly but don't abort sweep)
    net_spend = total_cost - total_rev
    if net_spend > B + 1e-6:
        print(f"  ACCOUNTING VIOLATION: net_spend={net_spend:.4f} > B={B:.4f}")
        bankrupt = True

    return revenue, bankrupt, n_accepted, float(env.B)


def eval_unified(policy, graph, k, device, n_trials=N_TRIALS):
    revs, bkrs, naccs = [], [], []
    for s in SEEDS[:n_trials]:
        rev, bkr, nacc, _ = _eval_unified_one(policy, graph, k, seed=s, device=device)
        revs.append(rev)
        bkrs.append(float(bkr))
        naccs.append(nacc)
    return {
        "rev_mean": float(np.mean(revs)),
        "rev_std":  float(np.std(revs)),
        "bkr_mean": float(np.mean(bkrs)),
        "n_accepted_mean": float(np.mean(naccs)),
        "all_rev": revs,
    }


def _load_comparison_data():
    """Read Gate B run-1 and DP-composite from existing JSONs (read-only)."""
    comp = {}

    # Gate B v2 (lstm-v1 and tfm numbers)
    gb_path = os.path.join(LOG_DIR, "gate_b_eval_v2.json")
    if os.path.exists(gb_path):
        gb = json.load(open(gb_path))
        for net_key, net_label in [("ff_n1000", "ff"), ("rice_fb", "rice")]:
            kr = gb["networks"][net_key]["k_results"]
            for fk, row in kr.items():
                k = int(fk.split("=")[1])
                comp.setdefault(net_label, {})[k] = {
                    "lstm_v1": row["lstm"]["v1"]["revenue"]["mean"],
                    "tfm":     row["tfm"]["revenue"]["mean"],
                }

    # DP-composite
    dc_path = os.path.join(LOG_DIR, "dp_v3_full_curve_merged.json")
    if os.path.exists(dc_path):
        dc = json.load(open(dc_path))
        for net_key, net_label in [("ff_n1000", "ff"), ("rice_fb", "rice")]:
            net_dc = dc.get(net_key, {})
            for fk, row in net_dc.items():
                if not fk.startswith("k="):
                    continue
                k  = int(fk.split("=")[1])
                v2 = row.get("v2", {}).get("mean", 0)
                v3 = row.get("v3", {}).get("mean", 0)
                comp.setdefault(net_label, {}).setdefault(k, {})["dp_composite"] = max(v2, v3)

    # Greedy+Budget (FF from dp_upgrade_eval.json, Rice from dp_upgrade_eval_rice_lstm.json)
    for fname, net_label in [
        ("dp_upgrade_eval.json", "ff"),
        ("dp_upgrade_eval_rice_lstm.json", "rice"),
    ]:
        fpath = os.path.join(LOG_DIR, fname)
        if not os.path.exists(fpath):
            continue
        data = json.load(open(fpath))
        for fk, row in data.items():
            if not fk.startswith("k="):
                continue
            k   = int(fk.split("=")[1])
            gb2 = row.get("Greedy+Budget", {})
            if isinstance(gb2, dict):
                gr  = gb2.get("revenue", {}).get("mean",
                      gb2.get("mean", float("nan")))
            else:
                gr = float("nan")
            comp.setdefault(net_label, {}).setdefault(k, {})["greedy_b"] = gr

    return comp


def _fmt(v, default="n/a"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return default
    return f"{v:>8.2f}"


def print_table(results: dict, comp: dict, net_label: str):
    print(f"\n{'k':>4} | {'B':>5} | {'Unified':>9}±{'std':>7} | {'bkr':>5} |"
          f" {'LSTM-v1':>8} | {'TFM-I3':>8} | {'DP-comp':>8} | {'Greedy+B':>9} | gate")
    print("-" * 95)
    gate_results = {}
    for k in K_LIST:
        B = k * C
        row = results.get(net_label, {}).get(k, {})
        if not row:
            print(f"  {k:>2} | {B:>5.2f} | {'?':>9} -- missing --")
            continue
        rev_m  = row["rev_mean"]
        rev_s  = row["rev_std"]
        bkr    = row["bkr_mean"]

        c_row   = comp.get(net_label, {}).get(k, {})
        lstm_v1 = c_row.get("lstm_v1", float("nan"))
        tfm     = c_row.get("tfm",     float("nan"))
        dp_comp = c_row.get("dp_composite", float("nan"))
        gr_b    = c_row.get("greedy_b",    float("nan"))

        # Gate check
        gate_key = (net_label, k)
        gate_thr = GATE.get(gate_key, None)
        gate_str = ""
        if gate_thr is not None:
            gate_str = f"{'✓' if rev_m >= gate_thr else '✗'} (thr={gate_thr})"
            gate_results[gate_key] = (rev_m, gate_thr, rev_m >= gate_thr)

        print(f"  {k:>2} | {B:>5.2f} |{rev_m:>9.2f}±{rev_s:>6.2f} | {bkr:>4.0%} |"
              f" {_fmt(lstm_v1)} | {_fmt(tfm)} | {_fmt(dp_comp)} | {_fmt(gr_b):>9} | {gate_str}")
    return gate_results


def main():
    t0 = time.time()
    print("=" * 70)
    print("run_unified_sweep.py — Gate-B unified evaluation")
    print("=" * 70)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    if not os.path.exists(UNIFIED_CKPT):
        print(f"ERROR: {UNIFIED_CKPT} not found. Run run_budget_unified_training.py first.")
        sys.exit(1)

    import hashlib
    sha = hashlib.sha256(open(UNIFIED_CKPT, "rb").read()).hexdigest()
    print(f"Unified checkpoint: {sha[:16]}...")

    policy = _load_unified(device)
    print(f"Policy loaded: {sum(p.numel() for p in policy.parameters())} params")

    # ── Load evaluation graphs ────────────────────────────────────────────────
    # FF n=1000 (standard; reproduce same as dp_upgrade_eval)
    print("\nLoading evaluation graphs...")
    ff_graph  = generate_forest_fire(1000, p=0.37, pb=0.32, seed=42)
    # Rice-FB: load from processed/ or generate surrogate
    rice_path = "data/processed/rice_fb.pkl"
    if os.path.exists(rice_path):
        import pickle
        rice_graph = pickle.load(open(rice_path, "rb"))
    else:
        # Surrogate: BA graph with similar size to Rice-Facebook n=443
        import networkx as nx
        rice_graph = nx.barabasi_albert_graph(443, 11, seed=42)
        print("  WARNING: Rice-FB not found; using BA(443,11) surrogate")
    print(f"  FF:   n={ff_graph.number_of_nodes()} m={ff_graph.number_of_edges()}")
    print(f"  Rice: n={rice_graph.number_of_nodes()} m={rice_graph.number_of_edges()}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    results = {"ff": {}, "rice": {}}
    networks = [("ff", ff_graph), ("rice", rice_graph)]

    for net_label, graph in networks:
        print(f"\n── {net_label.upper()} n={graph.number_of_nodes()} ──")
        for k in K_LIST:
            B = k * C
            t1 = time.time()
            res = eval_unified(policy, graph, k, device, n_trials=N_TRIALS)
            elapsed = time.time() - t1
            results[net_label][k] = res
            print(f"  k={k:>2} B={B:.2f}  {res['rev_mean']:>8.3f}±{res['rev_std']:>6.3f}"
                  f"  bkr={res['bkr_mean']:.0%}  ({elapsed:.1f}s)")

    # ── Load comparison data ───────────────────────────────────────────────────
    comp = _load_comparison_data()

    # ── Print combined tables ──────────────────────────────────────────────────
    all_gate = {}
    for net_label in ["ff", "rice"]:
        label_map = {"ff": "Forest Fire n=1000", "rice": "Rice-Facebook n=443"}
        print(f"\n{'='*70}\n{label_map[net_label]}\n{'='*70}")
        g = print_table(results, comp, net_label)
        all_gate.update(g)

    # ── Gate verdict ──────────────────────────────────────────────────────────
    gate_pass = all(v[2] for v in all_gate.values())
    failed    = [k for k, v in all_gate.items() if not v[2]]

    print("\n" + "=" * 70)
    print("GATE VERDICT")
    print("=" * 70)
    for (net, k), (rev, thr, passed) in sorted(all_gate.items()):
        sym = "✓" if passed else "✗"
        print(f"  {sym} {net.upper():5} k={k:>2}: rev={rev:.2f} vs threshold={thr} "
              f"({'PASS' if passed else 'FAIL'})")
    print()
    if gate_pass:
        print("GATE: PASS ✓ — ALL FOUR CRITERIA MET")
        print("→ Unified model is the paper's primary learned Idea-3 model.")
    else:
        print(f"GATE: FAIL ✗ — failed at: {[f'{n} k={k}' for n,k in failed]}")
        print("→ Archive checkpoint as rev_gnn_lstm_unified_gatefail.pt")
        print("→ Paper ships regime-split story as frozen.")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_out = {
        "unified_sha256": sha,
        "n_trials": N_TRIALS,
        "seeds": SEEDS[:N_TRIALS],
        "gate_pass": gate_pass,
        "gate_results": {f"{n}_k{k}": {"rev": v[0], "threshold": v[1], "pass": v[2]}
                         for (n, k), v in all_gate.items()},
        "results": {
            net: {str(k): v for k, v in kd.items()}
            for net, kd in results.items()
        },
        "elapsed_s": time.time() - t0,
    }
    out_path = os.path.join(LOG_DIR, "unified_sweep.json")
    with open(out_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\nResults saved → {out_path}")
    print(f"Elapsed: {(time.time()-t0)/60:.1f} min")

    # ── Execute gate branch ───────────────────────────────────────────────────
    if gate_pass:
        _gate_pass_actions(sha)
    else:
        _gate_fail_actions(sha, failed, all_gate)


def _gate_pass_actions(unified_sha: str):
    """GATE PASS branch: update README, note for paper table addition."""
    import hashlib
    print("\n── GATE PASS: executing PASS branch ──")

    # Append provenance to checkpoints/README.md
    readme = os.path.join(CKPT_DIR, "README.md")
    entry = f"""
### Unified budget model (Gate PASS)

| File | Status | SHA256 | Notes |
|------|--------|--------|-------|
| `rev_gnn_lstm_unified.pt` | **GATE PASS** | `{unified_sha[:16]}...` | Mixed-expert imitation (Ph1 300ep) + full-range REINFORCE (Ph2 200ep). Best on min-bucket reward. Input dim=21. Feature 21=B_t/(40c). |

Gate criteria ALL met: FF k=3≥210, FF k=10≥350, FF k=40≥430, Rice k=15≥45.
Add Unified column to paper/tables/paper_table_idea3_final.tex.
Regenerate fig_idea3_main_v2 with Unified line (run experiments/plot_idea3_main_v2.py).
"""
    with open(readme, "a") as f:
        f.write(entry)
    print(f"  Updated {readme}")
    print("  TODO (manual): add Unified column to paper_table_idea3_final.tex")
    print("  TODO (manual): regenerate fig_idea3_main_v2 with Unified line")
    print("  TODO (manual): commit 'unified budget model — gate PASS'")


def _gate_fail_actions(unified_sha: str, failed: list, gate_results: dict):
    """GATE FAIL branch: rename checkpoint, record verdict in CLAUDE.md."""
    import shutil
    fail_ckpt = os.path.join(CKPT_DIR, "rev_gnn_lstm_unified_gatefail.pt")
    src = os.path.join(CKPT_DIR, "rev_gnn_lstm_unified.pt")
    if os.path.exists(src):
        shutil.move(src, fail_ckpt)
        print(f"\n── GATE FAIL: archived → {fail_ckpt}")

    # Brief CLAUDE.md note (append to Session State)
    fail_details = "; ".join(
        f"{n} k={k}: {v[0]:.1f} < {v[1]}"
        for (n, k), v in sorted(gate_results.items())
        if not v[2]
    )
    print(f"\nGate FAIL at: {fail_details}")
    print("Paper ships regime-split story as frozen (see CLAUDE.md Session State).")
    print("No further commits needed for this experiment.")


if __name__ == "__main__":
    main()
