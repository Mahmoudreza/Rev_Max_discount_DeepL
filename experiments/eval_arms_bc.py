#!/usr/bin/env python3
"""eval_arms_bc.py — Evaluation for Arms B and C (and baselines).

Protocol: influence_model=monotone, reward_type=flat, Uniform(0,2), n_mc_samples=200.
GREEDY action selection (argmax, not sampling).

Tables produced:
  1. Unconstrained rev mean±std: Greedy-Discount, CGS, LSTM-existing,
     Phase-1-only, Arm B, Arm C  ×  FF-1000, FF-2000, Modular-FF, Rice-FB
  2. Budget eval kappa in {5,20}: profit, below-cost, |S_T|, revenue
  3. Ordering ablation: paired delta + 95% CI vs degree order
  4. Mean degree of first 50 buyers per arm

Usage:
  python eval_arms_bc.py                    # uses latest b/c checkpoints
  python eval_arms_bc.py --ckpt_b X.pt --ckpt_c Y.pt
  python eval_arms_bc.py --skip_arms        # baselines only
"""
import argparse, json, os, sys, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from scipy import stats as scipy_stats

from src.env.revenue_env import RevenueEnv, RevenueEnvConfig
from src.env.graph_generators import generate_forest_fire
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.utils.features import (compute_static_features, build_graph_feature_cache,
                                 compute_node_features_fast)
from src.utils.helpers import set_seed
from src.evaluation.baselines import greedy_discount_trajectory

_ROOT = str(Path(__file__).parent.parent)
CKPT  = os.path.join(_ROOT, "results", "checkpoints")
OUT   = os.path.join(_ROOT, "results", "logs", "eval_arms_bc.json")

SEEDS    = list(range(10))
MC       = 200          # paper protocol
HID      = 64
C_PROD   = 0.3

# ── model helpers ─────────────────────────────────────────────────────────────

def _make_pol():
    enc  = GraphSAGEEncoder(in_dim=20, hidden_dim=HID, n_layers=2)
    lstm = EpisodeLSTM(graph_dim=HID, lstm_hidden=HID, n_layers=1)
    return SequentialJointPolicy(enc, lstm, gnn_dim=HID, context_dim=HID)

def load_pol(path, device):
    pol = _make_pol().to(device)
    ckpt = torch.load(path, map_location=device)
    # Try common key names, then assume raw state dict
    for key in ('policy_state_dict', 'model_state_dict', 'state_dict'):
        if isinstance(ckpt, dict) and key in ckpt:
            pol.load_state_dict(ckpt[key]); break
    else:
        pol.load_state_dict(ckpt)   # raw state dict
    pol.eval()
    return pol

def _ei(G):
    m = {v: i for i, v in enumerate(G.nodes())}; E = list(G.edges())
    s = [m[u] for u,_ in E]+[m[v] for _,v in E]
    d = [m[v] for _,v in E]+[m[u] for u,_ in E]
    return torch.tensor([s, d], dtype=torch.long)

# ── network loaders ───────────────────────────────────────────────────────────

def _load_rice():
    from src.env.graph_generators import load_rice_facebook
    return load_rice_facebook()

def _load_modular_ff(seed=0):
    """Two-block modular: half forest-fire, half Erdos-Renyi, one cross-edge."""
    try:
        from src.env.graph_generators import generate_modular_ff
        return generate_modular_ff(1000, seed=seed)
    except Exception:
        G1 = generate_forest_fire(500, 0.37, 0.32, seed=seed)
        G2 = nx.erdos_renyi_graph(500, 0.006, seed=seed+1)
        G  = nx.disjoint_union(G1, G2)
        G.add_edge(0, 500)
        return nx.convert_node_labels_to_integers(G)

NETWORKS = {
    "FF-1000":    lambda s: generate_forest_fire(1000, 0.37, 0.32, seed=s),
    "FF-2000":    lambda s: generate_forest_fire(2000, 0.37, 0.32, seed=s),
    "Modular-FF": lambda s: _load_modular_ff(seed=s),
    "Rice-FB":    lambda s: _load_rice(),   # fixed graph, seed ignored
}

# ── episode runners ────────────────────────────────────────────────────────────

def _env(G, seed):
    cfg = RevenueEnvConfig(influence_model="monotone", b=1.0,
                           weight_low=0.0, weight_high=2.0,
                           n_mc_samples=MC, reward_type="flat",
                           gamma=1.0, seed=seed)
    return RevenueEnv(G, cfg)

def run_pol_greedy(pol, G, seed, device, force_degree_order=False):
    """GREEDY (argmax) episode. Returns (revenue, list_of_chosen_nodes_in_order)."""
    set_seed(seed)
    env = _env(G, seed); env.reset()
    nodes = list(G.nodes()); n = len(nodes)
    ei = _ei(G).to(device)
    cache = build_graph_feature_cache(G, compute_static_features(G))
    degs = dict(G.degree())
    pol.reset_episode(device)
    total = 0.0; chosen = []

    for _ in range(n):
        avail = [v for v in nodes if v not in env.offered]
        if not avail: break
        feats = compute_node_features_fast(cache, env.S, set(env.offered), env.t, n, env)
        x  = torch.tensor(feats, dtype=torch.float32, device=device)
        av = torch.tensor([v not in env.offered for v in nodes], dtype=torch.bool, device=device)
        # Single forward per step — LSTM advances once; h used for both selection + pricing
        with torch.no_grad():
            ms, h, ctx, _ = pol.forward(x, ei, av)
            if force_degree_order:
                sel_node = max(avail, key=lambda v: degs[v])
                sel = nodes.index(sel_node)
            else:
                safe = ms.clone(); safe[~av] = -1e9
                sel = int(safe.argmax())     # GREEDY argmax
            disc = float(pol.get_discount_distribution(
                torch.cat([h[sel], ctx])).mean.clamp(1e-4, 1-1e-4).detach())

        v = nodes[sel]; chosen.append(v)
        _, r, done, _ = env.step(env.node_to_idx[v], disc)
        total += r
        if done: break

    return total, chosen

def run_greedy_discount(G, seed):
    """Greedy-Discount baseline revenue."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({"influence":{"model":"monotone","b":1.0,"weight_low":0.0,
                             "weight_high":2.0,"n_mc_samples":MC},
                             "reward":{"type":"flat","gamma":1.0},"budget":{"k":50},
                             "project":{"seed":seed}})
    set_seed(seed)
    traj = greedy_discount_trajectory(G, cfg)
    return float(sum(r for _,r,_,_ in traj) if traj else 0.0)

# ── budget eval ───────────────────────────────────────────────────────────────

def run_pol_budget(pol, G, seed, kappa, device):
    """Run with budget B=kappa*C. Returns (revenue, profit, n_below_cost, |S_T|)."""
    from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetRevenueEnvConfig
    B = kappa * C_PROD
    bcfg = BudgetRevenueEnvConfig(influence_model="monotone", b=1.0,
                                  weight_low=0.0, weight_high=2.0,
                                  n_mc_samples=MC, reward_type="flat",
                                  gamma=1.0, seed=seed, c=C_PROD, B=B)
    set_seed(seed)
    env = BudgetRevenueEnv(G, bcfg); env.reset()
    nodes = list(G.nodes()); n = len(nodes)
    ei = _ei(G).to(device)
    cache = build_graph_feature_cache(G, compute_static_features(G))
    pol.reset_episode(device)
    revenue = 0.0; n_below = 0; prices = []

    for _ in range(n):
        avail = [v for v in nodes if v not in env.offered]
        if not avail or getattr(env, '_check_bankrupt', lambda: False)(): break
        feats = compute_node_features_fast(cache, env.S, set(env.offered), env.t, n, env)
        x  = torch.tensor(feats, dtype=torch.float32, device=device)
        av = torch.tensor([v not in env.offered for v in nodes], dtype=torch.bool, device=device)
        with torch.no_grad():
            ms, h, ctx, _ = pol.forward(x, ei, av)
            safe = ms.clone(); safe[~av] = -1e9; sel = int(safe.argmax())
            disc = float(pol.get_discount_distribution(
                torch.cat([h[sel], ctx])).mean.clamp(1e-4, 1-1e-4).detach())
        v = nodes[sel]
        _, r, done, info = env.step(env.node_to_idx[v], disc)
        if r > 0:
            p = float(info.get('offered_price', r))
            prices.append(p)
            if p < C_PROD: n_below += 1
        revenue += r
        if done: break

    S_T = len(env.S)
    profit = revenue - C_PROD * S_T
    return revenue, profit, n_below, S_T

# ── CGS (calibrated-greedy) baseline ─────────────────────────────────────────

def run_cgs(G, seed):
    try:
        from _cal_episode_utils import calibrate, BudgetEnvConfig as BECfg, _sel_episode, C
        from src.evaluation.budget_baselines import _make_env
        cal_cfg = BECfg(weight_low=0.0, weight_high=2.0, n_mc_samples=5)
        cal = calibrate(G, cal_cfg)
        V, A, _, cb, ib = cal
        env = _make_env(G, B=1e7, c=C, seed=seed); env.reset()
        _sel_episode(3, 1.0, env, G, A, V, cb, ib)
        return float(getattr(env, 'total_revenue', sum(
            env._estimate_valuation(v) for v in env.S)))
    except Exception as e:
        return float('nan')

# ── existing LSTM checkpoint ──────────────────────────────────────────────────

def _find_existing_lstm():
    for name in ["rev_gnn_lstm_unified.pt", "arm_b_best.pt", "c1_b_s0_ep0030.pt"]:
        p = os.path.join(CKPT, name)
        if os.path.exists(p): return p
    return None

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_b",  default="", help="Arm B checkpoint")
    ap.add_argument("--ckpt_c",  default="", help="Arm C checkpoint")
    ap.add_argument("--ckpt_p1", default=os.path.join(CKPT, "c1_p1_s1_ep0200.pt"))
    ap.add_argument("--device",  default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip_arms", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device}  MC={MC}", flush=True)

    # Auto-find latest arm checkpoints
    def _latest(arm):
        import glob
        pts = sorted(glob.glob(os.path.join(CKPT, f"c1_{arm}_s*_ep*.pt")))
        return pts[-1] if pts else ""

    ckpt_b  = args.ckpt_b  or _latest("b")
    ckpt_c  = args.ckpt_c  or _latest("c")
    ckpt_p1 = args.ckpt_p1
    ckpt_ex = _find_existing_lstm()

    print(f"Phase-1: {ckpt_p1}")
    print(f"Arm B:   {ckpt_b or 'NOT FOUND'}")
    print(f"Arm C:   {ckpt_c or 'NOT FOUND'}")
    print(f"Exist.:  {ckpt_ex or 'NOT FOUND'}", flush=True)

    pols = {}
    pols["Phase-1"] = load_pol(ckpt_p1, device)
    if ckpt_b and not args.skip_arms: pols["Arm-B"] = load_pol(ckpt_b, device)
    if ckpt_c and not args.skip_arms: pols["Arm-C"] = load_pol(ckpt_c, device)
    if ckpt_ex: pols["LSTM-existing"] = load_pol(ckpt_ex, device)

    all_results = {}

    # ── Table 1: Unconstrained ────────────────────────────────────────────────
    print("\n=== TABLE 1: UNCONSTRAINED ===", flush=True)
    hdr = f"{'network':12s} {'method':16s}  {'mean':7s} {'std':6s}  seeds"
    print(hdr)
    for net_name, net_fn in NETWORKS.items():
        net_res = {}
        for seed in SEEDS:
            G = net_fn(seed)
            net_res.setdefault("Greedy-Discount", []).append(run_greedy_discount(G, seed))
            net_res.setdefault("CGS", []).append(run_cgs(G, seed))
            for m, pol in pols.items():
                r, _ = run_pol_greedy(pol, G, seed, device)
                net_res.setdefault(m, []).append(r)

        all_results[net_name] = {}
        for m, vals in net_res.items():
            mean_v = float(np.mean(vals)); std_v = float(np.std(vals))
            all_results[net_name][m] = {"mean": round(mean_v,2), "std": round(std_v,2),
                                        "vals": [round(v,2) for v in vals]}
            print(f"  {net_name:12s} {m:16s}  {mean_v:7.2f} {std_v:6.2f}", flush=True)

    # ── Table 2: Budget ───────────────────────────────────────────────────────
    print("\n=== TABLE 2: BUDGET ===", flush=True)
    budget_results = {}
    G_ff = NETWORKS["FF-1000"](0)   # single graph, vary seed for env stochasticity
    for kappa in [5, 20]:
        budget_results[str(kappa)] = {}
        print(f"\n  kappa={kappa}  B={kappa*C_PROD:.2f}")
        for m, pol in pols.items():
            revs=[]; profits=[]; n_belows=[]; s_ts=[]
            for seed in SEEDS:
                rev, profit, n_below, s_t = run_pol_budget(pol, G_ff, seed, kappa, device)
                revs.append(rev); profits.append(profit)
                n_belows.append(n_below); s_ts.append(s_t)
            budget_results[str(kappa)][m] = {
                "profit":     round(float(np.mean(profits)),2),
                "revenue":    round(float(np.mean(revs)),2),
                "n_below_cost": round(float(np.mean(n_belows)),1),
                "S_T":        round(float(np.mean(s_ts)),1),
            }
            print(f"    {m:16s}  profit={np.mean(profits):7.2f}  rev={np.mean(revs):7.2f}"
                  f"  below_cost={np.mean(n_belows):5.1f}  |S|={np.mean(s_ts):5.1f}", flush=True)

    # ── Ordering Ablation ─────────────────────────────────────────────────────
    print("\n=== ORDERING ABLATION (FF-1000) ===", flush=True)
    ablation = {}
    for m, pol in pols.items():
        if m in ("Greedy-Discount", "CGS"): continue
        native=[]; forced=[]
        G = NETWORKS["FF-1000"](0)
        for seed in SEEDS:
            r_n, ch_n = run_pol_greedy(pol, G, seed, device, force_degree_order=False)
            r_f, _    = run_pol_greedy(pol, G, seed, device, force_degree_order=True)
            native.append(r_n); forced.append(r_f)
        diff = [n-f for n,f in zip(native, forced)]
        mean_d = float(np.mean(diff)); std_d = float(np.std(diff, ddof=1))
        t, p = scipy_stats.ttest_1samp(diff, 0.0)
        ci95 = scipy_stats.t.ppf(0.975, len(diff)-1) * std_d / np.sqrt(len(diff))
        ablation[m] = {"native_mean": round(float(np.mean(native)),2),
                       "forced_mean": round(float(np.mean(forced)),2),
                       "delta_mean": round(mean_d,2), "delta_ci95": round(ci95,2),
                       "p": round(p,4)}
        flag = "LARGE_DELTA" if abs(mean_d) > 10 else "small_delta"
        print(f"  {m:16s}  native={np.mean(native):.2f}  forced={np.mean(forced):.2f}"
              f"  delta={mean_d:+.2f} ±{ci95:.2f}  p={p:.4f}  [{flag}]", flush=True)

    # ── Mean degree first 50 buyers ───────────────────────────────────────────
    print("\n=== MEAN DEGREE FIRST 50 BUYERS (FF-1000, seed=0) ===", flush=True)
    G = NETWORKS["FF-1000"](0); degs = dict(G.degree())
    deg50 = {}
    for m, pol in pols.items():
        if m in ("Greedy-Discount", "CGS"): continue
        _, chosen = run_pol_greedy(pol, G, 0, device)
        d50 = float(np.mean([degs[v] for v in chosen[:50]])) if len(chosen) >= 50 else 0.0
        deg50[m] = round(d50, 2)
        print(f"  {m:16s}  mean_deg_first50={d50:.2f}", flush=True)
    # degree-order baseline
    nodes_by_deg = sorted(G.nodes(), key=lambda v: -degs[v])
    d50_ctrl = float(np.mean([degs[v] for v in nodes_by_deg[:50]]))
    deg50["DegreeOrder-control"] = round(d50_ctrl, 2)
    print(f"  {'DegreeOrder-ctrl':16s}  mean_deg_first50={d50_ctrl:.2f}", flush=True)

    # ── Paired tests vs existing policy ──────────────────────────────────────
    if "LSTM-existing" in pols:
        print("\n=== PAIRED TESTS vs LSTM-existing (FF-1000 unconstrained) ===", flush=True)
        ref_vals = all_results["FF-1000"].get("LSTM-existing", {}).get("vals", [])
        if ref_vals:
            for m in ("Arm-B", "Arm-C", "Phase-1"):
                v = all_results["FF-1000"].get(m, {}).get("vals")
                if not v: continue
                diff = [a-b for a,b in zip(v, ref_vals)]
                t, p = scipy_stats.ttest_rel(v, ref_vals)
                ci95 = scipy_stats.t.ppf(0.975, len(diff)-1) * np.std(diff,ddof=1) / np.sqrt(len(diff))
                flag = "NOT SIG" if p >= 0.05 else ("sig+" if np.mean(diff)>0 else "sig-")
                print(f"  {m:10s} vs LSTM-ex: diff={np.mean(diff):+.2f} ±{ci95:.2f}"
                      f"  p={p:.4f}  {flag}", flush=True)

    # ── Save + commit ─────────────────────────────────────────────────────────
    results = {"unconstrained": all_results, "budget": budget_results,
               "ablation": ablation, "deg50": deg50,
               "checkpoints": {"p1": ckpt_p1, "b": ckpt_b, "c": ckpt_c}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nSaved → {OUT}")
    subprocess.run(["git", "add", "-f", OUT], cwd=_ROOT)
    subprocess.run(["git", "commit", "-m", "eval_arms_bc.json"], cwd=_ROOT)
    h = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True, cwd=_ROOT).stdout.strip()
    print(h)


if __name__ == "__main__":
    main()
