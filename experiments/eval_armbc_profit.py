#!/usr/bin/env python3
"""eval_armbc_profit.py — Budget/profit evaluation for Arms B and C.

Protocol: BudgetRevenueEnv, monotone+flat, Uniform(0,2), c=0.3,
          B0=kappa*c, SKIP-never-reprice, seeds [0..9], greedy argmax.
kappa: {5, 20}.
Networks: FF_1000, FF_2000, Modular_FF, Rice_FB, polblogs.
Methods: Arm-B, Arm-C, Phase-1, LSTM-densemix (0b549f93),
         CGS, IE+Budget, Greedy+Budget.
Metrics (per-seed, mean±std):
  profit Pi = R - c|S_T|  [primary]
  below-cost accepted offers
  |S_T|
  revenue R
Assert Pi = B_T - B0 per episode.

Usage (parallel per network):
  for NET in FF_1000 FF_2000 Modular_FF Rice_FB polblogs; do
    nohup venv/bin/python3 -u experiments/eval_armbc_profit.py \\
      --network $NET --ckpt_b c1_b_s1_ep0030.pt --ckpt_c c1_c_s1_ep0030.pt \\
      --device cuda:3 > /tmp/profit_${NET}.log 2>&1 &
  done
  # then merge: python eval_armbc_profit.py --merge
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

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.env.graph_generators import generate_forest_fire
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.utils.features import (compute_static_features, build_graph_feature_cache,
                                 compute_node_features_fast)
from src.utils.helpers import set_seed

_ROOT = str(Path(__file__).parent.parent)
CKPT  = os.path.join(_ROOT, "results", "checkpoints")
LOG   = os.path.join(_ROOT, "results", "logs")
OUT   = os.path.join(LOG, "armBC_profit.json")

HID    = 64
C      = 0.3
N_MC   = 200
W_HIGH = 2.0
SEEDS  = list(range(10))
KAPPAS = [5, 20]

EXISTING_CKPT = os.path.join(CKPT, "rev_gnn_lstm_densemix.pt")
EXISTING_SHA  = "0b549f93"

# ── CONFIG CHECK (printed before results) ─────────────────────────────────────
CONFIG_NOTE = """
=== CONFIGURATION CHECK ===
Arm C Phase-2 profit objective: used C_PROD=0.3, reward = R - 0.3*|S_T|.
Training environment: RevenueEnv (UNCONSTRAINED) — no budget enforced, no
  cost charged by the env. Arm C optimised a profit proxy without any
  budget or cost mechanics in the env. Budget eval here is a TRANSFER TEST.
Same situation as Arm A (documented in eval_arma_budget.py).
"""

# ── model helpers ──────────────────────────────────────────────────────────────

def _make_pol(in_dim=20):
    enc  = GraphSAGEEncoder(in_dim=in_dim, hidden_dim=HID, n_layers=2)
    lstm = EpisodeLSTM(graph_dim=HID, lstm_hidden=HID, n_layers=1)
    return SequentialJointPolicy(enc, lstm, gnn_dim=HID, context_dim=HID)

def load_pol(path, device):
    ckpt = torch.load(path, map_location=device)
    for key in ('policy_state_dict', 'model_state_dict', 'state_dict'):
        if isinstance(ckpt, dict) and key in ckpt:
            sd = ckpt[key]; break
    else:
        sd = ckpt
    w = sd.get('encoder.input_proj.weight',
               sd.get('encoder.layers.0.weight', None))
    in_dim = int(w.shape[1]) if w is not None else 20
    pol = _make_pol(in_dim=in_dim).to(device)
    pol.load_state_dict(sd)
    pol._in_dim = in_dim
    pol.eval()
    return pol

def _ei(G):
    m = {v: i for i, v in enumerate(G.nodes())}; E = list(G.edges())
    s = [m[u] for u,_ in E]+[m[v] for _,v in E]
    d = [m[v] for _,v in E]+[m[u] for u,_ in E]
    return torch.tensor([s, d], dtype=torch.long)

# ── network loaders ────────────────────────────────────────────────────────────

def load_graph(name):
    if name == "FF_1000":    return generate_forest_fire(1000, 0.37, 0.32, seed=0)
    if name == "FF_2000":    return generate_forest_fire(2000, 0.37, 0.32, seed=1)
    if name == "Rice_FB":
        from src.env.graph_generators import load_rice_facebook
        return load_rice_facebook()
    if name == "Modular_FF":
        try:
            from src.env.graph_generators import generate_modular_forest_fire
            return generate_modular_forest_fire(1000, seed=0)
        except Exception:
            from src.env.graph_generators import generate_modular_forest_fire as gmf
            return gmf(seed=0)
    if name == "polblogs":
        from src.env.polblogs_loader import load_polblogs
        return load_polblogs()
    raise ValueError(f"Unknown network: {name}")

# ── budget episode runner (unconstrained-trained policies) ────────────────────

def run_pol_budget(pol, G, cache, ei, B0, seed, device):
    """Run budget episode for unconstrained-trained policy (Arms B/C/Phase-1).
    Returns (revenue, profit, n_below_cost, S_T, B_remaining).
    """
    set_seed(seed)
    cfg = BudgetEnvConfig(budget_B=B0, production_cost=C, seed=seed,
                          weight_high=W_HIGH, n_mc_samples=N_MC)
    env = BudgetRevenueEnv(G, cfg); env.reset()
    nodes = list(G.nodes()); n = len(nodes)
    ei_d = ei.to(device)
    in_d = getattr(pol, '_in_dim', 20)
    pol.reset_episode(device)
    revenue = 0.0; n_below = 0; n_accepted = 0

    while getattr(env, 'available_nodes', None) is not None and \
          len(env.available_nodes) > 0 and not env._check_bankrupt():
        feats = compute_node_features_fast(cache, env.S, set(env.offered), env.t, n, env)
        f = feats[:, :in_d] if in_d <= feats.shape[1] else \
            np.concatenate([feats, np.zeros((feats.shape[0], in_d-feats.shape[1]))], axis=1)
        x  = torch.tensor(f, dtype=torch.float32, device=device)
        av = torch.tensor([v not in env.offered for v in nodes], dtype=torch.bool, device=device)
        if not av.any(): break
        with torch.no_grad():
            ms, h, ctx, _ = pol.forward(x, ei_d, av)
            safe = ms.clone(); safe[~av] = -1e9
            sel = int(safe.argmax())
            disc = float(pol.get_discount_distribution(
                torch.cat([h[sel], ctx])).mean.clamp(1e-4, 1-1e-4).detach())
        v = nodes[sel]
        _, r, done, info = env.step(env.node_to_idx[v], disc)
        if r > 0:
            n_accepted += 1
            price = float(info.get('offered_price', r))
            if price < C: n_below += 1
            revenue += r
        if done: break

    S_T    = len(env.S)
    B_rem  = float(env.B)   # env.B = remaining budget; B_T - B0 = R - c|S_T| = Pi
    profit = revenue - C * S_T
    pi_check = B_rem - B0
    assert abs(profit - pi_check) < 0.01 + 0.001 * abs(profit), \
        f"Pi identity FAIL: profit={profit:.3f} B_T-B0={pi_check:.3f}"
    return revenue, profit, n_below, S_T

# ── budget episode runner for existing LSTM (budget-trained, in_dim=21) ───────

def run_existing_budget(pol, G, cache, ei, B0, seed, device):
    """Run budget episode for budget-trained LSTM (0b549f93, in_dim=21).
    Budget fraction (env.B / B0) used as 21st feature.
    """
    set_seed(seed)
    cfg = BudgetEnvConfig(budget_B=B0, production_cost=C, seed=seed,
                          weight_high=W_HIGH, n_mc_samples=N_MC)
    env = BudgetRevenueEnv(G, cfg); env.reset()
    nodes = list(G.nodes()); n = len(nodes)
    ei_d = ei.to(device)
    pol.reset_episode(device)
    revenue = 0.0; n_below = 0

    while getattr(env, 'available_nodes', None) is not None and \
          len(env.available_nodes) > 0 and not env._check_bankrupt():
        feats20 = compute_node_features_fast(cache, env.S, set(env.offered), env.t, n, env)
        bfrac = np.full((n, 1), max(0.0, env.B / B0))   # budget fraction remaining
        feats = np.concatenate([feats20, bfrac], axis=1)
        x  = torch.tensor(feats[:, :21], dtype=torch.float32, device=device)
        av = torch.tensor([v not in env.offered for v in nodes], dtype=torch.bool, device=device)
        if not av.any(): break
        with torch.no_grad():
            ms, h, ctx, _ = pol.forward(x, ei_d, av)
            safe = ms.clone(); safe[~av] = -1e9
            sel = int(safe.argmax())
            disc = float(pol.get_discount_distribution(
                torch.cat([h[sel], ctx])).mean.clamp(1e-4, 1-1e-4).detach())
        v = nodes[sel]
        _, r, done, info = env.step(env.node_to_idx[v], disc)
        if r > 0:
            price = float(info.get('offered_price', r))
            if price < C: n_below += 1
            revenue += r
        if done: break

    S_T    = len(env.S)
    B_rem  = float(env.B)
    profit = revenue - C * S_T
    pi_check = B_rem - B0
    assert abs(profit - pi_check) < 0.01 + 0.001 * abs(profit), \
        f"Pi identity FAIL: profit={profit:.3f} B_T-B0={pi_check:.3f}"
    return revenue, profit, n_below, S_T

# ── CGS baseline ───────────────────────────────────────────────────────────────

def run_cgs_budget(G, B0, seeds):
    revs=[]; profits=[]; bcs=[]; s_ts=[]
    try:
        from _cal_episode_utils import calibrate, BudgetEnvConfig as BECfg, _sel_episode
        from src.evaluation.budget_baselines import _make_env
        cal_cfg = BECfg(weight_low=0.0, weight_high=W_HIGH, n_mc_samples=5)
        cal = calibrate(G, cal_cfg)
        V, A, _, cb, ib = cal
        for seed in seeds:
            env = _make_env(G, B=B0, c=C, seed=seed); env.reset()
            _sel_episode(3, 1.0, env, G, A, V, cb, ib)
            rev = float(getattr(env, 'total_revenue', 0.0))
            s_t = len(getattr(env, 'S', []))
            revs.append(rev); profits.append(rev - C*s_t)
            bcs.append(0); s_ts.append(s_t)
    except Exception as e:
        print(f"  CGS error: {e}", flush=True)
        revs=[float('nan')]*len(seeds); profits=revs[:]
        bcs=[float('nan')]*len(seeds); s_ts=bcs[:]
    return revs, profits, bcs, s_ts

# ── IE+Budget and Greedy+Budget baselines ─────────────────────────────────────

def _extract_bl(res, n):
    """Extract per-trial arrays from aggregated baseline result dict.
    budget_baselines._aggregate returns {"key": {"mean":X, "std":Y, "all":[r0..rN-1]}}.
    Keys used: revenue, n_paid_accepted (or n_accepted), n_subsidized.
    """
    if not isinstance(res, dict):
        return [float(res)]*n, [0]*n, [0]*n
    revs = res.get('revenue', {})
    revs = revs.get('all', [revs.get('mean', 0.0)]*n) if isinstance(revs,dict) else [float(revs)]*n
    ns_d = res.get('n_paid_accepted', res.get('n_accepted', res.get('n_sales', {})))
    ns   = ns_d.get('all', [ns_d.get('mean', 0)]*n) if isinstance(ns_d,dict) else [int(ns_d)]*n
    bc_d = res.get('n_subsidized', res.get('n_below_cost', {}))
    bcs  = bc_d.get('all', [bc_d.get('mean', 0)]*n) if isinstance(bc_d,dict) else [int(bc_d)]*n
    # Pad/trim to exactly n
    def _pad(lst): return (list(lst)+[float('nan')]*n)[:n]
    return _pad(revs), _pad(ns), _pad(bcs)

def run_ie_budget(G, B0, seeds):
    n = len(seeds)
    try:
        from src.evaluation.budget_baselines import efficiency_greedy_budget
        res = efficiency_greedy_budget(G, B0, C, n_trials=n)
        revs, ns, bcs = _extract_bl(res, n)
        profits = [r - C*s for r,s in zip(revs, ns)]
        s_ts = [int(s) for s in ns]
        return revs, profits, bcs, s_ts
    except Exception as e:
        print(f"  IE error: {e}", flush=True)
        nan = [float('nan')]*n
        return nan, nan, nan, nan

def run_greedy_budget(G, B0, seeds):
    n = len(seeds)
    try:
        from src.evaluation.budget_baselines import greedy_discount_budget
        res = greedy_discount_budget(G, B=B0, c=C, n_trials=n)
        revs, ns, bcs = _extract_bl(res, n)
        profits = [r - C*s for r,s in zip(revs, ns)]
        s_ts = [int(s) for s in ns]
        return revs, profits, bcs, s_ts
    except Exception as e:
        print(f"  Greedy+Budget error: {e}", flush=True)
        nan = [float('nan')]*n
        return nan, nan, nan, nan

# ── result helpers ─────────────────────────────────────────────────────────────

def _cell(vals):
    v = [x for x in vals if not (isinstance(x,float) and np.isnan(x))]
    if not v: return {"mean": float('nan'), "std": float('nan'), "vals": vals}
    return {"mean": round(float(np.mean(v)),2), "std": round(float(np.std(v)),2),
            "vals": [round(x,2) for x in vals]}

def _fmt(cell):
    if cell["mean"] != cell["mean"]: return "   nan ±  nan"
    return f"{cell['mean']:7.2f} ±{cell['std']:5.2f}"

# ── main (single network) ──────────────────────────────────────────────────────

def run_network(net_name, pols, device, kappas):
    print(f"\n{'='*60}\nNetwork: {net_name}\n{'='*60}", flush=True)
    G = load_graph(net_name)
    print(f"  n={G.number_of_nodes()} m={G.number_of_edges()}", flush=True)
    cache = build_graph_feature_cache(G, compute_static_features(G))
    ei = _ei(G)
    net_res = {}

    for kappa in kappas:
        B0 = kappa * C
        print(f"\n  kappa={kappa}  B0={B0:.2f}", flush=True)
        kr = {}

        for m_name, (pol, runner) in pols.items():
            revs=[]; profits=[]; bcs=[]; s_ts=[]
            for seed in SEEDS:
                try:
                    rev, profit, n_bc, s_t = runner(pol, G, cache, ei, B0, seed, device)
                    revs.append(rev); profits.append(profit)
                    bcs.append(n_bc); s_ts.append(s_t)
                except AssertionError as ae:
                    print(f"    {m_name} seed={seed} Pi-ASSERT: {ae}", flush=True)
                    revs.append(float('nan')); profits.append(float('nan'))
                    bcs.append(float('nan')); s_ts.append(float('nan'))
                except Exception as e:
                    print(f"    {m_name} seed={seed} error: {e}", flush=True)
                    revs.append(float('nan')); profits.append(float('nan'))
                    bcs.append(float('nan')); s_ts.append(float('nan'))
            kr[m_name] = {"profit": _cell(profits), "revenue": _cell(revs),
                          "n_below_cost": _cell(bcs), "S_T": _cell(s_ts)}
            print(f"    {m_name:20s}  Pi={_fmt(kr[m_name]['profit'])}  "
                  f"rev={_fmt(kr[m_name]['revenue'])}  "
                  f"below={kr[m_name]['n_below_cost']['mean']:5.1f}  "
                  f"|S|={kr[m_name]['S_T']['mean']:5.1f}", flush=True)

        # Baselines (CPU)
        for bl_name, bl_fn in [("IE+Budget", run_ie_budget),
                                ("Greedy+Budget", run_greedy_budget),
                                ("CGS", run_cgs_budget)]:
            try:
                r, p, bc, st = bl_fn(G, B0, SEEDS)
                kr[bl_name] = {"profit": _cell(p), "revenue": _cell(r),
                               "n_below_cost": _cell(bc), "S_T": _cell(st),
                               "Pi_assert_max_mismatch": None}
                print(f"    {bl_name:20s}  Pi={_fmt(kr[bl_name]['profit'])}  "
                      f"rev={_fmt(kr[bl_name]['revenue'])}", flush=True)
            except Exception as e:
                print(f"    {bl_name} error: {e}", flush=True)

        net_res[str(kappa)] = kr
    return net_res

# ── paired tests ───────────────────────────────────────────────────────────────

def paired_tests(all_results):
    print("\n=== PAIRED TESTS ON PROFIT (FF_1000, kappa=5) ===", flush=True)
    ref_key = "LSTM-densemix"
    for kappa_str in ["5", "20"]:
        print(f"\n  kappa={kappa_str}:", flush=True)
        for net in ["FF_1000", "polblogs"]:
            if net not in all_results: continue
            kr = all_results[net].get(kappa_str, {})
            ref = kr.get(ref_key, {}).get("profit", {}).get("vals")
            if not ref: continue
            for arm in ["Arm-B", "Arm-C", "Phase-1"]:
                v = kr.get(arm, {}).get("profit", {}).get("vals")
                if not v or len(v) != len(ref): continue
                diff = [a-b for a,b in zip(v, ref)]
                t_stat, p = scipy_stats.ttest_rel(v, ref)
                ci95 = scipy_stats.t.ppf(0.975, len(diff)-1) * \
                       np.std(diff, ddof=1) / np.sqrt(len(diff))
                flag = "NOT SIG" if p >= 0.05 else ("sig+" if np.mean(diff)>0 else "sig-")
                print(f"    {net} {arm} vs {ref_key}: "
                      f"diff={np.mean(diff):+.2f} ±{ci95:.2f}  p={p:.4f}  {flag}", flush=True)

# ── merge shards ───────────────────────────────────────────────────────────────

def merge_shards():
    NETS = ["FF_1000", "FF_2000", "Modular_FF", "Rice_FB", "polblogs"]
    combined = {}
    for net in NETS:
        shard = os.path.join(LOG, f"armBC_profit_{net}.json")
        if os.path.exists(shard):
            combined[net] = json.load(open(shard))
            print(f"  loaded {shard}", flush=True)
        else:
            print(f"  MISSING: {shard}", flush=True)
    json.dump(combined, open(OUT, "w"), indent=2)
    print(f"\nMerged → {OUT}")
    subprocess.run(["git", "add", "-f", OUT], cwd=_ROOT)
    subprocess.run(["git", "commit", "-m", "armBC_profit.json"], cwd=_ROOT)
    h = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True, cwd=_ROOT).stdout.strip()
    print(h)

# ── entry point ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="FF_1000",
                    choices=["FF_1000","FF_2000","Modular_FF","Rice_FB","polblogs","all"])
    ap.add_argument("--ckpt_b",  default="")
    ap.add_argument("--ckpt_c",  default="")
    ap.add_argument("--ckpt_p1", default=os.path.join(CKPT, "c1_p1_s1_ep0200.pt"))
    ap.add_argument("--device",  default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--merge",   action="store_true", help="Merge per-network shards into OUT")
    args = ap.parse_args()

    print(CONFIG_NOTE, flush=True)

    if args.merge:
        merge_shards(); return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device}  MC={N_MC}  C={C}", flush=True)

    def _latest(arm):
        import glob
        pts = sorted(glob.glob(os.path.join(CKPT, f"c1_{arm}_s*_ep*.pt")))
        return pts[-1] if pts else ""

    ckpt_b  = args.ckpt_b  or _latest("b")
    ckpt_c  = args.ckpt_c  or _latest("c")
    ckpt_p1 = args.ckpt_p1

    print(f"Phase-1: {ckpt_p1}")
    print(f"Arm-B:   {ckpt_b or 'NOT FOUND'}")
    print(f"Arm-C:   {ckpt_c or 'NOT FOUND'}")
    print(f"LSTM-densemix: {EXISTING_CKPT} (sha={EXISTING_SHA})", flush=True)

    pols = {}
    if ckpt_p1 and os.path.exists(ckpt_p1):
        pols["Phase-1"] = (load_pol(ckpt_p1, device), run_pol_budget)
    if ckpt_b and os.path.exists(ckpt_b):
        pols["Arm-B"]   = (load_pol(ckpt_b, device),  run_pol_budget)
    if ckpt_c and os.path.exists(ckpt_c):
        pols["Arm-C"]   = (load_pol(ckpt_c, device),  run_pol_budget)
    if os.path.exists(EXISTING_CKPT):
        pols["LSTM-densemix"] = (load_pol(EXISTING_CKPT, device), run_existing_budget)

    nets = [args.network] if args.network != "all" else \
           ["FF_1000","FF_2000","Modular_FF","Rice_FB","polblogs"]

    all_results = {}
    for net in nets:
        try:
            all_results[net] = run_network(net, pols, device, KAPPAS)
            shard = os.path.join(LOG, f"armBC_profit_{net}.json")
            os.makedirs(LOG, exist_ok=True)
            json.dump(all_results[net], open(shard, "w"), indent=2)
            print(f"\n  Saved shard → {shard}", flush=True)
        except Exception as e:
            print(f"  {net} FAILED: {e}", flush=True)
            import traceback; traceback.print_exc()

    if len(nets) > 1:
        paired_tests(all_results)


if __name__ == "__main__":
    main()
