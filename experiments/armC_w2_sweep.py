"""
armC_w2_sweep.py — Arm C (profit objective) vs Rev-GNN-LSTM, IE, Greedy, CGS
at W_HIGH=2.0, BudgetRevenueEnv, all 5 networks x 6 kappas x 10 seeds.

Arm C checkpoint: c1_ffba_2to1_final.pt (profit objective, trained at W_HIGH=1.0)
Rev-GNN-LSTM:    rev_gnn_lstm_densemix.pt (sha=0b549f93, revenue objective)
"""
import sys, os, json, hashlib, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, torch
from scipy import stats

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.env.polblogs_loader import load_polblogs
from src.env.graph_generators import (generate_forest_fire,
                                       generate_modular_forest_fire,
                                       load_rice_facebook)
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.utils.features import (compute_node_features_fast, compute_static_features,
                                  build_graph_feature_cache)
from src.utils.helpers import graph_to_pyg_data, set_seed
from src.evaluation.budget_baselines import greedy_discount_budget, _make_env
from src.evaluation.ie_budget import ie_strategy_budget, _greedy_seed_selection_celf, IE_K_SEEDS
from _cal_episode_utils import calibrate, _sel_episode
from _arm_b_utils import make_ei, N_MC, W_HIGH, C, ARM_B_SHA
from _arm_b_utils import _feat_unconstrained, _avail_mask, load_arm_b

assert W_HIGH == 2.0

# ── Arm C checkpoint ─────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARM_C_CKPT = os.path.join(_ROOT, "results/checkpoints/c1_ffba_2to1_final.pt")
ARM_C_SHA  = None  # computed at load time


def _sha8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


def load_policy(ckpt_path, device):
    enc  = GraphSAGEEncoder(in_dim=21, hidden_dim=64, n_layers=2)
    lstm = EpisodeLSTM(graph_dim=64, lstm_hidden=64, n_layers=1)
    pol  = SequentialJointPolicy(enc, lstm, gnn_dim=64, context_dim=64)
    sd = torch.load(ckpt_path, map_location="cpu")
    if "policy_state_dict" in sd: sd = sd["policy_state_dict"]
    elif "model_state_dict" in sd: sd = sd["model_state_dict"]
    pol.load_state_dict(sd, strict=True)
    return pol.eval().to(device)


K_VALUES = [5, 10, 15, 20, 30, 40]
SEEDS    = list(range(10))
NETS     = ["FF_1000", "FF_2000", "Modular_FF", "Rice_FB", "polblogs"]
N_SIMS   = 5


def load_graph(net):
    if net == "polblogs":   return load_polblogs()
    if net == "FF_1000":    return generate_forest_fire(1000, 0.37, 0.32, seed=0)
    if net == "Rice_FB":    return load_rice_facebook()
    if net == "Modular_FF": return generate_modular_forest_fire([250,250],0.37,0.32,0.05,seed=0)
    if net == "FF_2000":    return generate_forest_fire(2000, 0.37, 0.32, seed=1)
    raise ValueError(net)


def _stats(vals):
    a = np.array([v for v in vals if not np.isnan(v)], dtype=float)
    if not len(a): return {"mean": float("nan"), "std": float("nan"), "all": list(vals)}
    return {"mean": round(float(a.mean()), 3), "std": round(float(a.std()), 3),
            "all": [round(float(v), 3) for v in vals]}


@torch.no_grad()
def _run_policy_episode(pol, graph, cache, ei, B, seed, device, trace_first30=False):
    """Returns (revenue, |S_T|, n_below_c, profit, below_prices, above_prices, step_trace)."""
    n = graph.number_of_nodes()
    set_seed(seed)
    cfg = BudgetEnvConfig(budget_B=B, production_cost=C, seed=seed,
                          weight_high=W_HIGH, n_mc_samples=N_MC)
    env = BudgetRevenueEnv(graph, cfg); env.reset()
    pol.reset_episode(device)
    below_p, above_p, trace = [], [], []
    while env.available_nodes and not env._check_bankrupt():
        x  = torch.FloatTensor(_feat_unconstrained(cache, env, n)).to(device)
        av = _avail_mask(env, n, device)
        if not av.any(): break
        sc, h, ctx, _ = pol.forward(x, ei, av)
        ni = int(sc.argmax().item())
        node = env.nodes[ni]
        d  = float(pol.get_discount_distribution(torch.cat([h[ni],ctx])).mean.item())
        v_hat = float(env._estimate_valuation(node))
        price = v_hat * (1.0 - d)
        b_bef = env.B
        _, r, done, info = env.step(ni, d)
        acc = bool(info.get("accepted", r > 0))
        if acc:
            if price < C: below_p.append(price)
            else: above_p.append(price)
            if trace_first30 and len(trace) < 30:
                trace.append((env.t-1, node, graph.degree(node), d, price, b_bef, env.B))
        pol.update_sequence_state(d, acc, info.get("revenue_step", 0.0))
        if done: break
    rev = float(env.total_revenue); ns = len(env.S)
    profit = env.B - B  # = rev - C*|S|
    assert abs((rev - C*ns) - profit) < 1e-5, f"profit identity fail seed={seed}"
    return rev, ns, len(below_p), profit, below_p, above_p, trace


def run_net(net, pol_c, pol_b, device, out_dir, diag_net="polblogs", diag_k=15):
    graph = load_graph(net)
    ei, cache = make_ei(graph, device)
    cfg = BudgetEnvConfig(production_cost=C, weight_high=W_HIGH, n_mc_samples=N_MC)
    print(f"\n=== {net} ===", flush=True)

    # Calibrate once
    V, A, P, cb, ib = calibrate(graph, cfg)

    # IE seed orderings once
    ie_ord = {}
    for trial in SEEDS:
        env_tmp = _make_env(graph, B=float("inf"), c=C, seed=trial, weight_high=W_HIGH)
        env_tmp.reset()
        k_s = min(IE_K_SEEDS, 999)
        ie_ord[trial] = _greedy_seed_selection_celf(graph, env_tmp, k_s)

    results = {}
    for k in K_VALUES:
        B = k * C
        t0 = time.time()

        # arm_c
        ac_rev, ac_ns, ac_bc, ac_prof = [], [], [], []
        ac_bl_p, ac_ab_p = [], []
        for seed in SEEDS:
            rv, ns, bc, pf, bl, ab, tr = _run_policy_episode(
                pol_c, graph, cache, ei, B, seed, device,
                trace_first30=(net==diag_net and k==diag_k and seed==0))
            ac_rev.append(rv); ac_ns.append(ns); ac_bc.append(bc); ac_prof.append(pf)
            ac_bl_p.extend(bl); ac_ab_p.extend(ab)
            if net==diag_net and k==diag_k and seed==0:
                _arm_c_trace = tr; _arm_c_bl=[]; _arm_c_ab=[]
                _arm_c_bl.extend(bl); _arm_c_ab.extend(ab)
                _arm_c_sum = (rv, ns, pf, bc, bl, ab)

        # arm_b
        ab_rev, ab_ns, ab_bc, ab_prof = [], [], [], []
        for seed in SEEDS:
            rv, ns, bc, pf, bl, ab2, tr = _run_policy_episode(
                pol_b, graph, cache, ei, B, seed, device,
                trace_first30=(net==diag_net and k==diag_k and seed==0))
            ab_rev.append(rv); ab_ns.append(ns); ab_bc.append(bc); ab_prof.append(pf)
            if net==diag_net and k==diag_k and seed==0:
                _arm_b_sum = (rv, ns, pf, bc, bl, ab2)

        # IE
        ie_ord_k = {t: ie_ord[t][:min(k, len(ie_ord[t]))] for t in SEEDS}
        r_ie = ie_strategy_budget(graph, B, C, n_trials=10, weight_high=W_HIGH,
                                   seed_orderings=ie_ord_k)
        def _ex(r, key, n=10):
            sub = r.get(key, {})
            return sub.get("all", [sub.get("mean",0)]*n) if isinstance(sub,dict) else [float(sub)]*n
        ie_rev = _ex(r_ie,"revenue"); ie_ns = _ex(r_ie,"n_in_S")
        ie_prof = [rv - C*ns for rv,ns in zip(ie_rev, ie_ns)]

        # Greedy
        r_gd = greedy_discount_budget(graph, B, C, n_trials=10, weight_high=W_HIGH)
        gd_rev = _ex(r_gd,"revenue"); gd_ns = _ex(r_gd,"n_in_S")
        gd_prof = [rv - C*ns for rv,ns in zip(gd_rev, gd_ns)]

        # CGS
        cgs_rev, cgs_ns, cgs_prof = [], [], []
        for seed in SEEDS:
            env_c = _make_env(graph, B=B, c=C, seed=seed, weight_high=W_HIGH); env_c.reset()
            rv_c, _ = _sel_episode(3, 1.0, env_c, graph, A, V, cb, ib)
            ns_c = len(env_c.S)
            pf_c = env_c.B - B
            assert abs((rv_c - C*ns_c) - pf_c) < 1e-5, f"CGS identity fail seed={seed}"
            cgs_rev.append(float(rv_c)); cgs_ns.append(ns_c); cgs_prof.append(float(pf_c))

        # Paired tests arm_c vs arm_b, arm_c vs IE
        def _paired(a, b):
            d = np.array(a) - np.array(b)
            t, p = float(stats.ttest_1samp(d, 0).pvalue), None
            p = float(stats.ttest_1samp(d, 0).pvalue)
            ci = float(np.mean(d)) + np.array([-1,1])*stats.t.ppf(0.975,len(d)-1)*float(np.std(d,ddof=1))/np.sqrt(len(d))
            return {"mean_diff": round(float(np.mean(d)),3),
                    "ci": [round(float(ci[0]),3), round(float(ci[1]),3)],
                    "p": round(p,4), "not_sig": p >= 0.05}

        elapsed = time.time()-t0
        cell = {
            "arm_c":    {"profit":_stats(ac_prof),"below_c_n":_stats(ac_bc),
                          "below_c_mean_price": round(np.mean(ac_bl_p),4) if ac_bl_p else 0.0,
                          "n_in_S":_stats(ac_ns),"revenue":_stats(ac_rev)},
            "arm_b":    {"profit":_stats(ab_prof),"below_c_n":_stats(ab_bc),
                          "n_in_S":_stats(ab_ns),"revenue":_stats(ab_rev)},
            "IE+Budget":{"profit":_stats(ie_prof),"n_in_S":_stats(ie_ns),"revenue":_stats(ie_rev)},
            "Greedy":   {"profit":_stats(gd_prof),"n_in_S":_stats(gd_ns),"revenue":_stats(gd_rev)},
            "CGS":      {"profit":_stats(cgs_prof),"n_in_S":_stats(cgs_ns),"revenue":_stats(cgs_rev)},
            "paired": {"arm_c_vs_arm_b": _paired(ac_prof, ab_prof),
                       "arm_c_vs_ie":    _paired(ac_prof, ie_prof)},
            "elapsed_s": round(elapsed,1),
        }
        results[k] = cell

        print(f"  k={k:2d}"
              f"  ArmC_prof={_stats(ac_prof)['mean']:+.1f}±{_stats(ac_prof)['std']:.1f}"
              f"  ArmB_prof={_stats(ab_prof)['mean']:+.1f}"
              f"  IE_prof={_stats(ie_prof)['mean']:+.1f}"
              f"  CGS_prof={_stats(cgs_prof)['mean']:+.1f}"
              f"  ({elapsed:.0f}s)", flush=True)

        # Diagnostic for polblogs k=15
        if net == diag_net and k == diag_k:
            print(f"\n  --- DIAGNOSTIC {net} k={k} ---")
            for label, (rv,ns,pf,bc_n,bl,ab2) in [("ArmC",_arm_c_sum),("ArmB",_arm_b_sum)]:
                bm = np.mean(bl) if bl else float('nan')
                am = np.mean(ab2) if ab2 else float('nan')
                print(f"  {label}: below_c n={len(bl)} mean={bm:.4f} | above_c n={len(ab2)} mean={am:.4f}"
                      f" | |S|={ns} rev={rv:.2f} profit={pf:.2f}")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks", nargs="+", default=NETS)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    arm_c_sha = _sha8(ARM_C_CKPT)
    arm_b_sha = ARM_B_SHA
    print(f"Arm C: {os.path.basename(ARM_C_CKPT)}  sha={arm_c_sha}")
    print(f"Arm B: rev_gnn_lstm_densemix.pt  sha={arm_b_sha}")

    pol_c = load_policy(ARM_C_CKPT, device)
    pol_b = load_arm_b(device)

    all_results = {"arm_c_ckpt": os.path.basename(ARM_C_CKPT), "arm_c_sha": arm_c_sha,
                   "arm_b_sha": arm_b_sha, "W_HIGH": W_HIGH, "N_MC": N_MC,
                   "c": C, "seeds": SEEDS, "networks": {}}

    for net in args.networks:
        res = run_net(net, pol_c, pol_b, device, "results/logs")
        all_results["networks"][net] = res
        # Write shard immediately so parallel runs don't race on the merged file
        shard_out = f"results/logs/armC_w2_{net}.json"
        os.makedirs("results/logs", exist_ok=True)
        with open(shard_out, "w") as f:
            json.dump({**{k: all_results[k] for k in all_results if k != "networks"},
                       "network": net, "results": res}, f, indent=2)
        print(f"  shard → {shard_out}", flush=True)

    # If all networks ran in this process, also write the merged file
    if len(all_results["networks"]) == len(args.networks):
        out = "results/logs/armC_w2.json"
        with open(out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
