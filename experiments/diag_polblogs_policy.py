"""diag_polblogs_policy.py — Items 1-3: Arm B vs Arm C on polblogs, hub analysis, pricing correlations."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from scipy.stats import pearsonr
from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.env.polblogs_loader import load_polblogs
from src.env.graph_generators import generate_forest_fire, load_rice_facebook
from src.utils.helpers import set_seed
from src.evaluation.dp_calibrated_v2_obs import calibrate_v2_obs_table as _cal_v2
from _arm_b_utils import make_ei, N_MC, W_HIGH, C, _feat_unconstrained, _avail_mask, load_arm_b

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARM_C_CKPT = os.path.join(_ROOT, "results/checkpoints/rev_gnn_lstm_tc.pt")
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy

def _load_pol_c(device):
    enc = GraphSAGEEncoder(in_dim=20, hidden_dim=64, n_layers=2)
    lstm = EpisodeLSTM(graph_dim=64, lstm_hidden=64, n_layers=1)
    pol = SequentialJointPolicy(enc, lstm, gnn_dim=64, context_dim=64)
    sd = torch.load(ARM_C_CKPT, map_location="cpu")
    if "policy_state_dict" in sd: sd = sd["policy_state_dict"]
    elif "model_state_dict" in sd: sd = sd["model_state_dict"]
    pol.load_state_dict(sd, strict=True)
    return pol.eval().to(device)

def _feat_c(cache, env, n):
    from src.utils.features import compute_node_features_fast
    return compute_node_features_fast(cache, env.S, env.offered, env.t, k=50, env=env)

@torch.no_grad()
def run_episode(pol, graph, ei, cache, B, seed, device, feat_fn, infl_bnds, max_trace=30, collect_samples=False):
    n = graph.number_of_nodes()
    set_seed(seed)
    cfg = BudgetEnvConfig(budget_B=B, production_cost=C, seed=seed, weight_high=W_HIGH, n_mc_samples=N_MC)
    env = BudgetRevenueEnv(graph, cfg); env.reset()
    pol.reset_episode(device)
    accepted_trace, all_trace, samples = [], [], []
    step_global = 0
    while env.available_nodes and not env._check_bankrupt():
        x = torch.FloatTensor(feat_fn(cache, env, n)).to(device)
        av = _avail_mask(env, n, device)
        if not av.any(): break
        sc, h, ctx, _ = pol.forward(x, ei, av)
        ni = int(sc.argmax().item())
        node = env.nodes[ni]
        d = float(pol.get_discount_distribution(torch.cat([h[ni], ctx])).mean.item())
        v_hat = float(env._estimate_valuation(node))
        price = v_hat * (1.0 - d)
        b_bef = float(env.B)
        _, r, done, info = env.step(ni, d)
        acc = bool(info.get("accepted", r > 0))
        deg = graph.degree(node)
        ib = int(np.searchsorted(infl_bnds, v_hat)) - 1
        ib = max(0, min(ib, len(infl_bnds)-2))
        rec = (step_global, node, deg, ib, d, price, b_bef, float(env.B), acc)
        all_trace.append(rec)
        if acc and len(accepted_trace) < max_trace:
            accepted_trace.append(rec)
        if collect_samples:
            samples.append((deg, ib, v_hat, d, price, b_bef, acc))
        pol.update_sequence_state(d, acc, info.get("revenue_step", 0.0))
        step_global += 1
        if done: break
    return accepted_trace, all_trace, samples, env

def episode_summary(all_trace, env, B):
    offers = len(all_trace)
    acc_recs = [r for r in all_trace if r[8]]
    skipped = offers - len(acc_recs)
    bl = [r for r in acc_recs if r[5] < C]
    ab = [r for r in acc_recs if r[5] >= C]
    rev = float(env.total_revenue)
    ns = len(env.S)
    profit = float(env.B) - B
    return {"offers": offers, "accepted": len(acc_recs), "skipped": skipped,
            "below_c_n": len(bl), "below_c_mean": np.mean([r[5] for r in bl]) if bl else float("nan"),
            "above_c_n": len(ab), "above_c_mean": np.mean([r[5] for r in ab]) if ab else float("nan"),
            "revenue": rev, "n_S": ns, "profit": profit}

def print_trace(label, trace):
    print(f"\n--- {label}: first {len(trace)} accepted offers ---")
    print(f"  {'step':>4} {'node':>6} {'deg':>5} {'ib':>3} {'disc':>6} {'price':>7} {'bal_after':>10}")
    for (st, nd, dg, ib, d, p, bb, ba, _) in trace:
        print(f"  {st:4d} {nd:6d} {dg:5d} {ib:3d} {d:6.4f} {p:7.4f} {ba:10.4f}")

def print_summary(label, s):
    print(f"\n{label} episode summary:")
    print(f"  offers={s['offers']}  accepted={s['accepted']}  skipped={s['skipped']}")
    print(f"  below-cost: n={s['below_c_n']} mean={s['below_c_mean']:.4f}")
    print(f"  above-cost: n={s['above_c_n']} mean={s['above_c_mean']:.4f}")
    print(f"  revenue={s['revenue']:.3f}  |S|={s['n_S']}  profit={s['profit']:.3f}")

def hub_analysis(label, all_trace, graph, B):
    max_deg_node = max(graph.nodes(), key=lambda v: graph.degree(v))
    max_deg = graph.degree(max_deg_node)
    offered_step = next((r[0] for r in all_trace if r[1]==max_deg_node), None)
    offered_price = next((r[5] for r in all_trace if r[1]==max_deg_node), None)
    acc_recs = [r for r in all_trace if r[8]]
    b_before_10th = acc_recs[9][6] if len(acc_recs) >= 10 else float("nan")
    spent_before_10th = B - b_before_10th if b_before_10th == b_before_10th else float("nan")
    frac = spent_before_10th / B if B > 0 else float("nan")
    print(f"\n{label} hub analysis (max_deg={max_deg} node={max_deg_node}):")
    print(f"  Hub offered at step {offered_step}, posted_price={offered_price:.4f}")
    print(f"  Budget before 10th accepted: {b_before_10th:.4f}  spent={spent_before_10th:.4f}  frac={frac:.3f}")

def corr_analysis(label, samples):
    arr = np.array([(s[0], s[1], s[2], s[3]) for s in samples])  # deg, ib, v_hat, discount
    degs, ibs, vhats, discs = arr[:,0], arr[:,1], arr[:,2], arr[:,3]
    r_deg, _ = pearsonr(degs, discs)
    r_ib,  _ = pearsonr(ibs,  discs)
    r_vh,  _ = pearsonr(vhats,discs)
    q25, q75 = np.percentile(degs, 25), np.percentile(degs, 75)
    d_low  = discs[degs <= q25].mean()
    d_high = discs[degs >= q75].mean()
    print(f"\n{label} pricing correlation (n={len(samples)}):")
    print(f"  r(discount,degree)={r_deg:+.3f}  r(discount,ib)={r_ib:+.3f}  r(discount,v_hat)={r_vh:+.3f}")
    print(f"  mean_disc Q1_deg(deg<={q25:.0f})={d_low:.4f}  Q4_deg(deg>={q75:.0f})={d_high:.4f}")

def main():
    device = torch.device("cpu")
    pol_b = load_arm_b(device)
    pol_c = _load_pol_c(device)
    K = 15; B = K * C

    graphs = {
        "polblogs": load_polblogs(),
        "FF_1000":  generate_forest_fire(1000, 0.37, 0.32, seed=0),
        "Rice_FB":  load_rice_facebook(),
    }
    cfg0 = BudgetEnvConfig(production_cost=C, weight_high=W_HIGH, n_mc_samples=N_MC)
    cals = {}
    for nm, g in graphs.items():
        V, A, P, cb, ib = _cal_v2(g, cfg0, n_sims=5, seed=0)
        cals[nm] = ib

    # ── ITEM 1 ──────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("ITEM 1  polblogs k=15 seed=0  Arm B vs Arm C  first-30 accepted")
    print("="*70)
    for label, pol, fn in [("Arm B", pol_b, _feat_unconstrained), ("Arm C", pol_c, _feat_c)]:
        g = graphs["polblogs"]
        ei, cache = make_ei(g, device)
        tr, all_tr, _, env = run_episode(pol, g, ei, cache, B, 0, device, fn, cals["polblogs"])
        print_trace(label, tr)
        s = episode_summary(all_tr, env, B)
        print_summary(label, s)

    # ── ITEM 2 ──────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("ITEM 2  Hub analysis: Arm B, polblogs / FF_1000 / Rice_FB  k=15 s=0")
    print("="*70)
    for nm in ["polblogs", "FF_1000", "Rice_FB"]:
        g = graphs[nm]
        ei, cache = make_ei(g, device)
        _, all_tr, _, env = run_episode(pol_b, g, ei, cache, B, 0, device, _feat_unconstrained, cals[nm])
        hub_analysis(f"Arm B / {nm}", all_tr, g, B)

    # ── ITEM 3 ──────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("ITEM 3  Pricing correlations: Arm B on polblogs vs FF_1000  300 samples")
    print("="*70)
    for nm in ["polblogs", "FF_1000"]:
        all_samps = []
        g = graphs[nm]
        ei, cache = make_ei(g, device)
        for seed in range(5):  # ~300 samples across 5 episodes
            _, _, samps, _ = run_episode(pol_b, g, ei, cache, B, seed, device,
                                          _feat_unconstrained, cals[nm], collect_samples=True)
            all_samps.extend(samps)
        corr_analysis(f"Arm B / {nm}", all_samps[:300])

    print("\n" + "="*70)
    print("DIAGNOSIS:")
    print("  See correlations above. If polblogs shows r(discount,v_hat)>0")
    print("  (discounts MORE for high-value buyers), that is PRICING-TRANSFER failure.")
    print("  If hub is offered early and drains budget, that is OPENING-MOVE failure.")
    print("="*70)

if __name__ == "__main__":
    main()
