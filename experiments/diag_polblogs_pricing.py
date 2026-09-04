"""
diag_polblogs_pricing.py — Items 1-3 polblogs pricing failure analysis.
polblogs kappa=15 (B=4.5, c=0.3) seed=0.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, torch
from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.env.polblogs_loader import load_polblogs
from src.env.graph_generators import generate_forest_fire
from _arm_b_utils import load_arm_b, make_ei, ARM_B_SHA, N_MC, W_HIGH, C
from _arm_b_utils import _feat_unconstrained, _avail_mask
from _cal_episode_utils import calibrate, _sel_episode, _ib as _ib_fn
from src.utils.helpers import set_seed

device = torch.device("cpu")
pol = load_arm_b(device)
print(f"arm_b sha={ARM_B_SHA}  W_HIGH={W_HIGH}  c={C}  N_MC={N_MC}")

def run_armed_trace(graph, B, seed, max_trace=30):
    n = graph.number_of_nodes()
    set_seed(seed)
    cfg = BudgetEnvConfig(budget_B=B, production_cost=C, seed=seed,
                          weight_high=W_HIGH, n_mc_samples=N_MC)
    env = BudgetRevenueEnv(graph, cfg); env.reset()
    pol.reset_episode(device)
    static = __import__('src.utils.features', fromlist=['compute_static_features']).compute_static_features(graph)
    from src.utils.features import build_graph_feature_cache
    cache = build_graph_feature_cache(graph, static)
    from src.utils.helpers import graph_to_pyg_data; import numpy as _np
    dummy = _np.zeros((n, 21), _np.float32)
    from torch_geometric.data import Data
    ei = make_ei(graph, device)[0]

    trace=[]; offers=0; accepted=0; skipped=0
    below_c=[]; above_c=[]; step=0
    decile_data=[]  # (step, accepted_flag, price, b_before)
    disc_vhat=[]    # (discount, v_hat)

    while env.available_nodes and not env._check_bankrupt():
        x  = torch.FloatTensor(_feat_unconstrained(cache, env, n)).to(device)
        av = _avail_mask(env, n, device)
        if not av.any(): break
        sc, h, ctx, _ = pol.forward(x, ei, av)
        ni = int(sc.argmax().item())
        node = env.idx_to_node[ni] if hasattr(env,'idx_to_node') else ni
        d  = float(pol.get_discount_distribution(torch.cat([h[ni],ctx])).mean.item())
        v_hat = float(env._estimate_valuation(node))
        price = v_hat * (1.0 - d)
        b_before = env.B
        _, r, done, info = env.step(ni, d)
        b_after = env.B
        acc = bool(info.get("accepted", r > 0))
        offers += 1
        decile_data.append((step, acc, price, b_before, b_after))
        disc_vhat.append((d, v_hat, acc, step))
        if acc:
            accepted += 1
            if price < C: below_c.append(price)
            else: above_c.append(price)
            if len(trace) < max_trace:
                deg = graph.degree(node)
                trace.append((step, node, deg, d, price, b_before, b_after))
        else:
            skipped += 1
        pol.update_sequence_state(d, info.get("accepted", acc), info.get("revenue_step",0.0))
        if done: break
        step += 1

    rev = float(env.total_revenue); ns = len(env.S)
    profit = rev - C * ns; fb = env.B
    return trace, offers, accepted, skipped, below_c, above_c, rev, ns, profit, fb, decile_data, disc_vhat

# ─── ITEM 1 ──────────────────────────────────────────────────────────────────
for net_name, graph, k in [("polblogs", load_polblogs(), 15), ("polblogs", load_polblogs(), 15)]:
    break  # just polblogs k=15

B = k * C
print(f"\n=== ITEM 1  {net_name}  kappa={k}  B={B}  seed=0 ===")
tr, off, acc, skip, bl, ab, rev, ns, prof, fb, deciles, dv = run_armed_trace(graph, B, seed=0)
print(f"{'step':>5} {'node':>6} {'deg':>5} {'disc':>6} {'price':>7} {'B_bef':>8} {'B_aft':>8}")
for row in tr:
    step, node, deg, d, price, bb, ba = row
    print(f"{step:5d} {node:6d} {deg:5d} {d:6.3f} {price:7.4f} {bb:8.4f} {ba:8.4f}")
print(f"\nOffers={off}  Accepted={acc}  Skipped={skip}")
print(f"Below-c: n={len(bl)}  mean={np.mean(bl):.4f}" if bl else "Below-c: n=0")
print(f"Above-c: n={len(ab)}  mean={np.mean(ab):.4f}" if ab else "Above-c: n=0")
print(f"Revenue={rev:.2f}  |S|={ns}  Profit={prof:.2f}  FinalB={fb:.4f}")

# CGS comparison
print(f"\n--- CGS same cell ---")
V,A,P,cb,ib = calibrate(graph)
from src.evaluation.budget_baselines import _make_env
env_cgs = _make_env(graph, B=B, c=C, seed=0, weight_high=W_HIGH); env_cgs.reset()
rev_c, ns_c = _sel_episode(3, 1.0, env_cgs, graph, A, V, cb, ib)
prof_c = rev_c - C*ns_c
print(f"Revenue={rev_c:.2f}  |S|={ns_c}  Profit={prof_c:.2f}  FinalB={env_cgs.B:.4f}")

# ─── ITEM 2 ──────────────────────────────────────────────────────────────────
print(f"\n=== ITEM 2  polblogs kappa={k} deciles by step ===")
if deciles:
    total_steps = len(deciles)
    bins = np.array_split(deciles, 10)
    print(f"{'dec':>4} {'n_acc':>6} {'mean_p':>8} {'B_end':>8}")
    for i, bin_ in enumerate(bins):
        n_acc = sum(1 for s,a,p,bb,ba in bin_ if a)
        prices = [p for s,a,p,bb,ba in bin_ if a]
        b_end = bin_[-1][4] if bin_ else float('nan')
        print(f"{i:4d} {n_acc:6d} {np.mean(prices) if prices else float('nan'):8.4f} {b_end:8.4f}")

# ─── ITEM 3 ──────────────────────────────────────────────────────────────────
print(f"\n=== ITEM 3  discount vs v_hat correlation ===")
for net_name2, g2, k2 in [("polblogs", graph, 15), ("FF_1000", generate_forest_fire(1000,0.37,0.32,seed=0), 15)]:
    B2 = k2 * C
    _, _, _, _, _, _, _, _, _, _, _, dv2 = run_armed_trace(g2, B2, seed=0)
    if len(dv2) > 4:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(dv2), min(200, len(dv2)), replace=False)
        discs = np.array([dv2[i][0] for i in idx])
        vhats = np.array([dv2[i][1] for i in idx])
        corr = float(np.corrcoef(discs, vhats)[0,1])
        q1 = vhats <= np.percentile(vhats, 25)
        q4 = vhats >= np.percentile(vhats, 75)
        print(f"{net_name2:12s}  k={k2}  corr(disc,vhat)={corr:+.3f}  "
              f"mean_disc_q1(lo_v)={discs[q1].mean():.3f}  mean_disc_q4(hi_v)={discs[q4].mean():.3f}")

print("\n=== DIAGNOSIS ===")
