#!/usr/bin/env python3
"""run_collapse_probe1.py
Step 2: Collapse diagnostic (unconstrained, 1 seed each)
         IM-RL/polblogs, LSTM/polblogs, LSTM/RiceFB
         Metrics: mean_d, frac_d_gt09, mean_price, accept_rate
Step 3a: Probe-1 redo (unconstrained, 3 seeds)
         Greedy-Discount vs LSTM on BA n=600 m=10 seed=999
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, torch, networkx as nx
from types import SimpleNamespace as NS
_orig_bc = nx.betweenness_centrality
nx.betweenness_centrality = lambda G,normalized=True,**kw: _orig_bc(
    G, k=min(200, G.number_of_nodes()), normalized=normalized, **kw)

from src.env.polblogs_loader import load_polblogs
from src.env.revenue_env import RevenueEnv, RevenueEnvConfig
from src.evaluation.baselines import greedy_discount
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.models.policies.joint_policy import JointPolicy
from src.utils.features import compute_static_features, build_graph_feature_cache, compute_node_features_fast

def _fast_gci(self, node):
    nb = list(self.graph.neighbors(node))
    if not nb: return 0.0
    tw = sum(self._link_weights.get((node,n),0.0) for n in nb)
    if tw == 0: return 0.0
    return sum(self._link_weights.get((node,n),0.0) for n in nb if n in self.S) / tw
RevenueEnv.get_current_influence = _fast_gci

HID = 64; FEAT_DIM = 20
CKPT_DIR = "results/checkpoints"

def _edge_index(G, device):
    edges = list(G.edges())
    if not edges: return torch.zeros((2,0), dtype=torch.long, device=device)
    nmap = {v:i for i,v in enumerate(G.nodes())}
    src = [nmap[u] for u,_ in edges] + [nmap[v] for _,v in edges]
    dst = [nmap[v] for _,v in edges] + [nmap[u] for u,_ in edges]
    return torch.tensor([src,dst], dtype=torch.long, device=device)

def load_lstm(dev):
    enc = GraphSAGEEncoder(FEAT_DIM, HID, 2, 0.)
    lstm = EpisodeLSTM(graph_dim=HID, lstm_hidden=HID, n_layers=1)
    pol = SequentialJointPolicy(enc, lstm, gnn_dim=HID, context_dim=HID)
    pol.load_state_dict(torch.load(os.path.join(CKPT_DIR,"rev_gnn_lstm.pt"), map_location=dev, weights_only=True))
    return pol.to(dev).eval()

def load_imrl(dev):
    enc = GraphSAGEEncoder(FEAT_DIM, HID, 2, 0.)
    pol = JointPolicy(enc, hidden_dim=HID)
    pol.load_state_dict(torch.load(os.path.join(CKPT_DIR,"rev_gnn_im_rl.pt"), map_location=dev, weights_only=True))
    return pol.to(dev).eval()

@torch.no_grad()
def run_unconstrained_diag(policy, G, cache, ei, k, seed, device, is_imrl=False):
    """Run unconstrained episode; return discount stats."""
    cfg = RevenueEnvConfig(seed=seed)
    env = RevenueEnv(G, cfg); env.reset()
    n = G.number_of_nodes()
    if not is_imrl: policy.reset_episode(device)
    discounts, prices_offered, accepted = [], [], 0
    for _ in range(n):
        if not env.available_nodes: break
        x_np = compute_node_features_fast(cache, env.S, env.offered, env.t, k, env)
        x_t = torch.FloatTensor(x_np).to(device)
        avail = torch.tensor([v not in env.offered and v not in env.S
                              for v in G.nodes()], dtype=torch.bool, device=device)
        if not avail.any(): break
        if is_imrl:
            ni, d, _ = policy.select_and_price(x_t, ei, avail, greedy=True)
        else:
            sc, h, ctx, _ = policy.forward(x_t, ei, avail)
            ni = int(sc.argmax().item())
            d = float(policy.get_discount_distribution(torch.cat([h[ni], ctx])).mean.item())
        discounts.append(d)
        price = (1.0 - d)  # price relative to reference
        prices_offered.append(price)
        obs, rw, done, info = env.step(ni, d)
        if info.get("accepted"): accepted += 1
        if done: break
    n_off = len(discounts)
    mean_d = float(np.mean(discounts)) if discounts else 0.0
    frac_hi = float(np.mean([dd > 0.9 for dd in discounts])) if discounts else 0.0
    mean_price = float(np.mean(prices_offered)) if prices_offered else 0.0
    acc_rate = accepted / n_off if n_off > 0 else 0.0
    return mean_d, frac_hi, mean_price, acc_rate

@torch.no_grad()
def run_unconstrained_revenue(policy, G, cache, ei, k, seed, device, is_imrl=False):
    """Unconstrained revenue for Probe-1."""
    cfg = RevenueEnvConfig(seed=seed)
    env = RevenueEnv(G, cfg); env.reset()
    n = G.number_of_nodes()
    if not is_imrl: policy.reset_episode(device)
    rev = 0.0
    for _ in range(n):
        if not env.available_nodes: break
        x_np = compute_node_features_fast(cache, env.S, env.offered, env.t, k, env)
        x_t = torch.FloatTensor(x_np).to(device)
        avail = torch.tensor([v not in env.offered and v not in env.S
                              for v in G.nodes()], dtype=torch.bool, device=device)
        if not avail.any(): break
        if is_imrl:
            ni, d, _ = policy.select_and_price(x_t, ei, avail, greedy=True)
        else:
            sc, h, ctx, _ = policy.forward(x_t, ei, avail)
            ni = int(sc.argmax().item())
            d = float(policy.get_discount_distribution(torch.cat([h[ni], ctx])).mean.item())
        obs, rw, done, info = env.step(ni, d)
        if info.get("accepted"): rev += info.get("offered_price", 0.0)
        if done: break
    return rev

def main():
    t0 = time.time()
    device = torch.device("cpu")
    K = 20

    print("Loading policies...")
    lstm = load_lstm(device)
    imrl = load_imrl(device)
    print("Done.")

    # ─── Step 2: Collapse diagnostic ─────────────────────────────────────────
    print("\n=== STEP 2: Collapse Diagnostic (unconstrained, seed=42) ===")
    print(f"{'Model':<20} | {'mean_d':>6} | {'frac_d>0.9':>10} | {'mean_price':>10} | {'acc_rate':>8}")
    print("-" * 70)

    polblogs = load_polblogs()
    pb_static = compute_static_features(polblogs)
    pb_cache = build_graph_feature_cache(polblogs, pb_static)
    pb_ei = _edge_index(polblogs, device)

    # IM-RL on polblogs
    md, fh, mp, ar = run_unconstrained_diag(imrl, polblogs, pb_cache, pb_ei, K, 42, device, is_imrl=True)
    print(f"{'IM-RL/polblogs':<20} | {md:>6.3f} | {fh:>10.3f} | {mp:>10.3f} | {ar:>8.3f}")

    # LSTM on polblogs
    md, fh, mp, ar = run_unconstrained_diag(lstm, polblogs, pb_cache, pb_ei, K, 42, device, is_imrl=False)
    print(f"{'LSTM/polblogs':<20} | {md:>6.3f} | {fh:>10.3f} | {mp:>10.3f} | {ar:>8.3f}")

    # LSTM on Rice-FB
    try:
        from src.env.ricefb_loader import load_ricefb
        ricefb = load_ricefb()
        rf_static = compute_static_features(ricefb)
        rf_cache = build_graph_feature_cache(ricefb, rf_static)
        rf_ei = _edge_index(ricefb, device)
        md, fh, mp, ar = run_unconstrained_diag(lstm, ricefb, rf_cache, rf_ei, K, 42, device, is_imrl=False)
        print(f"{'LSTM/rice-fb':<20} | {md:>6.3f} | {fh:>10.3f} | {mp:>10.3f} | {ar:>8.3f}")
    except Exception as e:
        print(f"{'LSTM/rice-fb':<20} | ERROR: {e}")

    # ─── Step 3a: Probe-1 redo (unconstrained) ───────────────────────────────
    print("\n=== STEP 3a: Probe-1 Redo (unconstrained, BA n=600 m=10 seed=999) ===")
    # Build BA proxy graph
    rng = np.random.default_rng(999)
    try:
        G_ba = nx.barabasi_albert_graph(600, 10, seed=999)
    except Exception:
        G_ba = nx.barabasi_albert_graph(600, 10)
    ba_static = compute_static_features(G_ba)
    ba_cache = build_graph_feature_cache(G_ba, ba_static)
    ba_ei = _edge_index(G_ba, device)

    seeds3 = [42, 123, 7]
    # Greedy-Discount baseline (uses SimpleNamespace cfg)
    greedy_revs = []
    for s in seeds3:
        try:
            cfg_g = NS(influence=NS(model="monotone",b=1.0,weight_low=0.,weight_high=2.,n_mc_samples=200),
                       reward=NS(type="flat",gamma=1.0),budget=NS(k=K),project=NS(seed=s))
            r = greedy_discount(G_ba, cfg_g)
            greedy_revs.append(float(r))
        except Exception as e:
            print(f"Greedy error seed={s}: {e}"); greedy_revs.append(0.0)

    # LSTM baseline
    lstm_revs = []
    for s in seeds3:
        r = run_unconstrained_revenue(lstm, G_ba, ba_cache, ba_ei, K, s, device)
        lstm_revs.append(r)

    g_mean = np.mean(greedy_revs); l_mean = np.mean(lstm_revs)
    ratio = l_mean / g_mean if g_mean > 0 else float("nan")
    print(f"Greedy seeds: {[round(r,1) for r in greedy_revs]} mean={g_mean:.1f}")
    print(f"LSTM   seeds: {[round(r,1) for r in lstm_revs]} mean={l_mean:.1f}")
    print(f"Ratio LSTM/Greedy = {ratio:.3f}  ({'VALID (>1)' if ratio>1 else 'INVALID (<1)'})")

    print(f"\nWall: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
