#!/usr/bin/env python3
"""fairness_kappa_sweep.py
Fairness sweep at kappa in {8, 10, 12, 15} for polblogs + rice_fb.
All 5 methods, seeds [0..9].
Reports: reach MIN/MAJ, price MIN/MAJ, abs gap, relative gap (min/maj-1), <5% flag.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from experiments.synthetic_fairness import (
    _group_metrics, _ep_cgs, _ep_lstm, _ep_greedy, _ep_ie,
    _ep_degree_blind, _get_C,
)
from _cal_episode_utils import calibrate, BudgetEnvConfig as BECfg
from _arm_b_utils import load_arm_b, make_ei

SEEDS   = list(range(10))
KAPPAS  = [8, 10, 12, 15]
METHODS = ["CGS", "LSTM", "Greedy+Budget", "IE+Budget", "DegreeBlind"]

def _get_B(kap_int):
    return kap_int * _get_C()

# ── graph loaders (same as fairness_real_v2.py) ────────────────────────────────

def _load_polblogs():
    import zipfile, urllib.request, networkx as nx
    _DATA = str(Path(__file__).parent.parent / "data" / "raw")
    gml_path = os.path.join(_DATA, "polblogs.gml")
    def _from_gml(path):
        G = nx.read_gml(path)
        if not nx.is_connected(G):
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        G = nx.convert_node_labels_to_integers(G)
        for v in G.nodes():
            val = G.nodes[v].get('value', None)
            G.nodes[v]['group'] = int(val) if val is not None else 0
        return G
    if os.path.exists(gml_path):
        return _from_gml(gml_path)
    from torch_geometric.datasets import PolBlogs
    import networkx as nx
    ds = PolBlogs('/tmp/polblogs_pyg'); d = ds[0]
    ei = d.edge_index.numpy()
    G = nx.Graph(); G.add_nodes_from(range(int(d.num_nodes)))
    G.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    y = d.y.numpy() if hasattr(d.y, 'numpy') else d.y
    for v in range(d.num_nodes): G.nodes[v]['group'] = int(y[v])
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        G = nx.convert_node_labels_to_integers(G)
    return G

def _load_rice():
    from src.env.graph_generators import load_rice_facebook
    import os
    _DATA = str(Path(__file__).parent.parent / "data" / "raw")
    G = load_rice_facebook()
    users_path = os.path.join(_DATA, "rice-facebook-undergrads-users.txt")
    _MINORITY = {3, 4, 9}
    user_dorm = {}
    if os.path.exists(users_path):
        with open(users_path) as f:
            for line in f:
                p = line.split()
                if len(p) >= 2: user_dorm[int(p[0])] = int(p[1])
    for v in G.nodes():
        d = user_dorm.get(v)
        G.nodes[v]['group'] = (0 if d in _MINORITY else 1) if d is not None else 1
    return G

NETWORKS = {"polblogs": _load_polblogs, "rice_fb": _load_rice}

# ── run one cell ──────────────────────────────────────────────────────────────

def _run_seed(method, G, B, seed, cal, pol, ei, cache, device):
    if method == "CGS":           return _ep_cgs(G, B, seed, cal)
    if method == "LSTM":          return _ep_lstm(pol, G, cache, ei, B, seed, device)
    if method == "Greedy+Budget": return _ep_greedy(G, B, seed)
    if method == "IE+Budget":     return _ep_ie(G, B, seed)
    if method == "DegreeBlind":   return _ep_degree_blind(G, B, seed, cal)
    raise ValueError(method)

def _run_cell(method, G, labels, g0, g1, B, seeds, cal, pol, ei, cache, device):
    reach_min=[]; reach_maj=[]; price_min=[]; price_maj=[]
    for s in seeds:
        try:
            env = _run_seed(method, G, B, s, cal, pol, ei, cache, device)
            gm  = _group_metrics(env, labels, [g0, g1])
            reach_min.append(gm[str(g0)]['reach'])
            reach_maj.append(gm[str(g1)]['reach'])
            price_min.append(gm[str(g0)]['price_mean'])
            price_maj.append(gm[str(g1)]['price_mean'])
        except Exception as e:
            print(f"    {method} seed={s} error: {e}", flush=True)
    def _s(v): return (float(np.mean(v)), float(np.std(v)))
    return {
        'reach_min': _s(reach_min), 'reach_maj': _s(reach_maj),
        'price_min': _s(price_min), 'price_maj': _s(price_maj),
    }

# ── format helpers ────────────────────────────────────────────────────────────

def _rel(a, b):
    """Relative gap: a/b - 1 (= minority/majority - 1)."""
    return a / b - 1.0 if abs(b) > 1e-9 else float('nan')

def _flag(v, n):
    """Flag if v < 5% of network."""
    return " [<5%]" if v < 0.05 else ""

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cal_cfg = BECfg(weight_low=0.0, weight_high=2.0, n_mc_samples=5)

    HDR = (f"\n{'net':8s} {'kap':3s} {'method':18s} "
           f"{'reach_min':>9s} {'reach_maj':>9s} {'abs_gap':>8s} {'rel_gap%':>8s} "
           f"{'price_min':>9s} {'price_maj':>9s} {'price_abs':>9s} {'price_rel%':>10s}")
    SEP = "-" * len(HDR.lstrip("\n"))

    for net_name, loader_fn in NETWORKS.items():
        import networkx as nx
        print(f"\n{'='*70}\nNetwork: {net_name}")
        G = loader_fn()
        n = G.number_of_nodes()
        labels = {v: G.nodes[v]['group'] for v in G.nodes()}
        groups = sorted(set(labels.values()))
        g0, g1 = groups[0], groups[1]
        n0 = sum(1 for l in labels.values() if l == g0)
        n1 = sum(1 for l in labels.values() if l == g1)
        print(f"  n={n}  g0(min)={n0}  g1(maj)={n1}", flush=True)

        cal = calibrate(G, cal_cfg)
        pol = load_arm_b(device)
        ei, cache = make_ei(G, device)

        print(HDR)
        print(SEP)

        for kap in KAPPAS:
            B = _get_B(kap)
            for method in METHODS:
                r = _run_cell(method, G, labels, g0, g1, B, SEEDS,
                              cal, pol, ei, cache, device)
                rm = r['reach_min'][0]; rM = r['reach_maj'][0]
                pm = r['price_min'][0]; pM = r['price_maj'][0]
                abs_r = rm - rM
                rel_r = _rel(rm, rM) * 100
                abs_p = pm - pM
                rel_p = _rel(pm, pM) * 100
                flag  = _flag(rm, n) or _flag(rM, n)
                print(f"{net_name:8s} {kap:3d} {method:18s} "
                      f"{rm:9.4f} {rM:9.4f} {abs_r:+8.4f} {rel_r:+8.2f}% "
                      f"{pm:9.4f} {pM:9.4f} {abs_p:+9.4f} {rel_p:+10.2f}%"
                      f"{flag}", flush=True)

        print(SEP, flush=True)

if __name__ == "__main__":
    main()
