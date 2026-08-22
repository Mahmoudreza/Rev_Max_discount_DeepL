#!/usr/bin/env python3
"""fairness_real_v2.py
Real-network fairness audit with POSTED-PRICE metrics.
Reuses ALL metric/episode code from synthetic_fairness.py unchanged.

Networks: polblogs (n≈1222), rice_fb (n=443)
Settings: unc, k5, k20
Methods: CGS, LSTM, Greedy+Budget, IE+Budget, DegreeBlind (corrected control)
"""
import json, os, sys, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import networkx as nx
from scipy import stats as scipy_stats

# ── Reuse ALL metric and episode code from synthetic_fairness AS IS ────────────
from experiments.synthetic_fairness import (
    _group_metrics, _total_revenue, _total_profit,
    _ep_cgs, _ep_lstm, _ep_greedy, _ep_ie, _ep_degree_blind, _get_C,
)
from _cal_episode_utils import calibrate, BudgetEnvConfig as BECfg
from _arm_b_utils import load_arm_b, make_ei

import torch

SEEDS = list(range(10))
KAPPAS_NAMES = ["unc", "k5", "k20"]
METHOD_NAMES = ["CGS", "LSTM", "Greedy+Budget", "IE+Budget", "DegreeBlind"]

_ROOT = str(Path(__file__).parent.parent)
_DATA = os.path.join(_ROOT, "data", "raw")
OUT   = os.path.join(_ROOT, "results", "logs", "fairness_real_v2.json")


def _get_B(kap_name):
    C = _get_C()
    return {"unc": 1e7, "k5": 5 * C, "k20": 20 * C}[kap_name]


# ── graph loaders (logic from fairness_audit.py) ──────────────────────────────

def _load_polblogs():
    """PolBlogs via GML cache, HTTP download, or torch_geometric fallback.
    Returns G with G.nodes[v]['group'] = binary_group (0=liberal, 1=conservative)."""
    import zipfile, urllib.request
    gml_path = os.path.join(_DATA, "polblogs.gml")

    def _from_gml(path):
        G = nx.read_gml(path)
        if not nx.is_connected(G):
            lcc = max(nx.connected_components(G), key=len)
            G = G.subgraph(lcc).copy()
        G = nx.convert_node_labels_to_integers(G)
        for v in G.nodes():
            val = G.nodes[v].get('value', None)
            G.nodes[v]['group'] = int(val) if val is not None else 0
        return G

    if os.path.exists(gml_path):
        return _from_gml(gml_path)

    url = "http://www-personal.umich.edu/~mejn/netdata/polblogs.zip"
    try:
        zip_path = os.path.join(_DATA, "polblogs_newman.zip")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            gml_names = [n for n in zf.namelist() if n.endswith('.gml')]
            zf.extract(gml_names[0], _DATA)
            extracted = os.path.join(_DATA, gml_names[0])
            if extracted != gml_path: os.rename(extracted, gml_path)
        return _from_gml(gml_path)
    except Exception:
        pass

    # torch_geometric fallback
    from torch_geometric.datasets import PolBlogs
    ds = PolBlogs('/tmp/polblogs_pyg'); d = ds[0]
    ei = d.edge_index.numpy()
    G = nx.Graph(); G.add_nodes_from(range(int(d.num_nodes)))
    G.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))
    G.remove_edges_from(nx.selfloop_edges(G))
    y = d.y.numpy() if hasattr(d.y, 'numpy') else d.y
    for v in range(d.num_nodes):
        G.nodes[v]['group'] = int(y[v])
    if not nx.is_connected(G):
        lcc = max(nx.connected_components(G), key=len)
        G = G.subgraph(lcc).copy()
        G = nx.convert_node_labels_to_integers(G)
    return G


_RICE_MINORITY_DORMS = {3, 4, 9}

def _load_rice():
    """Rice-Facebook with binary group: 0=dorms {3,4,9}, 1=rest."""
    from src.env.graph_generators import load_rice_facebook
    G = load_rice_facebook()
    users_path = os.path.join(_DATA, "rice-facebook-undergrads-users.txt")
    user_dorm = {}
    if os.path.exists(users_path):
        with open(users_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    user_dorm[int(parts[0])] = int(parts[1])
    links_path = os.path.join(_DATA, "rice-facebook-undergrads-links.txt")
    nodes = list(G.nodes())
    if max(nodes) > len(nodes) and user_dorm:
        for v in G.nodes():
            d = user_dorm.get(v)
            G.nodes[v]['group'] = (0 if d in _RICE_MINORITY_DORMS else 1) if d is not None else 1
    elif user_dorm and os.path.exists(links_path):
        seen = []
        with open(links_path) as f:
            for line in f:
                parts = line.split()
                for uid in (int(parts[0]), int(parts[1])):
                    if uid not in seen: seen.append(uid)
        uid2node = {uid: i for i, uid in enumerate(seen)}
        for uid, dorm in user_dorm.items():
            nid = uid2node.get(uid)
            if nid is not None and G.has_node(nid):
                G.nodes[nid]['group'] = 0 if dorm in _RICE_MINORITY_DORMS else 1
    # Fallback: any unlabelled node → majority
    for v in G.nodes():
        if 'group' not in G.nodes[v]:
            G.nodes[v]['group'] = 1
    return G


NETWORKS = {
    "polblogs": _load_polblogs,
    "rice_fb":  _load_rice,
}


# ── per-method sweep for one (graph, kap, seed) ───────────────────────────────

def _run_cell(G, labels, groups, B, seed, cal, pol, ei, cache, device):
    envs = {
        "CGS":           _ep_cgs(G, B, seed, cal),
        "LSTM":          _ep_lstm(pol, G, cache, ei, B, seed, device),
        "Greedy+Budget": _ep_greedy(G, B, seed),
        "IE+Budget":     _ep_ie(G, B, seed),
        "DegreeBlind":   _ep_degree_blind(G, B, seed, cal),
    }
    out = {}
    for m, env in envs.items():
        gm = _group_metrics(env, labels, groups)
        gm["_total_rev"]    = _total_revenue(env)
        gm["_total_profit"] = _total_profit(env)
        out[m] = gm
    return out


# ── sanity checks ─────────────────────────────────────────────────────────────

def _sanity(net_name, G, labels, cal, pol, ei, cache, device):
    C = _get_C()
    B_unc = _get_B("unc")
    groups = sorted(set(labels.values()))
    print(f"\n=== SANITY: {net_name} unc Greedy seed=0 ===")
    env = _ep_greedy(G, B_unc, 0)
    log = getattr(env, '_offer_log', {})
    ac  = [v for v in env.S]
    first10 = [round(log.get(v, -1), 4) for v in list(log)[:10]]
    n_below_c = sum(1 for v in ac if log.get(v, 0.0) < C)
    sub_frac = n_below_c / max(len(ac), 1)
    print(f"  offers={len(env.offered)}  accepted={len(ac)}  B_final={env.B:.4f}")
    print(f"  first10 posted prices = {first10}")
    print(f"  subsidy_frac (price<c) = {sub_frac:.4f}  (must be > 0)")
    if sub_frac <= 0.0:
        print("  SANITY FAIL: subsidy_frac == 0 — greedy free seeds not captured")
        sys.exit(1)
    # Check UNUSABLE cells
    B_k5 = _get_B("k5")
    env_ctrl = _ep_degree_blind(G, B_k5, 0, cal)
    ctrl_reach = len(env_ctrl.S) / max(G.number_of_nodes(), 1)
    if ctrl_reach < 0.05:
        print(f"  NOTE: control reach at k5 = {ctrl_reach:.4f} < 5% → k5 cell UNUSABLE")
    print("  SANITY PASS")


# ── aggregate over seeds + compute gaps ──────────────────────────────────────

def _agg_seeds(seed_data, g0, g1):
    """seed_data: list of per-seed {method: {gk: metrics, _total_*}} dicts."""
    out = {}
    for m in METHOD_NAMES:
        def mv(gk, key):
            return [sd[m][gk][key] for sd in seed_data if m in sd]
        def tv(key):
            return [sd[m].get(key, 0.0) for sd in seed_data if m in sd]
        def s(vals): return (round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4))
        g0k, g1k = str(g0), str(g1)
        out[m] = {
            g0k: {k: s(mv(g0k, k)) for k in ["reach","price_mean","subsidy_frac","rev_share_norm"]},
            g1k: {k: s(mv(g1k, k)) for k in ["reach","price_mean","subsidy_frac","rev_share_norm"]},
            "total_rev":    s(tv("_total_rev")),
            "total_profit": s(tv("_total_profit")),
        }
    return out


def _gaps(agg, g0, g1):
    """Compute MIN-MAJ gap for reach and price, with control gaps."""
    g0k, g1k = str(g0), str(g1)
    ctrl = agg["DegreeBlind"]
    rows = []
    for m in METHOD_NAMES:
        if m not in agg: continue
        a = agg[m]
        for metric in ["reach_mean", "price_mean"]:
            mk = "reach" if "reach" in metric else "price_mean"
            method_gap = a[g0k][mk][0] - a[g1k][mk][0]
            ctrl_gap   = ctrl[g0k][mk][0] - ctrl[g1k][mk][0]
            excess     = method_gap - ctrl_gap
            rows.append({
                "method": m, "metric": mk,
                "MIN": a[g0k][mk][0], "MAJ": a[g1k][mk][0],
                "method_gap": round(method_gap, 4),
                "ctrl_gap":   round(ctrl_gap, 4),
                "excess":     round(excess, 4),
            })
    return rows


# ── main sweep ────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity-only", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C = _get_C()
    cal_cfg = BECfg(weight_low=0.0, weight_high=2.0, n_mc_samples=5)

    all_results = {}

    for net_name, loader_fn in NETWORKS.items():
        print(f"\n{'='*60}")
        print(f"Network: {net_name}")
        G = loader_fn()
        labels = {v: G.nodes[v]['group'] for v in G.nodes()}
        groups = sorted(set(labels.values()))   # [0, 1]
        g0, g1 = groups[0], groups[1]
        n0 = sum(1 for l in labels.values() if l == g0)
        n1 = sum(1 for l in labels.values() if l == g1)
        print(f"  n={G.number_of_nodes()}  m={G.number_of_edges()}"
              f"  g0(minority)={n0}  g1(majority)={n1}")

        # Calibrate + load model once per network
        cal = calibrate(G, cal_cfg)
        pol = load_arm_b(device)
        ei, cache = make_ei(G, device)

        # Sanity checks
        _sanity(net_name, G, labels, cal, pol, ei, cache, device)
        if args.sanity_only:
            continue

        net_results = {}
        for kap_name in KAPPAS_NAMES:
            B = _get_B(kap_name)
            print(f"\n  kap={kap_name}  B0={B:.4f}  C={C:.4f}", flush=True)
            seed_data = []
            for s in SEEDS:
                sd = _run_cell(G, labels, groups, B, s, cal, pol, ei, cache, device)
                seed_data.append(sd)
            agg = _agg_seeds(seed_data, g0, g1)
            gaps = _gaps(agg, g0, g1)
            net_results[kap_name] = {"agg": agg, "gaps": gaps}

            # Inline print
            ctrl_reach = agg["DegreeBlind"][str(g0)]["reach"][0]
            if ctrl_reach < 0.05:
                print(f"    [UNUSABLE: DegreeBlind reach={ctrl_reach:.3f}<0.05]")
                continue
            g0k, g1k = str(g0), str(g1)
            for m in METHOD_NAMES:
                a = agg[m]
                print(f"  [{net_name}/{kap_name}] {m:18s}"
                      f"  MIN reach={a[g0k]['reach'][0]:.3f}±{a[g0k]['reach'][1]:.3f}"
                      f"  price={a[g0k]['price_mean'][0]:.3f}"
                      f"  MAJ reach={a[g1k]['reach'][0]:.3f}"
                      f"  price={a[g1k]['price_mean'][0]:.3f}"
                      f"  rev={agg[m]['total_rev'][0]:.2f}", flush=True)

        all_results[net_name] = net_results

    if args.sanity_only:
        return

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n=== PRICE GAP (MIN-MAJ) + control excess ===")
    print(f"{'net':8s} {'kap':4s} {'method':18s}  {'MIN_price':9s} {'MAJ_price':9s}"
          f" {'gap':>7s}  {'ctrl_gap':>8s}  {'excess':>7s}")
    for net, nr in all_results.items():
        for kap, kr in nr.items():
            for row in kr["gaps"]:
                if row["metric"] != "price_mean": continue
                print(f"{net:8s} {kap:4s} {row['method']:18s}"
                      f"  {row['MIN']:9.4f} {row['MAJ']:9.4f}"
                      f"  {row['method_gap']:+7.4f}  {row['ctrl_gap']:+7.4f}"
                      f"  {row['excess']:+7.4f}")

    print("\n=== REACH GAP (MIN-MAJ) + control excess ===")
    print(f"{'net':8s} {'kap':4s} {'method':18s}  {'MIN_reach':9s} {'MAJ_reach':9s}"
          f" {'gap':>7s}  {'ctrl_gap':>8s}  {'excess':>7s}")
    for net, nr in all_results.items():
        for kap, kr in nr.items():
            for row in kr["gaps"]:
                if row["metric"] != "reach": continue
                print(f"{net:8s} {kap:4s} {row['method']:18s}"
                      f"  {row['MIN']:9.4f} {row['MAJ']:9.4f}"
                      f"  {row['method_gap']:+7.4f}  {row['ctrl_gap']:+7.4f}"
                      f"  {row['excess']:+7.4f}")

    print("\n=== LARGEST EXCESS PER (network, kappa) ===")
    for net, nr in all_results.items():
        for kap, kr in nr.items():
            rows = [r for r in kr["gaps"] if r["method"] != "DegreeBlind"]
            if not rows: continue
            best = max(rows, key=lambda r: abs(r["excess"]))
            print(f"  {net:8s} {kap:4s}: {best['method']:18s}"
                  f"  {best['metric']:12s}  excess={best['excess']:+.4f}"
                  f"  ({('MIN pays more' if best['method_gap']>0 else 'MAJ pays more') if best['metric']=='price_mean' else ('MIN less reached' if best['method_gap']<0 else 'MIN more reached')})")

    # ── Save + commit ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(all_results, open(OUT, "w"), indent=2)
    print(f"\nSaved → {OUT}")
    subprocess.run(["git", "add", "-f", OUT], cwd=_ROOT)
    subprocess.run(["git", "commit", "-m", "fairness_real_v2.json"], cwd=_ROOT)
    h = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True, cwd=_ROOT).stdout.strip()
    print(h)


if __name__ == "__main__":
    main()
