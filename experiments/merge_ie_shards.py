#!/usr/bin/env python3
"""merge_ie_shards.py — Merge IE+Budget shard JSONs into budget_sweep_all_networks.json.

Steps:
  1. Glob results/logs/budget_sweep_IE_*.json shards.
  2. Assert all 30 IE cells (5 nets × 6 k-values) are present; abort if any missing.
  3. Load results/logs/budget_sweep_all_networks.json (existing sweep).
  4. Inject ie_budget key into each [net][k] cell.
  5. Write back to budget_sweep_all_networks.json.
  6. Verify-parallel: rerun FF_1000 k=20 sequentially, report actual delta.

Usage:
  python -u experiments/merge_ie_shards.py
"""
from __future__ import annotations
import glob, json, os, sys, time
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_orig_bc = nx.betweenness_centrality
nx.betweenness_centrality = lambda G, normalized=True, **kw: _orig_bc(
    G, k=min(200, G.number_of_nodes()), normalized=normalized, **kw)

EXPECTED_NETS = ["polblogs", "FF_1000", "Rice_FB", "Modular_FF", "FF_2000"]
EXPECTED_KS   = [5, 10, 15, 20, 30, 40]
MERGED        = "results/logs/budget_sweep_all_networks.json"
SHARD_GLOB    = "results/logs/budget_sweep_IE_*.json"
METHOD        = "ie_budget"


def load_shards():
    """Load and concatenate all IE shard JSONs."""
    shards = glob.glob(SHARD_GLOB)
    if not shards:
        print("ABORT: no IE shard files found matching", SHARD_GLOB)
        sys.exit(1)
    print(f"[merge] Found {len(shards)} IE shard(s):")
    merged: dict = {}
    for path in sorted(shards):
        print(f"  {os.path.basename(path)}")
        with open(path) as f:
            data = json.load(f)
        for net, k_dict in data["results"].items():
            if net not in merged:
                merged[net] = {}
            for k_str, cell in k_dict.items():
                k = int(k_str)
                if k not in merged[net]:
                    merged[net][k] = {}
                merged[net][k].update(cell)
    return merged


def assert_completeness(ie_data: dict):
    """Abort if any of the 30 expected cells is missing."""
    missing = []
    for net in EXPECTED_NETS:
        for k in EXPECTED_KS:
            if net not in ie_data or k not in ie_data[net] or METHOD not in ie_data[net][k]:
                missing.append((net, k))
    if missing:
        print(f"ABORT: {len(missing)} IE cells missing:")
        for net, k in missing:
            print(f"  {net}  k={k}")
        sys.exit(1)
    print(f"[merge] All 30 IE cells present ✓")


def inject_into_merged(ie_data: dict, merged_path: str):
    """Inject ie_budget into each cell in the existing merged JSON."""
    if not os.path.exists(merged_path):
        print(f"ABORT: merged file not found: {merged_path}")
        sys.exit(1)
    with open(merged_path) as f:
        merged = json.load(f)

    injected = 0
    for net in EXPECTED_NETS:
        if net not in merged.get("results", {}):
            print(f"WARNING: {net} not in merged results — skipping")
            continue
        for k in EXPECTED_KS:
            k_str = str(k)
            if k_str not in merged["results"][net]:
                print(f"WARNING: {net} k={k} not in merged results — skipping")
                continue
            merged["results"][net][k_str][METHOD] = ie_data[net][k][METHOD]
            injected += 1

    with open(merged_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"[merge] Injected {injected} IE cells into {merged_path}")
    return merged


def verify_parallel(ie_data: dict, net="FF_1000", k=20):
    """Rerun FF_1000 k=20 sequentially; report delta vs parallel value."""
    from src.env.graph_generators import generate_forest_fire
    from src.evaluation.ie_budget import ie_strategy_budget, IE_K_SEEDS

    C = 0.3; W = 2.0; N_TRIALS = 3
    print(f"\n[verify] Sequential rerun: {net} k={k}  B={k*C:.1f}", flush=True)
    G = generate_forest_fire(1000, 0.37, 0.32, seed=0)
    t0 = time.time()
    res = ie_strategy_budget(G, B=k*C, c=C, k_seeds=IE_K_SEEDS,
                              n_trials=N_TRIALS, weight_high=W)
    seq_rev = float(res["revenue"]["mean"])
    par_rev = float(ie_data[net][k][METHOD])
    delta   = abs(seq_rev - par_rev)
    elapsed = time.time() - t0
    print(f"[verify] {net} k={k}  parallel={par_rev:.2f}  sequential={seq_rev:.2f}  "
          f"delta={delta:.3f}  tol=1.0  {'PASS' if delta <= 1.0 else 'FAIL'}  ({elapsed:.0f}s)")
    return delta


def main():
    ie_data = load_shards()
    assert_completeness(ie_data)
    merged  = inject_into_merged(ie_data, MERGED)

    # Print IE+Budget table
    print("\n── IE+Budget raw revenue ──")
    print(f"{'network':<12}", end="")
    for k in EXPECTED_KS:
        print(f"  k={k:2d}", end="")
    print()
    print("─" * 60)
    for net in EXPECTED_NETS:
        print(f"{net:<12}", end="")
        for k in EXPECTED_KS:
            v = ie_data[net][k][METHOD]
            print(f"  {v:5.1f}", end="")
        print()

    verify_parallel(ie_data)


if __name__ == "__main__":
    main()
