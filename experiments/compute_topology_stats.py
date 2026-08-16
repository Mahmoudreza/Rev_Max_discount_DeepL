#!/usr/bin/env python3
"""
experiments/compute_topology_stats.py
======================================
Compute topology statistics for all 5 benchmark networks and save to
results/logs/network_topology_stats.json.

Stats: n, m, density, avg_degree, max_degree, clustering_coeff,
       n_components, diameter (of largest component),
       assortativity, avg_path_length (sampled).
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.polblogs_loader import load_polblogs
from src.env.graph_generators import (
    generate_forest_fire, generate_modular_forest_fire, load_rice_facebook,
)

NETWORKS = {
    "polblogs":   lambda: load_polblogs(),
    "FF_1000":    lambda: generate_forest_fire(1000, 0.37, 0.32, seed=0),
    "Rice_FB":    lambda: load_rice_facebook(),
    "Modular_FF": lambda: generate_modular_forest_fire([250,250], 0.37, 0.32, 0.05, seed=0),
    "FF_2000":    lambda: generate_forest_fire(2000, 0.37, 0.32, seed=1),
}


def stats(G: nx.Graph) -> dict:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    degs = [d for _, d in G.degree()]
    Gu = G.to_undirected() if G.is_directed() else G
    components = list(nx.connected_components(Gu))
    largest = Gu.subgraph(max(components, key=len))
    # Diameter: exact for small, sampled for large
    if n <= 500:
        try:
            diam = nx.diameter(largest)
        except Exception:
            diam = -1
    else:
        # BFS from 50 random nodes, take max eccentricity
        sample = np.random.choice(list(largest.nodes()), size=min(50, len(largest)), replace=False)
        diam = max(max(nx.single_source_shortest_path_length(largest, s).values()) for s in sample)
    # Average path length: sampled
    if n <= 300:
        try:
            apl = nx.average_shortest_path_length(largest)
        except Exception:
            apl = -1.0
    else:
        sample = np.random.choice(list(largest.nodes()), size=min(100, len(largest)), replace=False)
        lengths = []
        for s in sample:
            sp = nx.single_source_shortest_path_length(largest, s)
            lengths.extend(sp.values())
        apl = float(np.mean(lengths))
    try:
        assort = round(nx.degree_assortativity_coefficient(Gu), 4)
    except Exception:
        assort = None
    return {
        "n": n, "m": m,
        "density": round(2*m / (n*(n-1)), 6),
        "avg_degree": round(float(np.mean(degs)), 3),
        "max_degree": int(np.max(degs)),
        "std_degree": round(float(np.std(degs)), 3),
        "clustering_coeff": round(nx.average_clustering(Gu), 4),
        "n_components": len(components),
        "diameter": int(diam),
        "avg_path_length": round(apl, 3),
        "assortativity": assort,
    }


def main():
    np.random.seed(42)
    results = {}
    for name, loader in NETWORKS.items():
        t0 = time.time()
        G = loader()
        s = stats(G)
        print(f"{name}: n={s['n']} m={s['m']} avg_deg={s['avg_degree']:.1f} "
              f"clust={s['clustering_coeff']:.3f} diam={s['diameter']} "
              f"apl={s['avg_path_length']:.2f} ({time.time()-t0:.0f}s)")
        results[name] = s
    out = "results/logs/network_topology_stats.json"
    os.makedirs("results/logs", exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
