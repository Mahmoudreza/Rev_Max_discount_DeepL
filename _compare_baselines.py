"""Quick baseline comparison across all available networks. Run once, not committed."""
import sys, pickle, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from omegaconf import OmegaConf
from src.env.graph_generators import (
    generate_forest_fire, generate_ba, generate_sbm, generate_power_law_cluster
)
from src.evaluation.baselines import (
    random_baseline, myopic_full_price, ie_strategy, mu_discount,
    sigma_discount, greedy_discount, prisca_baseline,
    hill_climbing_baseline, local_search_baseline,
)

cfg = OmegaConf.load("configs/base_config.yaml")

graphs = {}
with open("data/processed/rice_facebook.pkl", "rb") as f:
    graphs["rice_facebook (443)"] = pickle.load(f)
graphs["forest_fire (100)"] = generate_forest_fire(100, p=0.37, pb=0.32, seed=42)
graphs["ba_graph (100)"]    = generate_ba(100, m=3, seed=42)
graphs["sbm_3blk (120)"]    = generate_sbm(120, n_blocks=3, p_in=0.3, p_out=0.01, seed=42)
graphs["plc (100)"]         = generate_power_law_cluster(100, m=3, p=0.3, seed=42)

# Fast baselines: O(n log n) or O(n) — always run
FAST = [
    ("random",          random_baseline),
    ("myopic_full",     myopic_full_price),
    ("ie_strategy",     ie_strategy),
    ("mu_discount",     mu_discount),
    ("sigma_discount",  sigma_discount),
    ("greedy_discount", greedy_discount),
    ("prisca",          prisca_baseline),
]
# Slow baselines: O(k·n²) — only run on n <= 150
SLOW = [
    ("hill_climbing",   hill_climbing_baseline),
    ("local_search",    local_search_baseline),
]

all_results = {}

for net_name, G in graphs.items():
    n = G.number_of_nodes()
    results = {}
    t0 = time.time()
    print(f"\nRunning: {net_name} ...", flush=True)
    for name, fn in FAST:
        t1 = time.time()
        results[name] = fn(G, cfg)
        print(f"  {name:<22} {results[name]:>9.3f}  ({time.time()-t1:.1f}s)", flush=True)
    if n <= 150:
        for name, fn in SLOW:
            t1 = time.time()
            results[name] = fn(G, cfg)
            print(f"  {name:<22} {results[name]:>9.3f}  ({time.time()-t1:.1f}s)", flush=True)
    else:
        for name, _ in SLOW:
            results[name] = None
            print(f"  {name:<22} {'SKIPPED (n>150)':>9}", flush=True)
    all_results[net_name] = results
    print(f"  total: {time.time()-t0:.1f}s", flush=True)

# ── Final comparison table ──────────────────────────────────────────────────
print(f"\n{'='*90}")
print("  FINAL COMPARISON TABLE  (single trial, seed=42)")
print(f"{'='*90}")
nets = list(graphs.keys())
print(f"  {'Method':<22}", end="")
for nm in nets:
    short = nm.split("(")[0].strip()[:12]
    print(f"  {short:>13}", end="")
print()
print(f"  {'─'*84}")
all_keys = [k for k,_ in FAST + SLOW]
for k in all_keys:
    print(f"  {k:<22}", end="")
    for nm in nets:
        v = all_results[nm].get(k)
        if v is None:
            print(f"  {'—':>13}", end="")
        else:
            print(f"  {v:>13.2f}", end="")
    print()
print(f"{'='*90}")

# Mark best per network
print("\n  ★ Best hand-crafted per network:")
for nm in nets:
    res = all_results[nm]
    valid = {k: v for k, v in res.items() if v is not None}
    best_k = max(valid, key=valid.get)
    ie_r = res.get("ie_strategy", 1.0) or 1.0
    best_r = valid[best_k]
    pct = 100*(best_r - ie_r)/(ie_r+1e-9)
    print(f"    {nm:<25} → {best_k:<22}  {best_r:.2f}  ({pct:+.1f}% vs IE)")
