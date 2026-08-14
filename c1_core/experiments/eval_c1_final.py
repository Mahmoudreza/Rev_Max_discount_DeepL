#!/usr/bin/env python3
"""
c1_core/experiments/eval_c1_final.py
=====================================
Re-generate all C1 (Contribution 1, unconstrained) results with full
checkpoint attribution.  Every output carries a populated `shas` dict
so numbers are traceable.

Protocols
---------
  P1: single-realization  seed=42
  P2: 5-seed              seeds [0..4]
  P3: 20-seed (FF_1000)   seeds [0..19]  — means, std, win_count, t, p

Networks
--------
  FF_500, FF_1000, FF_2000, Modular_FF, Rice_FB

Methods
-------
  Rev-GNN-LSTM  (results/checkpoints/rev_gnn_lstm.pt,   sha8=8fbc4648)
  Rev-GNN-IM-RL (results/checkpoints/rev_gnn_im_rl.pt,  sha8=computed)
  IE-Strategy, µ-Discount, Greedy-Discount  (stateless baselines)

Parallel usage
--------------
  bash c1_core/experiments/run_c1_final_parallel.sh

  Or per-shard:
    python eval_c1_final.py --shard 0 --config configs/experiments/rev_gnn_lstm.yaml
    ...
    python eval_c1_final.py --shard 4 --config configs/experiments/rev_gnn_lstm.yaml
    python eval_c1_final.py --merge   --config configs/experiments/rev_gnn_lstm.yaml

Output
------
  results/logs/c1_final_shard{N}.json  (intermediate)
  results/logs/c1_final.json           (merged, after --merge)
"""
import argparse, hashlib, json, math, os, sys, time
from pathlib import Path

import torch

# ── path setup ───────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
os.chdir(_REPO)

from src.utils.helpers import load_config_with_base, set_seed, ensure_dir
from src.utils.logging import ExperimentLogger
from src.evaluation.idea1_eval import (
    load_lstm_policy, load_im_policy,
    task1_robustness, task2_generalisation,
    task3_nonmonotone, task4_ablation,
)
from src.evaluation.baselines import ie_strategy, mu_discount, greedy_discount, _make_env
from src.env.graph_generators import generate_forest_fire, generate_modular_forest_fire, load_rice_facebook

# ── constants ────────────────────────────────────────────────────────────────
LSTM_CKPT = "results/checkpoints/rev_gnn_lstm.pt"
IM_CKPT   = "results/checkpoints/rev_gnn_im_rl.pt"

# 5 shards — one per network
SHARDS = [
    "FF_500",
    "FF_1000",
    "FF_2000",
    "Modular_FF",
    "Rice_FB",
]
SEED_P1 = 42          # single-realization seed
SEEDS_P2 = list(range(5))  # 5-seed
SEEDS_P3 = list(range(20)) # 20-seed (FF_1000 only)


# ── helpers ──────────────────────────────────────────────────────────────────

def sha8(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:8]


# Default FF params (standard for all C1 experiments)
_FF_P  = 0.37   # forward  burning probability
_FF_PB = 0.32   # backward burning probability
_MOD_SIZES     = [200, 300, 500]  # Modular_FF: 3 modules, 1000 nodes total
_MOD_INTER     = 0.01             # inter-module edge probability
_DATA_DIR      = "data/raw"       # Rice-FB data directory


def build_graph(network: str, seed: int, cfg):
    """Return nx.Graph for the requested network.

    FF params are read from cfg if available; fall back to module defaults.
    """
    p          = getattr(getattr(cfg, "graph", None), "p",  _FF_P)
    pb         = getattr(getattr(cfg, "graph", None), "pb", _FF_PB)
    data_dir   = getattr(getattr(cfg, "data",  None), "data_dir", _DATA_DIR)

    if network == "FF_500":
        return generate_forest_fire(500,  p=p, pb=pb, seed=seed)
    elif network == "FF_1000":
        return generate_forest_fire(1000, p=p, pb=pb, seed=seed)
    elif network == "FF_2000":
        return generate_forest_fire(2000, p=p, pb=pb, seed=seed)
    elif network == "Modular_FF":
        mod_sizes  = list(getattr(getattr(cfg, "graph", None), "module_sizes", _MOD_SIZES))
        inter_prob = getattr(getattr(cfg, "graph", None), "inter_module_prob", _MOD_INTER)
        return generate_modular_forest_fire(mod_sizes, p=p, pb=pb,
                                            inter_prob=inter_prob, seed=seed)
    elif network == "Rice_FB":
        return load_rice_facebook(data_dir=data_dir)
    else:
        raise ValueError(f"Unknown network: {network}")


def run_policy_ep(policy, graph, cfg, device, seed: int, is_lstm: bool) -> float:
    """Run one greedy episode; return total revenue."""
    set_seed(seed)
    env = _make_env(graph, cfg)
    obs = env.reset()
    if is_lstm:
        policy.reset_lstm()
    done = False
    with torch.no_grad():
        while not done:
            feat = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action = policy.act(feat, deterministic=True)
            obs, _, done, _ = env.step(action)
    return float(env.total_revenue)


def run_baseline_ep(fn, graph, cfg, seed: int) -> float:
    """Run one baseline episode (stateless, no policy)."""
    set_seed(seed)
    return float(fn(graph, cfg))


def t_test_paired(a, b):
    """Paired t-test; returns (t_stat, p_approx, win_count)."""
    n = len(a)
    diffs = [x - y for x, y in zip(a, b)]
    md = sum(diffs) / n
    sd = math.sqrt(sum((d - md) ** 2 for d in diffs) / (n - 1))
    t = md / (sd / math.sqrt(n)) if sd > 0 else float("inf")
    # Normal approx for p (adequate for n=20)
    z = abs(t)
    p = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
    wins = sum(1 for d in diffs if d > 0)
    return t, p, wins


# ── per-shard evaluation ─────────────────────────────────────────────────────

def run_shard(shard_id: int, cfg, device, lstm_policy, im_policy,
              lstm_sha: str, im_sha: str) -> dict:
    """Evaluate one network across all protocols and methods."""
    network = SHARDS[shard_id]
    print(f"\n=== Shard {shard_id}: {network} ===")
    t0 = time.time()

    # ── P1: single-realization seed=42 ──────────────────────────────────────
    graph_p1 = build_graph(network, SEED_P1, cfg)
    lstm_p1  = run_policy_ep(lstm_policy, graph_p1, cfg, device, SEED_P1, is_lstm=True)
    im_p1    = run_policy_ep(im_policy,   graph_p1, cfg, device, SEED_P1, is_lstm=False)
    ie_p1    = run_baseline_ep(ie_strategy,    graph_p1, cfg, SEED_P1)
    mu_p1    = run_baseline_ep(mu_discount,    graph_p1, cfg, SEED_P1)
    gd_p1    = run_baseline_ep(greedy_discount, graph_p1, cfg, SEED_P1)
    print(f"  P1 seed42: LSTM={lstm_p1:.2f}  IM={im_p1:.2f}  "
          f"IE={ie_p1:.2f}  µ={mu_p1:.2f}  GD={gd_p1:.2f}")

    # ── P2: 5-seed [0..4] ───────────────────────────────────────────────────
    lstm_p2_raw, im_p2_raw, ie_p2_raw, mu_p2_raw, gd_p2_raw = [], [], [], [], []
    for s in SEEDS_P2:
        g = build_graph(network, s, cfg)
        lstm_p2_raw.append(run_policy_ep(lstm_policy, g, cfg, device, s, True))
        im_p2_raw.append(run_policy_ep(im_policy,   g, cfg, device, s, False))
        ie_p2_raw.append(run_baseline_ep(ie_strategy,     g, cfg, s))
        mu_p2_raw.append(run_baseline_ep(mu_discount,     g, cfg, s))
        gd_p2_raw.append(run_baseline_ep(greedy_discount, g, cfg, s))
    lstm_p2m = sum(lstm_p2_raw)/5; lstm_p2s = math.sqrt(sum((x-lstm_p2m)**2 for x in lstm_p2_raw)/4)
    gd_p2m   = sum(gd_p2_raw)/5
    print(f"  P2 5-seed: LSTM={lstm_p2m:.2f}±{lstm_p2s:.2f}  GD={gd_p2m:.2f}  "
          f"margin={100*(lstm_p2m-gd_p2m)/gd_p2m if gd_p2m else 0:.1f}%")

    # ── P3: 20-seed FF_1000 only ─────────────────────────────────────────────
    p3_result = None
    if network == "FF_1000":
        lstm_p3_raw, im_p3_raw, gd_p3_raw = [], [], []
        for s in SEEDS_P3:
            g = build_graph("FF_1000", s, cfg)
            lstm_p3_raw.append(run_policy_ep(lstm_policy, g, cfg, device, s, True))
            im_p3_raw.append(run_policy_ep(im_policy,   g, cfg, device, s, False))
            gd_p3_raw.append(run_baseline_ep(greedy_discount, g, cfg, s))
        lstm_p3m = sum(lstm_p3_raw)/20; lstm_p3s = math.sqrt(sum((x-lstm_p3m)**2 for x in lstm_p3_raw)/19)
        im_p3m   = sum(im_p3_raw)/20;   im_p3s   = math.sqrt(sum((x-im_p3m)**2   for x in im_p3_raw)/19)
        gd_p3m   = sum(gd_p3_raw)/20
        t_stat, p_val, wins = t_test_paired(lstm_p3_raw, gd_p3_raw)
        t_stat2, p_val2, wins2 = t_test_paired(lstm_p3_raw, im_p3_raw)
        p3_result = {
            "lstm":  {"mean": lstm_p3m, "std": lstm_p3s, "all": lstm_p3_raw},
            "im_rl": {"mean": im_p3m,   "std": im_p3s,   "all": im_p3_raw},
            "greedy_discount": {"mean": gd_p3m, "all": gd_p3_raw},
            "lstm_vs_gd":  {"t": t_stat,  "p": p_val,  "wins": wins,  "n": 20},
            "lstm_vs_imrl":{"t": t_stat2, "p": p_val2, "wins": wins2, "n": 20},
        }
        print(f"  P3 20-seed FF_1000: LSTM={lstm_p3m:.2f}±{lstm_p3s:.2f}  "
              f"IM={im_p3m:.2f}  GD={gd_p3m:.2f}")
        print(f"    LSTM>GD: wins={wins}/20  t={t_stat:.3f}  p={p_val:.2e}")
        print(f"    LSTM>IM: wins={wins2}/20  t={t_stat2:.3f}  p={p_val2:.2e}")

    elapsed = time.time() - t0
    print(f"  Shard {shard_id} done in {elapsed:.1f}s")

    return {
        "network": network,
        "shas": {
            "rev_gnn_lstm":   {"path": LSTM_CKPT, "sha8": lstm_sha},
            "rev_gnn_im_rl":  {"path": IM_CKPT,   "sha8": im_sha},
        },
        "P1_seed42": {
            "seed": SEED_P1,
            "Rev-GNN-LSTM":  lstm_p1,
            "Rev-GNN-IM-RL": im_p1,
            "IE-Strategy":   ie_p1,
            "mu-Discount":   mu_p1,
            "Greedy-Discount": gd_p1,
        },
        "P2_5seed": {
            "seeds": SEEDS_P2,
            "Rev-GNN-LSTM":  {"mean": lstm_p2m, "std": lstm_p2s, "all": lstm_p2_raw},
            "Rev-GNN-IM-RL": {"mean": sum(im_p2_raw)/5, "all": im_p2_raw},
            "IE-Strategy":   {"mean": sum(ie_p2_raw)/5, "all": ie_p2_raw},
            "mu-Discount":   {"mean": sum(mu_p2_raw)/5, "all": mu_p2_raw},
            "Greedy-Discount":{"mean": gd_p2m, "all": gd_p2_raw},
        },
        "P3_20seed_FF1000": p3_result,
        "wall_s": elapsed,
    }


# ── merge shards ─────────────────────────────────────────────────────────────

def merge_shards(cfg_path: str):
    out = {
        "protocol": "C1 final — attributed re-run",
        "ckpt_lstm": LSTM_CKPT,
        "ckpt_imrl": IM_CKPT,
        "shas": {},
        "results": {},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    for i, net in enumerate(SHARDS):
        shard_path = f"results/logs/c1_final_shard{i}.json"
        if not os.path.exists(shard_path):
            print(f"  MISSING: {shard_path}")
            continue
        with open(shard_path) as f:
            d = json.load(f)
        out["results"][net] = d
        # Populate top-level shas from any shard (all same)
        if not out["shas"] and d.get("shas"):
            out["shas"] = d["shas"]

    # Flatten 20-seed result to top-level for convenience
    ff1k = out["results"].get("FF_1000", {})
    if ff1k.get("P3_20seed_FF1000"):
        out["FF_1000_20seed"] = ff1k["P3_20seed_FF1000"]

    out_path = "results/logs/c1_final.json"
    ensure_dir("results/logs")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nMerged → {out_path}")

    # Print summary tables
    print("\n── P1 single-realization (seed=42) ──")
    print(f"{'Network':<15} {'LSTM':>8} {'IM-RL':>8} {'IE':>8} {'µ':>8} {'GD':>8}")
    for net in SHARDS:
        r = out["results"].get(net, {}).get("P1_seed42", {})
        if r:
            print(f"{net:<15} {r.get('Rev-GNN-LSTM',0):>8.1f} "
                  f"{r.get('Rev-GNN-IM-RL',0):>8.1f} "
                  f"{r.get('IE-Strategy',0):>8.1f} "
                  f"{r.get('mu-Discount',0):>8.1f} "
                  f"{r.get('Greedy-Discount',0):>8.1f}")

    print("\n── P2 5-seed mean ──")
    print(f"{'Network':<15} {'LSTM':>8} {'IM-RL':>8} {'GD':>8} {'LSTM/GD':>8}")
    for net in SHARDS:
        r = out["results"].get(net, {}).get("P2_5seed", {})
        if r:
            lm = r.get("Rev-GNN-LSTM",{}).get("mean",0)
            im = r.get("Rev-GNN-IM-RL",{}).get("mean",0)
            gd = r.get("Greedy-Discount",{}).get("mean",0)
            pct = 100*(lm-gd)/gd if gd else 0
            print(f"{net:<15} {lm:>8.1f} {im:>8.1f} {gd:>8.1f} {pct:>+7.1f}%")

    p3 = out.get("FF_1000_20seed", {})
    if p3:
        print("\n── P3 20-seed FF_1000 ──")
        print(f"  LSTM={p3['lstm']['mean']:.2f}±{p3['lstm']['std']:.2f}  "
              f"IM-RL={p3['im_rl']['mean']:.2f}±{p3['im_rl']['std']:.2f}  "
              f"GD={p3['greedy_discount']['mean']:.2f}")
        vgd = p3["lstm_vs_gd"]
        vim = p3["lstm_vs_imrl"]
        print(f"  LSTM>GD:   wins={vgd['wins']}/20  t={vgd['t']:.3f}  p={vgd['p']:.2e}")
        print(f"  LSTM>IM-RL:wins={vim['wins']}/20  t={vim['t']:.3f}  p={vim['p']:.2e}")

    print(f"\n✓ c1_final.json  shas={list(out['shas'].keys())}")
    return out_path


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="C1 final eval with sha attribution")
    parser.add_argument("--config", default="configs/experiments/rev_gnn_lstm.yaml")
    parser.add_argument("--shard",  type=int, default=None,
                        help="0-4 (one network); omit for all shards sequentially")
    parser.add_argument("--merge",  action="store_true",
                        help="Merge shard JSONs into c1_final.json (no eval)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.merge:
        merge_shards(args.config)
        return

    # ── verify shas ──────────────────────────────────────────────────────────
    lstm_sha = sha8(LSTM_CKPT)
    im_sha   = sha8(IM_CKPT)
    print(f"rev_gnn_lstm.pt  sha8={lstm_sha}  (expected 8fbc4648)")
    print(f"rev_gnn_im_rl.pt sha8={im_sha}")
    if lstm_sha != "8fbc4648":
        print(f"  ⚠ LSTM sha mismatch!  got={lstm_sha}  want=8fbc4648")
        # Don't abort — record mismatch in output.

    cfg    = load_config_with_base(args.config)
    device = torch.device(args.device)
    ensure_dir("results/logs")
    logger = ExperimentLogger(cfg, run_name="c1_final_eval")

    lstm_policy = load_lstm_policy(LSTM_CKPT, cfg, device)
    im_policy   = load_im_policy(IM_CKPT,   cfg, device)
    lstm_policy.eval(); im_policy.eval()

    shards_to_run = [args.shard] if args.shard is not None else list(range(len(SHARDS)))

    for sid in shards_to_run:
        result = run_shard(sid, cfg, device, lstm_policy, im_policy, lstm_sha, im_sha)
        out_path = f"results/logs/c1_final_shard{sid}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  → {out_path}")

    if args.shard is None:
        # Sequential full run: also merge immediately
        merge_shards(args.config)

    logger.finish()


if __name__ == "__main__":
    main()
