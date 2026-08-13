#!/usr/bin/env python3
"""merge_all_methods_shards.py — merge per-network eval_all_methods_ksweep shards.

Expected input files (written by run_all_methods_parallel.sh workers):
  results/logs/budget_sweep_{NET}_k5-10-15-20-30-40.json  (one per network)

Output:
  results/logs/budget_sweep_all_networks.json

Also prints the final 9-method tables + SUMMARY LINES to stdout.
"""
import json, os, sys
import numpy as np

NETWORKS = ["polblogs", "FF_1000", "Rice_FB", "Modular_FF", "FF_2000"]
K_VALUES = [5, 10, 15, 20, 30, 40]
K_TAG    = "k" + "-".join(str(k) for k in K_VALUES)
IN_DIR   = "results/logs"
OUT_PATH = os.path.join(IN_DIR, "budget_sweep_all_networks.json")

METHODS = ["greedy_budget","ie_budget","caldp_composite","ours",
           "lstm_v1","arm_a","arm_b","c1_50_50","c1_2to1"]
MLABELS = ["Greedy+B","IE-Strat","Cal-DP","OURS",
           "lstm_v1","arm_a(unc)","arm_b(unc)","c1_50/50","c1_2:1"]

def load_shard(net):
    path = os.path.join(IN_DIR, f"budget_sweep_{net}_{K_TAG}.json")
    if not os.path.exists(path):
        print(f"  MISSING shard: {path}", file=sys.stderr)
        return None
    with open(path) as f:
        return json.load(f)

def main():
    all_results = {}
    protocol = None
    shas     = None
    total_wall = 0.0
    missing  = []

    for net in NETWORKS:
        shard = load_shard(net)
        if shard is None:
            missing.append(net)
            continue
        if protocol is None:
            protocol = shard.get("protocol", {})
            shas     = shard.get("shas", {})
        net_results = shard.get("results", {})
        if net not in net_results:
            print(f"  WARNING: shard for {net} has no results key '{net}'", file=sys.stderr)
            missing.append(net)
            continue
        # Keys in JSON are strings; convert k to int
        all_results[net] = {int(k): v for k, v in net_results[net].items()}
        total_wall += shard.get("wall_s", 0.0)
        print(f"  loaded shard: {net}  ({len(all_results[net])} k-values)")

    if missing:
        print(f"\nWARNING: missing shards for: {missing}", file=sys.stderr)

    if not all_results:
        print("ABORT: no shards found.", file=sys.stderr)
        sys.exit(1)

    # ── Print tables ──────────────────────────────────────────────────────
    k_filter = sorted({k for net in all_results for k in all_results[net]})
    for net_name in NETWORKS:
        if net_name not in all_results:
            continue
        print(f"\n── {net_name} ──")
        print(f"{'method':<14}", end="")
        for k in k_filter: print(f"  k={k:2d}", end="")
        print()
        print("─"*60)
        for m, ml in zip(METHODS, MLABELS):
            print(f"{ml:<14}", end="")
            for k in k_filter:
                v = all_results[net_name].get(k, {}).get(m, None)
                print(f"  {v:>5.0f}" if v is not None else "    --", end="")
            print()

    # ── Summary lines ─────────────────────────────────────────────────────
    computed_nets = [n for n in NETWORKS if n in all_results]
    print("\n\nSUMMARY LINES:")
    print("(a) Networks where OURS >= Greedy+Budget at EVERY k:")
    for net in computed_nets:
        if all(all_results[net].get(k,{}).get("ours",0) >=
               all_results[net].get(k,{}).get("greedy_budget",0) for k in k_filter):
            print(f"    {net}")
    print("(b) Networks where OURS >= Cal-DP at EVERY k:")
    for net in computed_nets:
        if all(all_results[net].get(k,{}).get("ours",0) >=
               all_results[net].get(k,{}).get("caldp_composite",0) for k in k_filter):
            print(f"    {net}")
    print("(c) (network,k) where arm_a OR arm_b beats BOTH Greedy+B AND Cal-DP:")
    found_c = False
    for net in computed_nets:
        for k in k_filter:
            r = all_results[net].get(k,{})
            for arm_lbl in ["arm_a","arm_b"]:
                av = r.get(arm_lbl)
                if av and av > r.get("greedy_budget",0) and av > r.get("caldp_composite",0):
                    print(f"    {net} k={k} {arm_lbl}: {av:.1f} > G={r['greedy_budget']:.1f} & D={r['caldp_composite']:.1f}")
                    found_c = True
    if not found_c: print("    None")
    if "polblogs" in all_results:
        print("(d) polblogs — OURS vs lstm_v1 at k∈{5,10,15}:")
        for k in [5,10,15]:
            if k in all_results["polblogs"]:
                r = all_results["polblogs"][k]
                ours = r.get("ours"); v1 = r.get("lstm_v1")
                if ours and v1:
                    print(f"    k={k}: OURS={ours:.1f}  lstm_v1={v1:.1f}  delta={ours-v1:+.1f}")

    print(f"\nTotal wall time (sum): {total_wall:.0f}s  ({total_wall/60:.1f} min)")
    print(f"Missing: {missing if missing else 'none'}")

    # ── Save merged result ────────────────────────────────────────────────
    os.makedirs(IN_DIR, exist_ok=True)
    out = {
        "protocol": protocol,
        "shas": shas,
        "results": {net: {str(k): v for k,v in kv.items()}
                    for net, kv in all_results.items()},
        "total_wall_s": total_wall,
        "missing_networks": missing,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nMerged → {OUT_PATH}")

if __name__ == "__main__":
    main()
