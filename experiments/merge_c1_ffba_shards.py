#!/usr/bin/env python3
"""merge_c1_ffba_shards.py — merge per-network C1 FFBA eval shards.

Input:  results/logs/c1_ffba_eval_{NET}.json  (one per network)
Output: results/logs/c1_ffba_eval.json  (merged) + final table + verdicts
"""
import json, os, sys
import numpy as np

NETWORKS   = ["FF_1000","FF_2000","Modular_FF","Rice_FB","polblogs"]
IN_DIR     = "results/logs"
OUT_PATH   = os.path.join(IN_DIR, "c1_ffba_eval.json")
FROZEN_REF = {"FF_1000":448.6,"FF_2000":915.0,"Modular_FF":414.4,"Rice_FB":214.1,"polblogs":374.2}

VERDICT_ROWS = ["c1_50_50_final","c1_2to1_final"]

def verdict(arm_5seed_means, net_names):
    beats = sum(1 for n in net_names
                if arm_5seed_means.get(n,0) > FROZEN_REF.get(n,float("inf")))
    if beats >= 4:   return "FIX CONFIRMED"
    elif beats >= 2: return "TRADE-OFF"
    else:            return "NO FIX"

def main():
    all_results = {}
    protocol    = None
    total_wall  = 0.0
    missing     = []

    for net in NETWORKS:
        path = os.path.join(IN_DIR, f"c1_ffba_eval_{net}.json")
        if not os.path.exists(path):
            print(f"  MISSING shard: {path}", file=sys.stderr)
            missing.append(net); continue
        with open(path) as f: shard = json.load(f)
        if protocol is None: protocol = shard.get("protocol",{})
        total_wall += shard.get("wall_s", 0.0)
        net_res = shard.get("results", {}).get(net)
        if net_res is None:
            print(f"  WARNING: shard {net} missing results key", file=sys.stderr)
            missing.append(net); continue
        all_results[net] = net_res
        print(f"  loaded: {net}")

    if missing: print(f"\nWARNING missing: {missing}", file=sys.stderr)
    if not all_results: print("ABORT: no shards.", file=sys.stderr); sys.exit(1)

    # ── Summary table (5-seed means) ─────────────────────────────────────
    net_names = [n for n in NETWORKS if n in all_results]
    # collect all row keys from first network
    all_keys = list(next(iter(all_results.values())).keys())
    row_order = (["frozen_ref","greedy_discount","ie_strategy"]
                 + [k for k in all_keys if k not in ["frozen_ref","greedy_discount","ie_strategy"]])

    print(f"\n{'model':<28}", end="")
    for n in net_names: print(f"  {n[:10]:>10}", end="")
    print()
    print("─"*90)
    for rk in row_order:
        print(f"{rk:<28}", end="")
        for net in net_names:
            cell = all_results.get(net,{}).get(rk)
            if cell is None: print(f"  {'--':>10}", end=""); continue
            v = cell.get("5seed",{}).get("mean") if "5seed" in cell else cell.get("mean")
            print(f"  {v:>10.1f}" if v is not None else f"  {'--':>10}", end="")
        print()

    # ── Discount stats for final arms ────────────────────────────────────
    print("\nDiscount stats (5-seed) — mean_disc | frac(d>0.9):")
    for key in VERDICT_ROWS:
        print(f"  {key}:")
        for net in net_names:
            cell = all_results.get(net,{}).get(key,{}).get("5seed",{})
            md   = cell.get("mean_disc","--")
            fd   = cell.get("frac_d_gt09","--")
            print(f"    {net}: disc={md}  frac(d>0.9)={fd}")

    # ── Verdicts ─────────────────────────────────────────────────────────
    print("\nVERDICTS (5-seed vs frozen_ref):")
    for key in VERDICT_ROWS:
        arm_means = {n: all_results.get(n,{}).get(key,{}).get("5seed",{}).get("mean",0)
                     for n in net_names}
        v = verdict(arm_means, net_names)
        beats = [f"{n}:{arm_means[n]:.1f}>ref:{FROZEN_REF[n]}"
                 for n in net_names if arm_means[n] > FROZEN_REF.get(n,float("inf"))]
        print(f"  {key:<28}  {v}  beats={beats or 'none'}")

    # ── Save ─────────────────────────────────────────────────────────────
    out = {
        "protocol": protocol,
        "frozen_ref": FROZEN_REF,
        "results": all_results,
        "total_wall_s": round(total_wall,1),
        "missing": missing,
    }
    os.makedirs(IN_DIR, exist_ok=True)
    with open(OUT_PATH,"w") as f: json.dump(out, f, indent=2, default=float)
    print(f"\nMerged → {OUT_PATH}  (wall={total_wall:.0f}s)")

if __name__ == "__main__":
    main()
