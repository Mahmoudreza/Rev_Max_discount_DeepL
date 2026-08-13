#!/usr/bin/env python3
"""merge_ksweep_shards.py — Merge per-worker JSON shards into budget_sweep_all_networks.json.

Steps:
  1. Glob all results/logs/budget_sweep_*.json except the final merged file.
  2. Merge all `results` dicts (net → k → cell) into one structure.
  3. Assert completeness: every (network, k) cell is present.
  4. Write results/logs/budget_sweep_all_networks.json.
  5. Run verify-parallel check: FF_1000 k=40 OURS from merged vs reference.

Usage:
    python -u experiments/merge_ksweep_shards.py 2>&1
    python -u experiments/merge_ksweep_shards.py --reference 473.0  # override ref
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np

EXPECTED_NETS = ["polblogs", "FF_1000", "Rice_FB", "Modular_FF", "FF_2000"]
EXPECTED_KS   = [5, 10, 15, 20, 30, 40]
MERGED_OUT    = "results/logs/budget_sweep_all_networks.json"
SHARD_GLOB    = "results/logs/budget_sweep_*.json"
EXCLUDE       = {"budget_sweep_all_networks.json"}


def _load_shards():
    paths = [p for p in sorted(glob.glob(SHARD_GLOB))
             if os.path.basename(p) not in EXCLUDE]
    if not paths:
        print("ERROR: no shard files found matching", SHARD_GLOB)
        sys.exit(1)
    print(f"Found {len(paths)} shard(s):", flush=True)
    merged = {}   # net → k → cell
    proto  = None
    shas   = None
    total_wall = 0.0
    for p in paths:
        d = json.load(open(p))
        print(f"  {os.path.basename(p)}: nets={d['protocol'].get('networks','?')} "
              f"k={d['protocol'].get('k_values','?')}", flush=True)
        if proto is None: proto = d["protocol"]
        if shas  is None: shas  = d.get("shas", {})
        total_wall += d.get("wall_s", 0.0)
        for net, kdict in d["results"].items():
            if net not in merged: merged[net] = {}
            for k, cell in kdict.items():
                k_int = int(k)
                if k_int in merged[net]:
                    print(f"  WARNING: duplicate cell ({net}, k={k_int}) — keeping first", flush=True)
                else:
                    merged[net][k_int] = cell
    return merged, proto, shas, total_wall


def _assert_complete(merged):
    print("\nCompleteness check:", flush=True)
    missing = []
    for net in EXPECTED_NETS:
        for k in EXPECTED_KS:
            if net not in merged or k not in merged[net]:
                missing.append(f"({net}, k={k})")
    if missing:
        print(f"  MISSING {len(missing)} cells:", flush=True)
        for m in missing: print(f"    {m}", flush=True)
        print("  ABORT: incomplete results — run missing shards first.", flush=True)
        sys.exit(1)
    else:
        print(f"  All {len(EXPECTED_NETS)*len(EXPECTED_KS)} cells present. ✓", flush=True)


def _verify_parallel(merged, ref_ff1000_k40_ours, tol=1.0):
    """Check FF_1000 k=40 OURS from merged vs reference value."""
    cell = merged.get("FF_1000", {}).get(40, {})
    val  = cell.get("ours")
    if val is None:
        print(f"\nVERIFY: FF_1000 k=40 OURS not found in merged results.", flush=True)
        return False
    diff = abs(val - ref_ff1000_k40_ours)
    ok   = diff <= tol
    print(f"\nVERIFY parallel vs reference:", flush=True)
    print(f"  FF_1000 k=40 OURS: merged={val:.2f}  ref={ref_ff1000_k40_ours:.2f}  "
          f"|diff|={diff:.3f}  tol={tol}  {'PASS ✓' if ok else 'FAIL ✗'}", flush=True)
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", type=float, default=None,
                   help="FF_1000 k=40 OURS reference value for parallel verify (default: from shard)")
    p.add_argument("--tol", type=float, default=1.0,
                   help="Tolerance for parallel verify (default: 1.0)")
    args = p.parse_args()

    merged, proto, shas, total_wall = _load_shards()
    _assert_complete(merged)

    # Determine reference for FF_1000 k=40 OURS
    ref = args.reference
    if ref is None:
        # Use the value from the merged shard itself (verify it's consistent)
        ref = merged["FF_1000"][40].get("ours")
        if ref is None:
            print("ERROR: --reference not given and FF_1000/k=40/ours not in shards")
            sys.exit(1)
        print(f"\n(reference=merged value {ref:.2f} — set --reference to override with sequential number)", flush=True)

    ok = _verify_parallel(merged, ref, args.tol)
    if not ok:
        print("ABORT: parallel result differs from reference by more than tol — results void.", flush=True)
        sys.exit(1)

    # Write merged output
    proto["k_values"]  = EXPECTED_KS
    proto["networks"]  = EXPECTED_NETS
    out = {
        "protocol": proto,
        "shas": shas,
        "results": {net: {str(k): merged[net][k] for k in EXPECTED_KS}
                    for net in EXPECTED_NETS},
        "wall_s": total_wall,
        "merge_note": "merged from parallel shards",
    }
    os.makedirs("results/logs", exist_ok=True)
    with open(MERGED_OUT, "w") as f: json.dump(out, f, indent=2)
    print(f"\nMerged → {MERGED_OUT}  ({total_wall:.0f}s total wall)", flush=True)


if __name__ == "__main__":
    main()
