"""
Merge per-network Cal-DP observation-only shards into one JSON.
Asserts exactly 30 cells (5 networks x 6 k). Prints comparison table.
Does NOT modify budget_sweep_all_networks.json.
"""
import json, os, sys

NETWORKS  = ["polblogs", "FF_1000", "Rice_FB", "Modular_FF", "FF_2000"]
K_VALUES  = [5, 10, 15, 20, 30, 40]
SHARD_DIR = "results/logs"
OUT_PATH  = os.path.join(SHARD_DIR, "caldp_obs_all_networks.json")

# Previously published oracle-calibrated values for comparison
ORACLE = {
    "polblogs":   {5: 52.5,  10: 96.7,  15: 140.2, 20: 195.1, 30: 257.0, 40: 386.2},
    "FF_1000":    {5: 293.3, 10: 381.0, 15: None,  20: 387.7, 30: None,  40: 394.3},
    "Rice_FB":    {5: None,  10: None,  15: None,  20: None,  30: None,  40: None},
    "Modular_FF": {5: None,  10: None,  15: None,  20: None,  30: None,  40: None},
    "FF_2000":    {5: None,  10: None,  15: None,  20: None,  30: None,  40: None},
}

def main():
    merged = {}
    for net in NETWORKS:
        path = os.path.join(SHARD_DIR, f"caldp_obs_{net}.json")
        if not os.path.exists(path):
            print(f"MISSING shard: {path}", file=sys.stderr)
            sys.exit(1)
        shard = json.load(open(path))
        merged[net] = shard

    # Assert 30 cells
    cells = [(net, k) for net in NETWORKS for k in K_VALUES
             if str(k) in merged.get(net, {}).get("results", {})]
    if len(cells) != 30:
        missing = [(net, k) for net in NETWORKS for k in K_VALUES
                   if str(k) not in merged.get(net, {}).get("results", {})]
        print(f"ABORT: only {len(cells)}/30 cells present. Missing: {missing}",
              file=sys.stderr)
        sys.exit(1)
    print(f"All 30 cells present. Writing {OUT_PATH}")

    with open(OUT_PATH, "w") as f:
        json.dump(merged, f, indent=2)

    # Print comparison table
    header = f"{'network':12s}" + "".join(f"  k={k:2d}       " for k in K_VALUES)
    print(header)
    print("-" * len(header))
    for net in NETWORKS:
        res = merged[net]["results"]
        row = f"{net:12s}"
        for k in K_VALUES:
            comp = res[str(k)]["composite"]
            ref  = ORACLE.get(net, {}).get(k)
            ref_s = f"({ref:.1f})" if ref is not None else "  (n/a)"
            row += f"  {comp:6.1f}{ref_s:8s}"
        print(row)
    print(f"\nTable: Cal-DP obs composite; oracle reference in parentheses.")

if __name__ == "__main__":
    main()
