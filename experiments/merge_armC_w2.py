"""merge_armC_w2.py — merge per-network shards into results/logs/armC_w2.json"""
import json, os, sys, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NETS = ["FF_1000", "FF_2000", "Modular_FF", "Rice_FB", "polblogs"]
shards = {}
meta = None
for net in NETS:
    p = f"results/logs/armC_w2_{net}.json"
    assert os.path.exists(p), f"MISSING shard: {p}"
    d = json.load(open(p))
    if meta is None:
        meta = {k: v for k,v in d.items() if k not in ("network","results")}
    shards[net] = d["results"]
    print(f"  {net}: k={sorted(shards[net].keys())} ✓")
merged = {**meta, "networks": shards}
out = "results/logs/armC_w2.json"
with open(out, "w") as f:
    json.dump(merged, f, indent=2)
sha = hashlib.sha256(open(out,"rb").read()).hexdigest()[:8]
print(f"\nMerged {len(NETS)} networks → {out}  sha={sha}")
# Print polblogs k=15 diagnostic summary
try:
    pb = shards["polblogs"]["15"]
    print(f"\nDIAGNOSTIC polblogs k=15:")
    for m in ["arm_c","arm_b","CGS","IE+Budget"]:
        if m in pb:
            d = pb[m]
            pf = d.get("profit",{}).get("mean","?")
            bc = d.get("below_c_n",{}).get("mean","?")
            ns = d.get("n_in_S",{}).get("mean","?")
            rv = d.get("revenue",{}).get("mean","?")
            print(f"  {m:12s}: profit={pf:+.2f}  below_c={bc}  |S|={ns}  rev={rv}")
    pa = pb.get("paired",{})
    for label, key in [("ArmC vs ArmB","arm_c_vs_arm_b"),("ArmC vs IE","arm_c_vs_ie")]:
        pt = pa.get(key,{})
        print(f"  {label}: diff={pt.get('mean_diff','?')} CI={pt.get('ci','?')} p={pt.get('p','?')} {'NOT_SIG' if pt.get('not_sig') else 'SIG'}")
except Exception as e:
    print(f"  (diagnostic error: {e})")
