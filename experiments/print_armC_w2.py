"""print_armC_w2.py — print full table + paired tests from results/logs/armC_w2.json"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

F = "results/logs/armC_w2.json"
assert os.path.exists(F), f"MISSING: {F}"
d = json.load(open(F))

print(f"Arm C: {d.get('arm_c_ckpt','?')}  sha={d.get('arm_c_sha','?')}")
print(f"Arm B:  rev_gnn_lstm_densemix.pt    sha={d.get('arm_b_sha','?')}")
print(f"W_HIGH={d.get('W_HIGH','?')}  c={d.get('c','?')}  seeds={d.get('seeds','?')}\n")

METHODS = ["arm_c","arm_b","IE+Budget","Greedy","CGS"]
KS      = [5,10,15,20,30,40]
NETS    = list(d["networks"].keys())

def g(cell, method, key, sub="mean"):
    m = cell.get(method, {})
    v = m.get(key, {})
    if isinstance(v, dict): return v.get(sub, float("nan"))
    return float("nan")

def gs(cell, method, key):
    m = cell.get(method, {})
    v = m.get(key, {})
    if isinstance(v, dict):
        mn = v.get("mean", float("nan"))
        sd = v.get("std", float("nan"))
        return f"{mn:+7.2f}±{sd:.2f}"
    return "   N/A  "

for net in NETS:
    netd = d["networks"][net]
    print(f"\n{'='*100}")
    print(f"NETWORK: {net}")
    print(f"{'='*100}")
    hdr = f"{'Method':<12} {'k':>3} | {'Profit':>13} | {'below_c':>7} | {'|S_T|':>6} | {'Revenue':>13}"
    print(hdr)
    print("-"*len(hdr))
    for k in KS:
        cell = netd.get(str(k), netd.get(k, {}))
        for m in METHODS:
            prof = gs(cell, m, "profit")
            bc   = g(cell, m, "below_c_n")
            ns   = g(cell, m, "n_in_S")
            rev  = gs(cell, m, "revenue")
            bc_s = f"{bc:7.1f}" if bc == bc else "    N/A"
            ns_s = f"{ns:6.1f}" if ns == ns else "   N/A"
            print(f"{m:<12} {k:>3} | {prof} | {bc_s} | {ns_s} | {rev}")
        print()

# Paired tests
print(f"\n{'='*100}")
print("PAIRED TESTS (profit): Arm C vs Arm B  |  Arm C vs IE+Budget")
print(f"{'='*100}")
print(f"{'Network':<12} {'k':>3} | {'ArmC-ArmB diff':>14} CI95                  p      sig | {'ArmC-IE diff':>12} CI95                  p      sig")
print("-"*120)
for net in NETS:
    netd = d["networks"][net]
    for k in KS:
        cell = netd.get(str(k), netd.get(k, {}))
        pa = cell.get("paired", {})
        def fmt(key):
            pt = pa.get(key, {})
            diff = pt.get("mean_diff", float("nan"))
            ci   = pt.get("ci", [float("nan"), float("nan")])
            p    = pt.get("p", float("nan"))
            ns   = "NOT_SIG" if pt.get("not_sig", True) else "SIG    "
            if diff != diff: return "     N/A                              N/A   "
            return f"{diff:+8.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] p={p:.4f} {ns}"
        ab = fmt("arm_c_vs_arm_b")
        ie = fmt("arm_c_vs_ie")
        print(f"{net:<12} {k:>3} | {ab} | {ie}")
    print()
