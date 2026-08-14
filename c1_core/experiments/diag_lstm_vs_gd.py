#!/usr/bin/env python3
"""
c1_core/experiments/diag_lstm_vs_gd.py
========================================
Diagnostic: cross-check LSTM vs GD on FF_1000 seed=42.
Compares the new eval_c1_final loop against idea1_eval reference.

Run on server:
  python c1_core/experiments/diag_lstm_vs_gd.py
"""
import sys, os, hashlib, torch
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
os.chdir(_REPO)

LSTM_CKPT = "results/checkpoints/rev_gnn_lstm.pt"
IM_CKPT   = "results/checkpoints/rev_gnn_im_rl.pt"

def sha8(p):
    h = hashlib.sha256()
    h.update(open(p, "rb").read())
    return h.hexdigest()[:8]

# ── 1. SHA check ──────────────────────────────────────────────────────────────
print("=" * 60)
print("1. CHECKPOINT SHAS")
s = sha8(LSTM_CKPT)
print(f"   rev_gnn_lstm.pt  sha8={s}  {'✓ MATCH' if s=='8fbc4648' else '✗ MISMATCH (want 8fbc4648)'}")
print(f"   rev_gnn_im_rl.pt sha8={sha8(IM_CKPT)}")

# ── 2. Load config ────────────────────────────────────────────────────────────
print("\n2. CONFIG")
from src.utils.helpers import load_config_with_base
cfg = load_config_with_base("configs/experiments/rev_gnn_lstm.yaml")
g = getattr(cfg, "graph", None)
print(f"   cfg.graph.p     = {getattr(g,'p','N/A')}")
print(f"   cfg.graph.pb    = {getattr(g,'pb','N/A')}")
print(f"   cfg.graph.n_nodes = {getattr(g,'n_nodes','N/A')}")
inf = getattr(cfg, "influence", None)
print(f"   cfg.influence.model = {getattr(inf,'model','N/A')}")
print(f"   cfg.influence.b     = {getattr(inf,'b','N/A')}")
print(f"   cfg.influence.n_mc_samples = {getattr(inf,'n_mc_samples','N/A')}")

# ── 3. Reference eval (idea1_eval) ────────────────────────────────────────────
print("\n3. REFERENCE EVAL via idea1_eval._eval_lstm_detailed  (FF_1000 seed=42)")
from src.evaluation.idea1_eval import load_lstm_policy, _eval_lstm_detailed, _eval_greedy_discount
from src.evaluation.baselines import greedy_discount
from src.env.graph_generators import generate_forest_fire

device = torch.device("cpu")
policy = load_lstm_policy(LSTM_CKPT, cfg, device)
policy.eval()

# Test both p=0.37/pb=0.32 (our module defaults) and p/pb from cfg
PARAM_SETS = [("defaults p=0.37 pb=0.32", 0.37, 0.32)]
if g is not None and hasattr(g, "p"):
    PARAM_SETS.append((f"cfg p={g.p} pb={g.pb}", float(g.p), float(g.pb)))

for label, p, pb in PARAM_SETS:
    G = generate_forest_fire(1000, p=p, pb=pb, seed=42)
    lstm_rev = _eval_lstm_detailed(policy, G, cfg, device)["revenue"]
    gd_ref   = _eval_greedy_discount(G, cfg)
    gd_base  = greedy_discount(G, cfg)
    print(f"\n   [{label}]")
    print(f"   LSTM  = {lstm_rev:.2f}   GD(idea1)={gd_ref:.2f}   GD(baselines)={gd_base:.2f}")
    print(f"   delta = LSTM-GD = {lstm_rev - gd_ref:+.2f}  {'LSTM wins ✓' if lstm_rev > gd_ref else 'GD wins — REGRESSION!'}")

# ── 4. Compare with known paper numbers ───────────────────────────────────────
print("\n4. KNOWN GOOD (paper_gen_updated.json, single-realization)")
print("   FF_500   LSTM=217.0  GD=209.9  delta=+7.1")
print("   FF_1000  LSTM=448.6  GD=417.7  delta=+30.9")
print("   FF_2000  LSTM=915.0  GD=839.2  delta=+75.8")
print("   Modular  LSTM=414.4  GD=356.6  delta=+57.8")
print("   Rice-FB  LSTM=214.1  GD=159.7  delta=+54.4")

# ── 5. Check if GD is stochastic (same graph, same cfg, different calls) ─────
print("\n5. GD REPRODUCIBILITY CHECK (same graph, 3 calls)")
G_fixed = generate_forest_fire(1000, p=0.37, pb=0.32, seed=42)
r1 = greedy_discount(G_fixed, cfg)
r2 = greedy_discount(G_fixed, cfg)
r3 = greedy_discount(G_fixed, cfg)
print(f"   GD call1={r1:.4f}  call2={r2:.4f}  call3={r3:.4f}")
print(f"   Deterministic: {'YES ✓' if r1==r2==r3 else 'NO — stochastic! Check n_mc_samples'}")

print("\n" + "=" * 60)
