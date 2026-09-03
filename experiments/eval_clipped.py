#!/usr/bin/env python3
"""eval_clipped.py — Price-floor clipping: prevent below-cost offers.

Clipping rule (applied after policy's pricing head):
    v_hat_i = feats21[ni, 17]  # current MC valuation estimate
    d_max   = max(0.0, 1.0 - C / v_hat_i) if v_hat_i > C else 0.0
    d       = min(d_raw, d_max)

Evaluated on:
  - arm_b (rev_gnn_lstm_densemix.pt, sha=0b549f93) [SequentialJointPolicy]
  - transformer (transf_budget_s0_best.pt, sha=9a416465) [TransformerJointPolicy]
Both: clipped vs unclipped, on Rice-FB and Modular-FF.
Budget protocol: kappa in {5,10,20,40}, seeds [0..9], W_HIGH=2.0, N_MC=200.
NOTE: arm_b was trained with W_HIGH=1.0, N_MC=5 — evaluated OOD here.

Report per cell: profit mean±std, below-cost count, |S_T|, revenue mean±std.
Also: IE-aware baseline for comparison.

Usage:
  venv/bin/python3 -u experiments/eval_clipped.py \
    --transf_ckpt results/checkpoints/transf_budget_s0_best.pt \
    --device cuda:0 > /tmp/clipped.log 2>&1
"""
from __future__ import annotations
import argparse, hashlib, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

_ROOT  = str(Path(__file__).parent.parent)
C      = 0.3
W_HIGH = 2.0
N_MC   = 200
SEEDS  = list(range(10))
KAPPAS = [5, 10, 20, 40]
V_HAT_COL = 17   # current_valuation column in 21-dim feats (confirmed by sanity check)

ARM_B_CKPT = os.path.join(_ROOT, "results/checkpoints/rev_gnn_lstm_densemix.pt")
ARM_B_SHA  = "0b549f93"
ARM_K      = 50   # arm_b always uses k=50 for features (budget-blind training)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sha8(p): return hashlib.sha256(open(p,"rb").read()).hexdigest()[:8]

def _ei(G, device):
    m = {v: i for i, v in enumerate(G.nodes())}
    edges = list(G.edges())
    if not edges: return torch.zeros((2,0), dtype=torch.long, device=device)
    s = [m[u] for u,_ in edges] + [m[v] for _,v in edges]
    d = [m[v] for _,v in edges] + [m[u] for u,_ in edges]
    return torch.tensor([s,d], dtype=torch.long, device=device)

def _avail(env, n, device):
    mask = torch.zeros(n, dtype=torch.bool, device=device)
    for i in env.available_nodes: mask[i] = True
    return mask

def _clip(d_raw, v_hat_i):
    """Apply price-floor clip: d <= 1 - C/v_hat."""
    if v_hat_i > C:
        d_max = 1.0 - C / v_hat_i
    else:
        d_max = 0.0   # v_hat <= c: no discount (full price), buyer likely rejects
    return min(float(d_raw), max(0.0, d_max))

def _feats21_transf(cache, env, k):
    from src.utils.features import compute_node_features_fast
    base = compute_node_features_fast(cache, env.S, env.offered, env.t, k, env)
    bcol = np.full((cache["n"],1), env.B/(40*C), dtype=np.float32)
    return np.concatenate([base, bcol], axis=1)

def _feats21_armb(cache, env, n):
    """arm_b: always k=50, budget column = ones."""
    from src.utils.features import compute_node_features_fast
    base = compute_node_features_fast(cache, env.S, env.offered, env.t, ARM_K, env)
    return np.concatenate([base, np.ones((n,1), dtype=np.float32)], axis=1)


# ── Load policies ─────────────────────────────────────────────────────────────

def _load_arm_b(device):
    from src.models.encoders.graphsage import GraphSAGEEncoder
    from src.models.encoders.sequence_models import EpisodeLSTM
    from src.models.policies.sequential_joint_policy import SequentialJointPolicy
    sha = _sha8(ARM_B_CKPT)
    assert sha == ARM_B_SHA, f"arm_b sha={sha} != {ARM_B_SHA}"
    enc  = GraphSAGEEncoder(in_dim=21, hidden_dim=64, n_layers=2)
    lstm = EpisodeLSTM(graph_dim=64, lstm_hidden=64, n_layers=1)
    pol  = SequentialJointPolicy(enc, lstm, gnn_dim=64, context_dim=64)
    sd = torch.load(ARM_B_CKPT, map_location="cpu", weights_only=True)
    if "policy_state_dict" in sd: sd = sd["policy_state_dict"]
    elif "model_state_dict" in sd: sd = sd["model_state_dict"]
    pol.load_state_dict(sd, strict=True)
    print(f"arm_b loaded sha={sha}  params={sum(p.numel() for p in pol.parameters()):,}")
    return pol.eval().to(device)

def _load_transformer(ckpt_path, device):
    from src.utils.helpers import load_config_with_base
    from src.models.encoders.graphsage import GraphSAGEEncoder
    from src.models.encoders.episode_transformer import EpisodeTransformerSliding
    from src.models.policies.transformer_joint_policy import TransformerJointPolicy
    cfg  = load_config_with_base(
        os.path.join(_ROOT,"configs/experiments/rev_gnn_transformer_300ep.yaml"))
    H,NL,DO = int(cfg.encoder.hidden_dim),int(cfg.encoder.n_layers),float(cfg.encoder.dropout)
    enc = GraphSAGEEncoder(int(cfg.features.dim)+1, H, NL, DO)
    tfm = EpisodeTransformerSliding.from_config(cfg.transformer)
    pol = TransformerJointPolicy(enc, tfm, gnn_dim=H, context_dim=tfm.context_dim).to(device)
    sd  = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(sd,dict) and "state_dict" in sd: sd=sd["state_dict"]
    pol.load_state_dict(sd, strict=True)
    sha = _sha8(ckpt_path)
    print(f"transformer loaded sha={sha}  params={sum(p.numel() for p in pol.parameters()):,}")
    return pol.eval(), sha


# ── Episode rollout (generic) ─────────────────────────────────────────────────

def _rollout(pol, G, cache, ei_t, B0, k, seed, device, feat_fn, do_clip):
    """
    Returns (revenue, profit, n_below_cost, S_T)
    do_clip: if True, apply d_max = max(0, 1-C/v_hat_i) before step
    """
    from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
    from src.utils.helpers import set_seed
    set_seed(seed)
    n   = G.number_of_nodes()
    cfg = BudgetEnvConfig(budget_B=B0, production_cost=C, seed=seed,
                          weight_high=W_HIGH, n_mc_samples=N_MC)
    env = BudgetRevenueEnv(G, cfg); env.reset()
    pol.reset_episode(device)
    revenue=0.0; n_below=0; B_start=env.B

    while env.available_nodes and not env._check_bankrupt():
        f21 = feat_fn(cache, env, k)    # (n, 21); arm_b ignores 3rd arg (uses ARM_K=50)
        x   = torch.tensor(f21, dtype=torch.float32, device=device)
        av  = _avail(env, n, device)
        if not av.any(): break
        with torch.no_grad():
            sc, h, ctx, _ = pol.forward(x, ei_t, av)
        safe = sc.clone(); safe[~av] = -1e9
        ni   = int(safe.argmax())

        # Raw discount from policy
        d_raw = float(pol.get_discount_distribution(
            torch.cat([h[ni], ctx])).mean.clamp(1e-4, 1-1e-4))

        # Apply clipping if requested
        if do_clip:
            v_hat_i = float(f21[ni, V_HAT_COL])
            d = _clip(d_raw, v_hat_i)
        else:
            d = d_raw

        _, r, done, _ = env.step(ni, d)
        revenue += r
        if 0 < r < C: n_below += 1
        pol.update_sequence_state(d, r > 0, r)
        if done: break

    S_T    = len(env.S)
    profit = revenue - C * S_T
    # Budget check: profit = B_T - B_start
    pi_check = env.B - B_start
    if abs(profit - pi_check) > 0.05:
        print(f"  WARN budget check: profit={profit:.3f}  B_T-B0={pi_check:.3f}")
    return revenue, profit, n_below, S_T


# ── Sweep (all seeds, one k, one policy, clipped/unclipped) ──────────────────

def _sweep(pol, G, cache, ei_t, k, device, feat_fn, do_clip):
    B0=k*C; revs=[]; profs=[]; bcs=[]; s_ts=[]
    for seed in SEEDS:
        try:
            rev,prof,bc,s_t=_rollout(pol,G,cache,ei_t,B0,k,seed,device,feat_fn,do_clip)
        except Exception as e:
            print(f"  ERR seed={seed} k={k}: {e}"); rev=prof=bc=s_t=float("nan")
        revs.append(rev); profs.append(prof); bcs.append(bc); s_ts.append(s_t)
    pm,ps=float(np.nanmean(profs)),float(np.nanstd(profs))
    rm,rs=float(np.nanmean(revs)), float(np.nanstd(revs))
    return pm,ps,rm,rs,float(np.nanmean(bcs)),float(np.nanmean(s_ts))


# ── IE-aware baseline ─────────────────────────────────────────────────────────

def _ie_aware(G, k):
    B0=k*C
    try:
        from src.evaluation.ie_budget import ie_strategy_budget_aware
        ag = ie_strategy_budget_aware(G, B0, C, n_trials=len(SEEDS), weight_high=W_HIGH)
        rs = ag.get("revenue",{}).get("all",[])
        ss = ag.get("n_in_S",{}).get("all",[])
        profs=[r-C*s for r,s in zip(rs,ss)]
        return float(np.mean(profs)), float(np.std(profs))
    except Exception as e:
        return float("nan"), float("nan")


# ── Report helper ─────────────────────────────────────────────────────────────

def _row(label, pm,ps, rm,rs, bc, s_t):
    return (f"  {label:30s}  profit={pm:+7.2f}±{ps:.2f}  "
            f"below={bc:4.1f}  |S|={s_t:5.1f}  rev={rm:7.2f}±{rs:.2f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transf_ckpt", default=os.path.join(
        _ROOT,"results/checkpoints/transf_budget_s0_best.pt"))
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip_armb", action="store_true",
                    help="Skip arm_b if episode_lstm unavailable")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    from src.env.graph_generators import generate_modular_forest_fire, load_rice_facebook
    from src.utils.features import compute_static_features, build_graph_feature_cache

    graphs = {
        "Rice_FB":    load_rice_facebook(),
        "Modular_FF": generate_modular_forest_fire([250,250], 0.37, 0.32, 0.05, seed=0),
    }
    for name, G in graphs.items():
        print(f"{name}  n={G.number_of_nodes()}  m={G.number_of_edges()}")

    # Load policies
    pol_t, sha_t = _load_transformer(args.transf_ckpt, device)
    pol_b = None
    if not args.skip_armb:
        try:
            pol_b = _load_arm_b(device)
        except Exception as e:
            print(f"arm_b load failed: {e}  (re-run with --skip_armb to suppress)")

    # For arm_b feature function — wraps with k=50 always
    def feat_armb(cache, env, n): return _feats21_armb(cache, env, n)
    # For transformer feature function
    def feat_transf(cache, env, k): return _feats21_transf(cache, env, k)

    import json
    all_results = {}

    for gname, G in graphs.items():
        cache = build_graph_feature_cache(G, compute_static_features(G))
        ei_t  = _ei(G, device)
        n     = G.number_of_nodes()
        print(f"\n{'='*80}")
        print(f"Network: {gname}  n={n}")
        print(f"{'='*80}")

        net_res = {}
        for k in KAPPAS:
            ie_pm, ie_ps = _ie_aware(G, k)

            # Transformer unclipped
            t_u = _sweep(pol_t, G, cache, ei_t, k, device,
                         lambda c,e,kk,_c=cache: feat_transf(_c,e,kk), False)
            # Transformer clipped
            t_c = _sweep(pol_t, G, cache, ei_t, k, device,
                         lambda c,e,kk,_c=cache: feat_transf(_c,e,kk), True)

            print(f"\n  k={k}  IE-aware profit={ie_pm:+.2f}±{ie_ps:.2f}")
            print(_row(f"transf UNCLIPPED (sha={sha_t[:6]})", *t_u))
            print(_row(f"transf CLIPPED  (sha={sha_t[:6]})", *t_c))
            print(f"    clip Δprofit={t_c[0]-t_u[0]:+.2f}  Δbelow={t_c[4]-t_u[4]:+.1f}  ΔS_T={t_c[5]-t_u[5]:+.1f}")

            b_u = b_c = (float("nan"),)*6
            if pol_b is not None:
                # arm_b uses fixed k=50 features — pass n as 3rd arg via wrapper
                def _fb(cache_, env_, k_): return feat_armb(cache_, env_, n)
                b_u = _sweep(pol_b, G, cache, ei_t, k, device, _fb, False)
                b_c = _sweep(pol_b, G, cache, ei_t, k, device, _fb, True)
                print(_row(f"arm_b UNCLIPPED (sha=0b549f93)", *b_u))
                print(_row(f"arm_b CLIPPED   (sha=0b549f93)", *b_c))
                print(f"    clip Δprofit={b_c[0]-b_u[0]:+.2f}  Δbelow={b_c[4]-b_u[4]:+.1f}  ΔS_T={b_c[5]-b_u[5]:+.1f}")

            net_res[f"k{k}"] = {
                "ie_aware": {"profit_m": ie_pm, "profit_s": ie_ps},
                "transf_unclipped": {"pm":t_u[0],"ps":t_u[1],"rm":t_u[2],"rs":t_u[3],"bc":t_u[4],"s_t":t_u[5]},
                "transf_clipped":   {"pm":t_c[0],"ps":t_c[1],"rm":t_c[2],"rs":t_c[3],"bc":t_c[4],"s_t":t_c[5]},
                "armb_unclipped":   {"pm":b_u[0],"ps":b_u[1],"rm":b_u[2],"rs":b_u[3],"bc":b_u[4],"s_t":b_u[5]},
                "armb_clipped":     {"pm":b_c[0],"ps":b_c[1],"rm":b_c[2],"rs":b_c[3],"bc":b_c[4],"s_t":b_c[5]},
            }
        all_results[gname] = net_res

    out = os.path.join(_ROOT,"results/logs/eval_clipped.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"sha_t":sha_t,"W_HIGH":W_HIGH,"N_MC":N_MC,"C":C,"results":all_results},
              open(out,"w"), indent=2)
    print(f"\nSaved → {out}", flush=True)

    import subprocess
    subprocess.run(["git","add","-f",out], cwd=_ROOT)
    subprocess.run(["git","commit","-m",f"eval_clipped sha_t={sha_t}"], cwd=_ROOT)
    h = subprocess.run(["git","rev-parse","--short","HEAD"],
                        capture_output=True, text=True, cwd=_ROOT).stdout.strip()
    print(h, flush=True)


if __name__ == "__main__":
    main()
