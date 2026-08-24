#!/usr/bin/env python3
"""eval_transf_rice.py — Rice-FB budget eval: transformer vs all baselines.

Usage (after Phase 2 completes):
  venv/bin/python3 -u experiments/eval_transf_rice.py \
    --ckpt results/checkpoints/transf_budget_s9_best.pt \
    --device cuda:0 > /tmp/transf_rice.log 2>&1
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import scipy.stats as st
import torch

_ROOT = str(Path(__file__).parent.parent)
C = 0.3; W_HIGH = 2.0; N_MC = 200; SEEDS = list(range(10))
KAPPAS = [5, 10, 20, 40]
LSTM_CKPT = os.path.join(_ROOT, "results/checkpoints/rev_gnn_lstm_unified.pt")
LSTM_SHA  = "0b549f93"


# ── Load networks ─────────────────────────────────────────────────────────────
def _load_rice():
    from src.env.graph_generators import load_rice_facebook
    return load_rice_facebook()


# ── Feature helpers ───────────────────────────────────────────────────────────
def _features21(cache, env, k):
    from src.utils.features import compute_node_features_fast
    base = compute_node_features_fast(cache, env.S, env.offered, env.t, k, env)
    bcol = np.full((cache["n"], 1), env.B / (40 * C), dtype=np.float32)
    return np.concatenate([base, bcol], axis=1)

def _features20(cache, env, k):
    from src.utils.features import compute_node_features_fast
    return compute_node_features_fast(cache, env.S, env.offered, env.t, k, env)

def _ei(G, device):
    edges = list(G.edges())
    if not edges: return torch.zeros((2, 0), dtype=torch.long, device=device)
    m = {v: i for i, v in enumerate(G.nodes())}
    s = [m[u] for u, _ in edges] + [m[v] for _, v in edges]
    d = [m[v] for _, v in edges] + [m[u] for u, _ in edges]
    return torch.tensor([s, d], dtype=torch.long, device=device)

def _avail(env, n, device):
    mask = torch.zeros(n, dtype=torch.bool, device=device)
    for idx in env.available_nodes: mask[idx] = True
    return mask


# ── BudgetRevenueEnv rollout ──────────────────────────────────────────────────
def _make_env(G, B0, seed):
    from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
    from src.utils.helpers import set_seed
    set_seed(seed)
    cfg = BudgetEnvConfig(budget_B=B0, production_cost=C, seed=seed,
                          weight_high=W_HIGH, n_mc_samples=N_MC)
    env = BudgetRevenueEnv(G, cfg); env.reset(); return env


def _rollout(pol, G, cache, ei_t, B0, k, seed, device, feat_fn,
             degree_order=None):
    """Greedy rollout. degree_order: list of node indices for ablation."""
    env = _make_env(G, B0, seed)
    nodes = list(G.nodes()); n = len(nodes)
    pol.reset_episode(device)
    revenue = 0.0; n_below = 0; n_skips = 0
    B_start = env.B

    while env.available_nodes and not env._check_bankrupt():
        feats = feat_fn(cache, env, k)
        x  = torch.tensor(feats, dtype=torch.float32, device=device)
        av = _avail(env, n, device)
        if not av.any(): break
        with torch.no_grad():
            scores, h, ctx, _ = pol.forward(x, ei_t, av)

        if degree_order is not None:
            # Ablation: select by degree order among available
            ni = next((i for i in degree_order if av[i]), None)
            if ni is None: break
        else:
            safe = scores.clone(); safe[~av] = -1e9
            ni = int(safe.argmax())

        disc = float(pol.get_discount_distribution(
            torch.cat([h[ni], ctx])).mean.clamp(1e-4, 1 - 1e-4))

        _, r, done, _ = env.step(ni, disc)
        revenue += r
        if 0 < r < C: n_below += 1
        pol.update_sequence_state(disc, r > 0, r)
        if done: break

    # Count skips = nodes that were in available but env stepped without them
    S_T = len(env.S)
    profit = revenue - C * S_T
    # Assert Pi = B_T - B0
    B_T = env.B
    pi_check = B_T - B_start
    if abs(profit - pi_check) > 0.01:
        print(f"  ASSERT FAIL: profit={profit:.3f} B_T-B0={pi_check:.3f}", flush=True)
    return revenue, profit, n_below, S_T


# ── Load models ───────────────────────────────────────────────────────────────
def _load_transformer(ckpt_path, device):
    import hashlib
    from src.utils.helpers import load_config_with_base
    from src.models.encoders.graphsage import GraphSAGEEncoder
    from src.models.encoders.episode_transformer import EpisodeTransformerSliding
    from src.models.policies.transformer_joint_policy import TransformerJointPolicy
    sha = hashlib.sha256(open(ckpt_path,"rb").read()).hexdigest()[:8]
    cfg = load_config_with_base(
        os.path.join(_ROOT, "configs/experiments/rev_gnn_transformer_300ep.yaml"))
    in_dim = int(cfg.features.dim) + 1  # 21
    H = int(cfg.encoder.hidden_dim); NL = int(cfg.encoder.n_layers)
    DO = float(cfg.encoder.dropout)
    enc = GraphSAGEEncoder(in_dim, H, NL, DO)
    tfm = EpisodeTransformerSliding.from_config(cfg.transformer)
    pol = TransformerJointPolicy(enc, tfm, gnn_dim=H,
                                  context_dim=tfm.context_dim).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    pol.load_state_dict(sd, strict=True); pol.eval()
    print(f"Transformer loaded sha={sha}  in_dim=21  params={sum(p.numel() for p in pol.parameters()):,}")
    return pol, sha


def _load_lstm(device):
    import hashlib
    from src.utils.helpers import load_config_with_base
    from src.models.encoders.graphsage import GraphSAGEEncoder
    from src.models.encoders.episode_lstm import EpisodeLSTM
    from src.models.policies.joint_policy import JointPolicy
    sha = hashlib.sha256(open(LSTM_CKPT,"rb").read()).hexdigest()[:8]
    assert sha == LSTM_SHA, f"LSTM sha={sha} != {LSTM_SHA}"
    cfg = load_config_with_base(
        os.path.join(_ROOT, "configs/experiments/rev_gnn_lstm_unified.pt"))
    in_dim = int(cfg.features.dim)  # 20
    H = int(cfg.encoder.hidden_dim); NL = int(cfg.encoder.n_layers)
    DO = float(cfg.encoder.dropout)
    enc = GraphSAGEEncoder(in_dim, H, NL, DO)
    lstm = EpisodeLSTM.from_config(cfg.lstm)
    pol = JointPolicy(enc, lstm, gnn_dim=H, context_dim=lstm.context_dim).to(device)
    sd = torch.load(LSTM_CKPT, map_location=device, weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    pol.load_state_dict(sd, strict=True); pol.eval()
    print(f"LSTM loaded sha={sha}  in_dim=20  params={sum(p.numel() for p in pol.parameters()):,}")
    return pol


# ── Run all seeds for one method ──────────────────────────────────────────────
def _sweep(pol, G, cache, ei_t, k, device, feat_fn, degree_order=None):
    profs=[]; revs=[]; bcs=[]; sts=[]
    B0 = k * C
    for seed in SEEDS:
        try:
            rev, prof, bc, st = _rollout(pol, G, cache, ei_t, B0, k, seed,
                                          device, feat_fn, degree_order)
        except Exception as e:
            print(f"  ERR seed={seed} k={k}: {e}", flush=True)
            rev=prof=bc=st=float("nan")
        profs.append(prof); revs.append(rev); bcs.append(bc); sts.append(st)
    return (float(np.nanmean(profs)), float(np.nanstd(profs)),
            float(np.nanmean(bcs)),  float(np.nanmean(sts)),
            float(np.nanmean(revs)), float(np.nanstd(revs)),
            profs)


# ── Baselines ─────────────────────────────────────────────────────────────────
def _baselines(G, k):
    B0 = k * C
    out = {}

    # IE-aware
    try:
        from src.evaluation.ie_budget import ie_strategy_budget_aware
        ag = ie_strategy_budget_aware(G, B0, C, n_trials=len(SEEDS), weight_high=W_HIGH)
        rs = ag.get("revenue",{}).get("all",[])
        ss = ag.get("n_in_S",{}).get("all",[])
        profs = [r-C*s for r,s in zip(rs,ss)]
        out["IE_aware"] = {"profit_m": float(np.mean(profs)),
                           "profit_s": float(np.std(profs)),
                           "all": profs}
    except Exception as e:
        out["IE_aware"] = {"err": str(e)}

    # Greedy+Budget faithful
    try:
        from src.evaluation.greedy_budget_faithful import greedy_discount_budget_faithful
        gf = greedy_discount_budget_faithful(G, B0, C, n_trials=len(SEEDS), weight_high=W_HIGH)
        out["Greedy_faithful"] = {"profit_m": gf["profit"]["mean"],
                                   "profit_s": gf["profit"]["std"],
                                   "all": gf["profit"]["all"]}
    except Exception as e:
        out["Greedy_faithful"] = {"err": str(e)}

    # CGS (try known names)
    for fn_name in ["cal_dp_budget", "cgs_budget", "two_phase_dp_budget"]:
        try:
            from src.evaluation import budget_baselines as bb
            fn = getattr(bb, fn_name)
            ag2 = fn(G, B0, C, n_trials=len(SEEDS), weight_high=W_HIGH)
            rs2 = ag2.get("revenue",{}).get("all",[])
            ss2 = ag2.get("n_in_S",{}).get("all",[])
            profs2 = [r-C*s for r,s in zip(rs2,ss2)]
            out["CGS"] = {"profit_m": float(np.mean(profs2)),
                          "profit_s": float(np.std(profs2)),
                          "fn": fn_name, "all": profs2}
            break
        except Exception:
            pass
    if "CGS" not in out:
        out["CGS"] = {"err": "not found"}

    return out


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    from src.utils.features import compute_static_features, build_graph_feature_cache
    G = _load_rice()
    cache = build_graph_feature_cache(G, compute_static_features(G))
    ei_t  = _ei(G, device)
    deg_order = sorted(range(G.number_of_nodes()),
                       key=lambda i: G.degree(list(G.nodes())[i]), reverse=True)
    print(f"Rice-FB  n={G.number_of_nodes()}  m={G.number_of_edges()}", flush=True)

    pol_t, sha_t = _load_transformer(args.ckpt, device)
    try:
        pol_l = _load_lstm(device)
    except Exception as e:
        print(f"LSTM load failed: {e}"); pol_l = None

    all_res = {}
    hdr = (f"{'k':3s}  {'profit_t':>10s}±{'std':>5s}  {'below':>5s}  {'|S_T|':>5s}  "
           f"{'skips':>5s}  {'rev_t':>8s}  |  "
           f"{'LSTM':>8s}  {'IE_aw':>8s}  {'GrFa':>8s}  {'CGS':>8s}  "
           f"{'diff':>7s}  {'p':>6s}  {'sig?':>5s}")
    print(f"\n=== Rice-FB: Transformer sha={sha_t} vs baselines ===")
    print(hdr); print("-"*len(hdr))

    for k in KAPPAS:
        pm, ps, bc, s_t, rm, rs, profs_t = _sweep(pol_t, G, cache, ei_t, k,
                                                    device, _features21)
        # Ordering ablation
        pm_deg, ps_deg, _, _, _, _, profs_deg = _sweep(pol_t, G, cache, ei_t, k,
                                                        device, _features21,
                                                        degree_order=deg_order)

        lstm_pm = float("nan"); lstm_profs = []
        if pol_l is not None:
            lstm_pm, lstm_ps, _, _, _, _, lstm_profs = _sweep(
                pol_l, G, cache, ei_t, k, device, _features20)

        bl = _baselines(G, k)
        ie_pm  = bl["IE_aware"].get("profit_m", float("nan"))
        gf_pm  = bl["Greedy_faithful"].get("profit_m", float("nan"))
        cgs_pm = bl["CGS"].get("profit_m", float("nan"))

        # Paired t-test: transformer vs LSTM
        diff = float("nan"); pval = float("nan"); not_sig = True; ci = (float("nan"), float("nan"))
        if lstm_profs and len(lstm_profs) == len(profs_t):
            td, pval = st.ttest_rel(profs_t, lstm_profs, alternative="two-sided")
            diff = float(np.mean(profs_t)) - float(np.mean(lstm_profs))
            ci = st.t.interval(0.95, df=len(profs_t)-1, loc=diff,
                                scale=st.sem(np.array(profs_t) - np.array(lstm_profs)))
            not_sig = pval > 0.05

        print(f"  {k:2d}  {pm:8.2f}±{ps:5.2f}  {bc:5.1f}  {s_t:5.1f}  ?  {rm:8.2f}  |  "
              f"{lstm_pm:8.2f}  {ie_pm:8.2f}  {gf_pm:8.2f}  {cgs_pm:8.2f}  "
              f"{diff:+7.2f}  {pval:6.4f}  {'NS' if not_sig else 'SIG'}", flush=True)
        print(f"     ablation(deg-order): {pm_deg:8.2f}±{ps_deg:5.2f}  diff={pm-pm_deg:+.2f}  "
              f"CI95=[{(pm-pm_deg):.2f}±?]", flush=True)

        all_res[f"k{k}"] = {
            "transformer": {"profit": [pm, ps], "rev": [rm, rs], "bc": bc,
                            "s_t": s_t, "sha": sha_t, "all": profs_t},
            "transformer_deg": {"profit": [pm_deg, ps_deg], "all": profs_deg},
            "lstm": {"profit": lstm_pm, "all": lstm_profs},
            "ie_aware": bl["IE_aware"],
            "greedy_faithful": bl["Greedy_faithful"],
            "cgs": bl["CGS"],
            "paired_t_vs_lstm": {"diff": diff, "p": pval,
                                  "ci95": list(ci), "not_sig": not_sig},
        }

    out = os.path.join(_ROOT, "results/logs/transf_rice.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"network": "Rice_FB", "sha_t": sha_t, "results": all_res},
              open(out, "w"), indent=2)
    print(f"\nSaved → {out}", flush=True)

    subprocess.run(["git", "add", out, args.ckpt,
                    os.path.join(_ROOT, "results/checkpoints/README.md")],
                   cwd=_ROOT)
    subprocess.run(["git", "commit", "-m",
                    f"transf_rice eval sha={sha_t}"], cwd=_ROOT)
    h = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True, cwd=_ROOT).stdout.strip()
    print(h, flush=True)


if __name__ == "__main__":
    main()
