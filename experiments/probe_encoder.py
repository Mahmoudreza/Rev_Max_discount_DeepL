#!/usr/bin/env python3
"""probe_encoder.py — Linear probe: do frozen encoders predict buyer value?

For each encoder (LSTM 0b549f93, P1-ep200), for each graph (FF-1000, Rice-FB):
  Roll out 20 episodes. At every 50th step, record all remaining nodes:
    x = encoder h_i (64-dim)
    y = env._estimate_valuation(node_i)  ← MC value estimate
  Collect ~20K (x,y) pairs total.
  Fit Ridge on 80/20 train/test split. Report R^2.
Compare with (a) raw 21-dim features and (b) degree + bucket (2-dim).

Usage:
  venv/bin/python3 -u experiments/probe_encoder.py \
    --p1_ckpt results/checkpoints/c1_p1_s1_ep0200.pt \
    --device cpu > /tmp/probe.log 2>&1
"""
from __future__ import annotations
import argparse, hashlib, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

_ROOT = str(Path(__file__).parent.parent)
LSTM_CKPT = os.path.join(_ROOT, "results/checkpoints/rev_gnn_lstm_unified.pt")
LSTM_SHA  = "0b549f93"
C = 0.3; W_HIGH = 2.0; N_MC = 50  # 50 vs paper's 200 — noisier y, R^2 uniformly depressed


def _sha8(path): return hashlib.sha256(open(path,"rb").read()).hexdigest()[:8]

def _ei(G, device):
    m = {v: i for i, v in enumerate(G.nodes())}
    edges = list(G.edges())
    if not edges: return torch.zeros((2,0),dtype=torch.long,device=device)
    s=[m[u] for u,_ in edges]+[m[v] for _,v in edges]
    d=[m[v] for _,v in edges]+[m[u] for u,_ in edges]
    return torch.tensor([s,d],dtype=torch.long,device=device)


def _load_policy(ckpt_path, in_dim, device):
    from src.utils.helpers import load_config_with_base
    from src.models.encoders.graphsage import GraphSAGEEncoder
    cfg = load_config_with_base(
        os.path.join(_ROOT, "configs/experiments/rev_gnn_transformer_300ep.yaml"))
    H  = int(cfg.encoder.hidden_dim)
    NL = int(cfg.encoder.n_layers)
    DO = float(cfg.encoder.dropout)
    enc = GraphSAGEEncoder(in_dim, H, NL, DO)

    # Choose sequence encoder based on what the checkpoint contains.
    # Defer imports so a missing episode_lstm doesn't block transformer-only runs.
    sd_raw = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(sd_raw, dict) and "policy_state_dict" in sd_raw:
        sd_raw = sd_raw["policy_state_dict"]
    if isinstance(sd_raw, dict) and "state_dict" in sd_raw:
        sd_raw = sd_raw["state_dict"]
    has_transformer = any("transformer" in k for k in sd_raw)

    if has_transformer:
        from src.models.encoders.episode_transformer import EpisodeTransformerSliding
        from src.models.policies.transformer_joint_policy import TransformerJointPolicy
        tfm = EpisodeTransformerSliding.from_config(cfg.transformer)
        pol = TransformerJointPolicy(enc, tfm, gnn_dim=H,
                                     context_dim=tfm.context_dim).to(device)
    else:
        from src.models.encoders.episode_lstm import EpisodeLSTM
        from src.models.policies.joint_policy import JointPolicy
        lstm = EpisodeLSTM.from_config(cfg.lstm)
        pol = JointPolicy(enc, lstm, gnn_dim=H,
                          context_dim=lstm.context_dim).to(device)

    model_keys = set(pol.state_dict().keys())
    ckpt_keys  = set(sd_raw.keys())
    missing = model_keys - ckpt_keys
    unexpect = ckpt_keys - model_keys
    if missing:
        raise RuntimeError(f"strict=True ABORT: {len(missing)} keys missing "
                           f"from checkpoint, e.g. {sorted(missing)[:3]}")
    if unexpect:
        print(f"  WARN: {len(unexpect)} unexpected keys in ckpt (will be ignored by strict=True)")
    pol.load_state_dict(sd_raw, strict=True)
    pol.eval()
    print(f"  loaded {len(ckpt_keys)} keys strict=True  seq={'transformer' if has_transformer else 'lstm'}")
    return pol, H


def _collect(pol, G, cache, ei_t, n_episodes=20, sample_every=50, device="cpu"):
    """Collect (h, raw_feats, y_value, degree, bucket) tuples."""
    from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
    from src.utils.features import compute_node_features_fast
    from src.utils.helpers import set_seed
    H_vecs=[]; raw_vecs=[]; y_vals=[]; deg_vecs=[]; preln_vecs=[]
    nodes = list(G.nodes()); n = len(nodes)
    degrees = np.array([G.degree(v) for v in nodes], dtype=np.float32)
    deg_max = max(degrees) + 1e-9

    # Hook the LAST LayerNorm in the encoder to capture pre-LN activations.
    # If LN discards magnitude info (degree/influence signal), pre-LN R² >> post-LN R².
    pre_ln_buf = {}
    # Scope to pol.encoder (GNN) only — not the transformer stack.
    enc_module = getattr(pol, "encoder", pol)
    ln_mods = [(nm, m) for nm, m in enc_module.named_modules()
               if isinstance(m, torch.nn.LayerNorm)]
    _hook_handle = None
    if ln_mods:
        last_ln_name, last_ln = ln_mods[-1]
        print(f"  Hooking pre-LN at: {last_ln_name}")
        def _ln_hook(module, inp, out):
            pre_ln_buf['h'] = inp[0].detach().cpu().numpy()
        _hook_handle = last_ln.register_forward_hook(_ln_hook)

    for seed in range(n_episodes):
        set_seed(seed)
        k = 10; B0 = k * C
        cfg = BudgetEnvConfig(budget_B=B0, production_cost=C, seed=seed,
                              weight_high=W_HIGH, n_mc_samples=N_MC)
        env = BudgetRevenueEnv(G, cfg); env.reset()
        pol.reset_episode(device)

        for step in range(n):
            if not env.available_nodes: break
            avail = list(env.available_nodes)

            if step % sample_every == 0:
                # Compute encoder embeddings for ALL nodes
                feats = compute_node_features_fast(cache, env.S, env.offered, env.t, k, env)
                if feats.shape[1] == 20:
                    bcol = np.full((n,1), env.B/(40*C), dtype=np.float32)
                    feats21 = np.concatenate([feats, bcol], axis=1)
                else:
                    feats21 = feats.copy()
                    feats = feats[:, :20]
                x = torch.tensor(feats21, dtype=torch.float32, device=device)
                pre_ln_buf.clear()
                with torch.no_grad():
                    _, h_all, _, _ = pol.forward(
                        x, ei_t,
                        torch.ones(n, dtype=torch.bool, device=device))
                h_np = h_all.cpu().numpy()          # post-LN (n, H)
                preln_np = pre_ln_buf.get('h', None)  # pre-LN (n, H) or None

                sample = (avail if len(avail) <= 50
                          else list(np.random.choice(avail, 50, replace=False)))
                for idx in sample:  # random 50, not first-by-index
                    try:
                        y = env._estimate_valuation(nodes[idx])
                    except Exception:
                        continue
                    infl = env.get_current_influence(idx)
                    bkt  = 0 if infl < 2/6 else (1 if infl < 4/6 else 2)
                    H_vecs.append(h_np[idx])
                    if preln_np is not None and idx < len(preln_np):
                        preln_vecs.append(preln_np[idx])
                    raw_vecs.append(feats21[idx])
                    y_vals.append(float(y))
                    deg_vecs.append([degrees[idx]/deg_max, float(bkt)/2.0])

            # Step with first available node
            ni = avail[0]
            disc = 0.5
            _, _, done, _ = env.step(ni, disc)
            pol.update_sequence_state(disc, False, 0.0)
            if done: break

        if len(y_vals) >= 20000: break

    if _hook_handle: _hook_handle.remove()
    preln_arr = (np.array(preln_vecs, dtype=np.float32)
                 if preln_vecs else np.zeros((0, H_vecs[0].shape[0] if H_vecs else 1)))
    return (np.array(H_vecs, dtype=np.float32),
            np.array(raw_vecs, dtype=np.float32),
            np.array(y_vals,  dtype=np.float32),
            np.array(deg_vecs, dtype=np.float32),
            preln_arr)


def _probe(X, y, name):
    if len(X) < 20: return float("nan")
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=0)
    r = Ridge(alpha=1.0).fit(Xtr, ytr)
    r2 = float(r.score(Xte, yte))
    print(f"    {name:25s}  n={len(X):6d}  R²(ridge)={r2:+.4f}", end="")
    # MLP probe — tests non-linear decodability
    try:
        from sklearn.neural_network import MLPRegressor
        mlp = MLPRegressor(hidden_layer_sizes=(64,64), max_iter=500,
                           learning_rate_init=1e-3, random_state=0, verbose=False)
        mlp.fit(Xtr, ytr); r2_mlp = float(mlp.score(Xte, yte))
    except Exception:
        r2_mlp = float("nan")
    print(f"  R²(mlp)={r2_mlp:+.4f}")
    return r2, r2_mlp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1_ckpt", required=True)
    ap.add_argument("--device",  default="cpu")
    args = ap.parse_args()
    device = args.device

    from src.env.graph_generators import generate_forest_fire, load_rice_facebook
    from src.utils.features import compute_static_features, build_graph_feature_cache
    graphs = {"FF_1000": generate_forest_fire(1000,0.37,0.32,seed=0)}
    try:
        graphs["Rice_FB"] = load_rice_facebook()
    except Exception:
        print("Rice-FB not available — FF-1000 only")

    checkpoints = {
        "LSTM_0b549f93": (LSTM_CKPT, 20),
        "P1_ep200": (args.p1_ckpt, 21),
    }

    results = {}
    for enc_name, (ckpt_path, in_dim) in checkpoints.items():
        if not os.path.exists(ckpt_path):
            print(f"\n=== {enc_name}  SKIP: not found: {ckpt_path} ==="); continue
        sha = _sha8(ckpt_path)
        print(f"\n=== {enc_name}  sha={sha}  in_dim={in_dim} ===")
        try:
            pol, H = _load_policy(ckpt_path, in_dim, device)
        except Exception as e:
            print(f"  SKIP: load failed: {e}"); continue
        for gname, G in graphs.items():
            cache = build_graph_feature_cache(G, compute_static_features(G))
            ei_t  = _ei(G, device)
            print(f"  Collecting from {gname} n={G.number_of_nodes()} …", flush=True)
            H_vecs, raw_vecs, y_vals, deg_vecs, preln_vecs = _collect(
                pol, G, cache, ei_t, n_episodes=25, sample_every=50, device=device)
            print(f"  Collected {len(y_vals)} pairs  y=[{y_vals.min():.3f},{y_vals.max():.3f}]"
                  f"  NOTE: off-policy steps (avail[0]+disc=0.5); N_MC=50 not 200")
            print(f"  Probes:")
            r2_enc   = _probe(H_vecs,   y_vals, f"post-LN h ({H}-dim)")
            r2_preln = (_probe(preln_vecs, y_vals, f"pre-LN  h ({H}-dim)")
                        if len(preln_vecs) > 20 else (float("nan"), float("nan")))
            r2_raw   = _probe(raw_vecs, y_vals, f"raw_feats (21-dim)")
            r2_2     = _probe(deg_vecs, y_vals, f"degree+bucket (2-dim)")
            results[(enc_name, gname)] = {
                "enc_ridge": r2_enc[0] if isinstance(r2_enc, tuple) else r2_enc,
                "enc_mlp":   r2_enc[1] if isinstance(r2_enc, tuple) else float("nan"),
                "preln_ridge": r2_preln[0] if isinstance(r2_preln, tuple) else r2_preln,
                "preln_mlp":   r2_preln[1] if isinstance(r2_preln, tuple) else float("nan"),
                "raw_ridge": r2_raw[0] if isinstance(r2_raw, tuple) else r2_raw,
                "2d_ridge":  r2_2[0]  if isinstance(r2_2, tuple) else r2_2,
                "n": len(y_vals)}

    def _r(v, key): return v.get(key, float("nan"))
    print("\n=== SUMMARY (ridge / mlp) ===")
    hdr = (f"{'Encoder':20s}  {'Graph':10s}  "
           f"{'post-LN':>12s}  {'pre-LN':>12s}  "
           f"{'raw':>8s}  {'2d':>8s}  LN-culprit?")
    print(hdr)
    for (en,gn), v in results.items():
        er = _r(v,"enc_ridge"); em = _r(v,"enc_mlp")
        pr = _r(v,"preln_ridge"); pm = _r(v,"preln_mlp")
        rr = _r(v,"raw_ridge"); dr = _r(v,"2d_ridge")
        ln_culprit = (isinstance(pr,float) and isinstance(er,float)
                      and pr > er + 0.1)
        interp = ("LN-CULPRIT" if ln_culprit
                  else ("BOTTLENECK" if em < rr - 0.05
                        else ("OK" if em > 0.5 else "INFO-POOR")))
        print(f"  {en:20s}  {gn:10s}  "
              f"{er:+.3f}/{em:+.3f}  {pr:+.3f}/{pm:+.3f}  "
              f"{rr:+.3f}  {dr:+.3f}  {interp}")

if __name__ == "__main__":
    main()
