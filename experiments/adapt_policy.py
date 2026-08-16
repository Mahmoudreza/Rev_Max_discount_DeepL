#!/usr/bin/env python3
"""
experiments/adapt_policy.py — Block D1
=======================================
Fine-tune arm_b on a target network (information-parity test).
Resumes Phase-2 REINFORCE from arm_b (sha 0b549f93) with 25k on-graph episodes
(500 episodes × 50 per epoch = 25k), same LR/clip/entropy as Phase 2.
Saves adapted checkpoint as results/checkpoints/adapted_{NET}_{sha8}.pt.
Evaluates on 10 seeds (budget protocol) at k=[5,10,15,20,30,40].
Writes: results/logs/adapt_{NET}.json
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, sys, time
import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
_orig_bc = nx.betweenness_centrality
nx.betweenness_centrality = lambda G, normalized=True, **kw: _orig_bc(
    G, k=min(200, G.number_of_nodes()), normalized=normalized, **kw)

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.env.polblogs_loader import load_polblogs
from src.env.graph_generators import (
    generate_forest_fire, generate_modular_forest_fire, load_rice_facebook,
)
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.utils.features import (
    compute_static_features, build_graph_feature_cache, compute_node_features_fast,
)
from src.utils.helpers import graph_to_pyg_data, get_available_mask, set_seed

# ── Hyper-params (Phase 2 recipe) ─────────────────────────────────────────────
LR          = 5e-4
CLIP_GRAD   = 1.0
ENTROPY_C   = 0.005
N_EPOCHS    = 500    # 500 × 50 per-epoch budget = 25k episodes
EP_PER_EPOCH= 50
K_VALUES    = [5, 10, 15, 20, 30, 40]
C           = 0.3; W_HIGH = 1.0; N_EVAL = 10
CKPT_DIR    = "results/checkpoints"
ARM_B       = os.path.join(CKPT_DIR, "rev_gnn_lstm_densemix.pt")
ARM_B_SHA   = "0b549f93"
NETWORKS    = ["polblogs", "FF_1000", "Rice_FB", "Modular_FF", "FF_2000"]


def _sha8(p): return hashlib.sha256(open(p,"rb").read()).hexdigest()[:8]


def load_graph(net):
    if net == "polblogs":   return load_polblogs()
    if net == "FF_1000":    return generate_forest_fire(1000,0.37,0.32,seed=0)
    if net == "Rice_FB":    return load_rice_facebook()
    if net == "Modular_FF": return generate_modular_forest_fire([250,250],0.37,0.32,0.05,seed=0)
    if net == "FF_2000":    return generate_forest_fire(2000,0.37,0.32,seed=1)
    raise ValueError(net)


def load_fresh(device):
    assert _sha8(ARM_B) == ARM_B_SHA, f"sha mismatch: got {_sha8(ARM_B)}"
    enc  = GraphSAGEEncoder(in_dim=21, hidden_dim=64, n_layers=2)
    lstm = EpisodeLSTM(graph_dim=64, lstm_hidden=64, n_layers=1)
    pol  = SequentialJointPolicy(enc, lstm, gnn_dim=64, context_dim=64)
    pol.load_state_dict(torch.load(ARM_B, map_location="cpu"))
    pol.to(device); return pol


def run_train_ep(pol, graph, cache, B, seed, device, opt):
    """One REINFORCE episode; returns (loss, reward)."""
    set_seed(seed)
    n, nodes = graph.number_of_nodes(), list(graph.nodes())
    cfg = BudgetEnvConfig(budget_B=B, production_cost=C, seed=seed, weight_high=W_HIGH)
    env = BudgetRevenueEnv(graph, cfg); env.reset(); pol.reset_episode(device)
    log_probs = []; rewards = []
    for _ in range(n):
        avail = env.available_nodes
        if not avail: break
        feats = compute_node_features_fast(cache=cache, S=frozenset(env.S),
            offered=frozenset(env.offered), t=env.t, k=n, env=env)
        data = graph_to_pyg_data(graph, feats, device)
        mask = get_available_mask(n, frozenset(env.offered), nodes, device)
        nidx, disc, log_p = pol.select_and_price(data.x, data.edge_index, mask, greedy=False)
        if nidx not in avail: nidx = avail[0]
        _, rew, done, _ = env.step(nidx, disc)
        pol.update_sequence_state(disc, rew > 0, float(rew))
        log_probs.append(log_p); rewards.append(float(rew))
        if done: break
    R = float(env.total_revenue)
    if not log_probs: return 0., R
    baseline = getattr(pol, "_ema_baseline", 0.); alpha = 0.05
    pol._ema_baseline = alpha * R + (1-alpha) * baseline
    adv = torch.tensor(R - baseline, dtype=torch.float32, device=device)
    lps = torch.stack(log_probs)
    loss = -(lps * adv).mean() - ENTROPY_C * (-(lps.exp() * lps).sum())
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(pol.parameters(), CLIP_GRAD)
    opt.step()
    return float(loss.item()), R


def eval_k(pol, graph, cache, k, device):
    n, nodes = graph.number_of_nodes(), list(graph.nodes())
    B = k * C; revs = []
    for seed in range(N_EVAL):
        set_seed(seed)
        cfg = BudgetEnvConfig(budget_B=B, production_cost=C, seed=seed, weight_high=W_HIGH)
        env = BudgetRevenueEnv(graph, cfg); env.reset(); pol.reset_episode(device)
        with torch.no_grad():
            for _ in range(n):
                avail = env.available_nodes
                if not avail: break
                feats = compute_node_features_fast(cache=cache, S=frozenset(env.S),
                    offered=frozenset(env.offered), t=env.t, k=n, env=env)
                data = graph_to_pyg_data(graph, feats, device)
                mask = get_available_mask(n, frozenset(env.offered), nodes, device)
                nidx, disc, _ = pol.select_and_price(data.x, data.edge_index, mask, greedy=True)
                if nidx not in avail: nidx = avail[0]
                _, rew, done, _ = env.step(nidx, disc)
                pol.update_sequence_state(disc, rew > 0, float(rew))
                if done: break
        revs.append(float(env.total_revenue))
    return float(np.mean(revs)), float(np.std(revs))


def adapt_and_eval(net, device, out_dir):
    graph = load_graph(net)
    static = compute_static_features(graph)
    cache  = build_graph_feature_cache(graph, static)
    pol    = load_fresh(device)
    opt    = optim.Adam(pol.parameters(), lr=LR)
    k_train = 20  # mid-range k for training
    B_train = k_train * C
    print(f"\n=== adapt {net}: {N_EPOCHS} epochs × {EP_PER_EPOCH} episodes ===", flush=True)
    t_start = time.time()
    pol.train()
    for ep in range(N_EPOCHS):
        ep_rev = []
        for s in range(EP_PER_EPOCH):
            _, r = run_train_ep(pol, graph, cache, B_train, ep*EP_PER_EPOCH+s, device, opt)
            ep_rev.append(r)
        if ep % 50 == 0:
            print(f"  ep={ep:4d}  train_rev={np.mean(ep_rev):.1f}  "
                  f"t={time.time()-t_start:.0f}s", flush=True)
    # Save checkpoint
    pol.eval()
    out_ckpt_tmp = os.path.join(out_dir, f"adapted_{net}_tmp.pt")
    torch.save(pol.state_dict(), out_ckpt_tmp)
    sha = _sha8(out_ckpt_tmp)
    out_ckpt = os.path.join(out_dir, f"adapted_{net}_{sha}.pt")
    os.rename(out_ckpt_tmp, out_ckpt)
    print(f"  saved adapted ckpt: {out_ckpt}  sha8={sha}", flush=True)

    # Eval at all k
    eval_results = {}
    for k in K_VALUES:
        m, s = eval_k(pol, graph, cache, k, device)
        eval_results[k] = {"mean": round(m,2), "std": round(s,2)}
        print(f"  eval k={k}  {m:.1f}±{s:.1f}", flush=True)

    out = os.path.join("results/logs", f"adapt_{net}.json")
    os.makedirs("results/logs", exist_ok=True)
    with open(out, "w") as f:
        json.dump({"network": net, "adapted_sha": sha,
                   "adapted_ckpt": out_ckpt, "source_sha": ARM_B_SHA,
                   "n_epochs": N_EPOCHS, "ep_per_epoch": EP_PER_EPOCH,
                   "eval": {str(k):v for k,v in eval_results.items()}}, f, indent=2)
    print(f"Saved → {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks", nargs="+", default=NETWORKS)
    ap.add_argument("--ckpt-dir", default=CKPT_DIR)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    for net in args.networks:
        adapt_and_eval(net, device, args.ckpt_dir)


if __name__ == "__main__":
    main()
