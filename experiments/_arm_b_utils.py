"""
experiments/_arm_b_utils.py — shared arm_b eval helpers.
Mirrors the inference path used in eval_all_methods_ksweep.py exactly.

arm_b (rev_gnn_lstm_densemix.pt, sha=0b549f93):
  - Budget-blind (unconstrained training): uses _feat_unconstrained
  - Features: compute_node_features_fast(cache, S, offered, t, k=50, env) + ones column
  - in_dim = 21  (20 base + 1 dummy budget col)
  - Policy forward: sc, h, ctx, _ = policy.forward(x, ei, av)
  - Discount:  policy.get_discount_distribution(cat([h[ni], ctx])).mean.item()
  - update_sequence_state(d, info["accepted"], info.get("revenue_step", 0.0))
"""
from __future__ import annotations
import hashlib, os, sys
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)

from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.utils.features import compute_node_features_fast, compute_static_features, build_graph_feature_cache
from src.utils.helpers import graph_to_pyg_data, set_seed

N_MC   = 200    # BudgetEnvConfig n_mc_samples — must match paper (Def 2.1)
ARM_K  = 50     # k_feat used during arm_b training (unconstrained, k=50)
C      = 0.3
W_HIGH = 2.0    # Uniform(0, W_HIGH) per Definition 2.1
assert W_HIGH == 2.0, f"W_HIGH must be 2.0 per Def 2.1; got {W_HIGH}"

ARM_B_CKPT = os.path.join(_ROOT, "results/checkpoints/rev_gnn_lstm_densemix.pt")
ARM_B_SHA  = "0b549f93"


def _sha8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


def load_arm_b(device):
    """Load arm_b with correct architecture and assert sha."""
    sha = _sha8(ARM_B_CKPT)
    assert sha == ARM_B_SHA, f"arm_b sha mismatch: got {sha}"
    enc  = GraphSAGEEncoder(in_dim=21, hidden_dim=64, n_layers=2)
    lstm = EpisodeLSTM(graph_dim=64, lstm_hidden=64, n_layers=1)
    pol  = SequentialJointPolicy(enc, lstm, gnn_dim=64, context_dim=64)
    sd = torch.load(ARM_B_CKPT, map_location="cpu")
    if "policy_state_dict" in sd: sd = sd["policy_state_dict"]
    elif "model_state_dict" in sd: sd = sd["model_state_dict"]
    pol.load_state_dict(sd, strict=True)
    return pol.eval().to(device)


def make_ei(graph, device):
    """Precompute edge_index tensor for graph."""
    static = compute_static_features(graph)
    cache  = build_graph_feature_cache(graph, static)
    dummy  = np.zeros((graph.number_of_nodes(), 21), dtype=np.float32)
    data   = graph_to_pyg_data(graph, dummy, device)
    return data.edge_index, cache


def _feat_unconstrained(cache, env, n):
    """arm_b feature function: 20-dim base features + ones budget dummy."""
    base = compute_node_features_fast(cache, env.S, env.offered, env.t, ARM_K, env)
    return np.concatenate([base, np.ones((n, 1), dtype=np.float32)], axis=1)


def _avail_mask(env, n, device):
    m = torch.zeros(n, dtype=torch.bool, device=device)
    for i in env.available_nodes:
        m[i] = True
    return m


@torch.no_grad()
def eval_arm_b_episode(pol, graph, cache, ei, B, seed, device):
    """
    Run one arm_b episode on BudgetRevenueEnv.
    Returns (total_revenue, n_in_S, n_below_cost).
    """
    from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
    set_seed(seed)
    n = graph.number_of_nodes()
    cfg = BudgetEnvConfig(budget_B=B, production_cost=C, seed=seed,
                          weight_high=W_HIGH, n_mc_samples=N_MC)
    env = BudgetRevenueEnv(graph, cfg)
    env.reset()
    pol.reset_episode(device)
    n_below = 0
    while env.available_nodes and not env._check_bankrupt():
        x  = torch.FloatTensor(_feat_unconstrained(cache, env, n)).to(device)
        av = _avail_mask(env, n, device)
        if not av.any(): break
        sc, h, ctx, _ = pol.forward(x, ei, av)
        ni = int(sc.argmax().item())
        d  = float(pol.get_discount_distribution(
             torch.cat([h[ni], ctx])).mean.item())
        _, r, done, info = env.step(ni, d)
        if 0 < r < C: n_below += 1
        pol.update_sequence_state(d, info["accepted"],
                                  info.get("revenue_step", 0.0))
        if done: break
    return float(env.total_revenue), len(env.S), n_below


def eval_arm_b_k(pol, graph, cache, ei, B, n_trials, device):
    """Run arm_b for seeds 0..n_trials-1. Returns list of revenues (legacy)."""
    return [eval_arm_b_episode(pol, graph, cache, ei, B, s, device)[0]
            for s in range(n_trials)]


def eval_arm_b_k_full(pol, graph, cache, ei, B, n_trials, device):
    """Run arm_b for seeds 0..n_trials-1.
    Returns (revenues, n_in_S_list, n_below_list, profits)."""
    revs, s_ts, bcs = [], [], []
    for s in range(n_trials):
        rev, n_s, bc = eval_arm_b_episode(pol, graph, cache, ei, B, s, device)
        revs.append(rev); s_ts.append(n_s); bcs.append(bc)
    profs = [r - C * s for r, s in zip(revs, s_ts)]
    return revs, s_ts, bcs, profs


@torch.no_grad()
def eval_arm_b_episode_unc(pol, graph, cache, ei, seed, device):
    """
    Run one arm_b episode on RevenueEnv (unconstrained).
    Returns total_revenue (float).
    """
    from src.env.revenue_env import RevenueEnv, RevenueEnvConfig
    set_seed(seed)
    n = graph.number_of_nodes()
    env = RevenueEnv(graph, RevenueEnvConfig(seed=seed))
    env.reset()
    pol.reset_episode(device)
    while env.available_nodes:
        x  = torch.FloatTensor(_feat_unconstrained(cache, env, n)).to(device)
        av = _avail_mask(env, n, device)
        if not av.any(): break
        sc, h, ctx, _ = pol.forward(x, ei, av)
        ni = int(sc.argmax().item())
        d  = float(pol.get_discount_distribution(
             torch.cat([h[ni], ctx])).mean.item())
        _, rew, done, _ = env.step(ni, d)
        pol.update_sequence_state(d, rew > 0, float(rew))
        if done: break
    return float(env.total_revenue)
