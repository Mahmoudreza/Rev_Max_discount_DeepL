"""
src/evaluation/paper_eval.py

Paper-figure evaluation: 4 methods, acceptance-based budget curves.
Acceptance curve = cumulative revenue indexed by |S| (buyers IN seed set).

Methods:
  greedy_discount   — Babaei et al. 2013 (best hand-crafted)
  s2v_dqn_dec       — Degree-ordered + Greedy pricing (decoupled GNN proxy)
  im_rl             — Rev-GNN-IM-RL (joint, no LSTM)
  lstm              — Rev-GNN-LSTM  (joint + LSTM)
"""

import os, csv, json
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from omegaconf import OmegaConf

from src.evaluation.baselines import (
    greedy_discount_trajectory, _make_env,
    _compute_normalized_infl, _rayleigh_price,
    ie_strategy, mu_discount, sigma_discount, greedy_discount as greedy_discount_total,
)
from src.env.graph_generators import (
    generate_forest_fire, generate_modular_forest_fire, load_rice_facebook,
)
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.models.policies.joint_policy import JointPolicy
from src.utils.features import (
    compute_static_features, build_graph_feature_cache, compute_node_features_fast,
)
from src.utils.helpers import graph_to_pyg_data, get_available_mask, ensure_dir


# ── Type alias ─────────────────────────────────────────────────────────────────
# (cum_rev_by_S_entry, discount_per_offer, accepted_per_offer)
ACCurve = Tuple[List[float], List[float], List[bool]]


# ── Policy loaders ─────────────────────────────────────────────────────────────

def load_lstm(ckpt: str, cfg, device: torch.device) -> SequentialJointPolicy:
    enc = GraphSAGEEncoder(cfg.features.dim, cfg.encoder.hidden_dim,
                           cfg.encoder.n_layers, cfg.encoder.dropout)
    lstm = EpisodeLSTM(graph_dim=cfg.encoder.hidden_dim,
                       lstm_hidden=cfg.sequence_model.lstm_hidden,
                       n_layers=cfg.sequence_model.lstm_n_layers)
    pol = SequentialJointPolicy(enc, lstm, gnn_dim=cfg.encoder.hidden_dim,
                                context_dim=cfg.sequence_model.lstm_hidden)
    pol.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return pol.to(device).eval()


def load_im(ckpt: str, cfg, device: torch.device) -> JointPolicy:
    enc = GraphSAGEEncoder(cfg.features.dim, cfg.encoder.hidden_dim,
                           cfg.encoder.n_layers, cfg.encoder.dropout)
    pol = JointPolicy(enc, hidden_dim=cfg.encoder.hidden_dim)
    pol.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return pol.to(device).eval()


# ── Acceptance-curve collectors ────────────────────────────────────────────────

def ac_greedy(graph, cfg) -> ACCurve:
    """Greedy-Discount acceptance curve."""
    traj = greedy_discount_trajectory(graph, cfg)
    cum_rev: List[float] = []
    discounts: List[float] = []
    accepted: List[bool] = []
    total = 0.0
    for step in traj:
        d = float(step["discount"])
        acc = bool(step["accepted"])
        price = float(step.get("price", step.get("marginal_gain", 0.0)))
        if acc:
            total += price
            cum_rev.append(total)
        discounts.append(d)
        accepted.append(acc)
    return cum_rev, discounts, accepted


def ac_s2v_dec(graph, cfg) -> ACCurve:
    """S2V-DQN (dec.) — degree-sort + Greedy pricing (fast, no MC)."""
    nodes_sorted = sorted(graph.nodes(), key=lambda v: graph.degree(v), reverse=True)
    env = _make_env(graph, cfg)
    env.reset()
    b = float(cfg.influence.b)
    lw = env._link_weights
    cum_rev: List[float] = []
    discounts: List[float] = []
    accepted: List[bool] = []
    total = 0.0

    for node in nodes_sorted:
        if node in env.offered:
            continue
        infl = _compute_normalized_infl(graph, node, env.S, lw)
        if infl < 2.0 / 6.0:
            price = 0.0
        elif infl < 4.0 / 6.0:
            price = _rayleigh_price(2.0 / 6.0, b)
        else:
            price = _rayleigh_price(4.0 / 6.0, b)

        true_val = env._compute_valuation(node)

        if price == 0.0:
            env.S.add(node)
            env._influence_cache = {}
            disc = 1.0
            acc = True
            total += 0.0
            cum_rev.append(total)
        elif true_val >= price:
            env.S.add(node)
            env._influence_cache = {}
            disc = max(0.0, 1.0 - price / true_val) if true_val > 0 else 0.0
            acc = True
            total += price
            cum_rev.append(total)
        else:
            disc = max(0.0, 1.0 - price / true_val) if true_val > 0 else 0.0
            acc = False

        discounts.append(disc)
        accepted.append(acc)
        env.offered.add(node)
        env.t += 1

    return cum_rev, discounts, accepted


def ac_im_rl(policy: JointPolicy, graph, cfg, device: torch.device) -> ACCurve:
    """Rev-GNN-IM-RL acceptance curve (no LSTM)."""
    static = compute_static_features(graph)
    cache = build_graph_feature_cache(graph, static)
    n, nodes = graph.number_of_nodes(), list(graph.nodes())
    cum_rev: List[float] = []
    discounts: List[float] = []
    accepted: List[bool] = []
    total = 0.0

    with torch.no_grad():
        env = _make_env(graph, cfg)
        env.reset()
        for _ in range(n):
            avail = env.available_nodes
            if not avail:
                break
            feats = compute_node_features_fast(
                cache=cache, S=frozenset(env.S), offered=frozenset(env.offered),
                t=env.t, k=n, env=env)
            data = graph_to_pyg_data(graph, feats, device)
            mask = get_available_mask(n, frozenset(env.offered), nodes, device)
            nidx, disc, _ = policy.select_and_price(data.x, data.edge_index, mask, greedy=True)
            if nidx not in avail:
                nidx = avail[0]
            _, rew, done, _ = env.step(nidx, disc)
            acc = bool(rew > 0)
            discounts.append(float(disc))
            accepted.append(acc)
            if acc:
                total += float(rew)
                cum_rev.append(total)
            if done:
                break

    return cum_rev, discounts, accepted


def ac_lstm(policy: SequentialJointPolicy, graph, cfg, device: torch.device) -> ACCurve:
    """Rev-GNN-LSTM acceptance curve."""
    eval_dev = torch.device("cpu")
    policy.to(eval_dev)
    static = compute_static_features(graph)
    cache = build_graph_feature_cache(graph, static)
    n, nodes = graph.number_of_nodes(), list(graph.nodes())
    cum_rev: List[float] = []
    discounts: List[float] = []
    accepted: List[bool] = []
    total = 0.0

    with torch.no_grad():
        env = _make_env(graph, cfg)
        env.reset()
        policy.reset_episode(eval_dev)
        for _ in range(n):
            avail = env.available_nodes
            if not avail:
                break
            feats = compute_node_features_fast(
                cache=cache, S=frozenset(env.S), offered=frozenset(env.offered),
                t=env.t, k=n, env=env)
            data = graph_to_pyg_data(graph, feats, eval_dev)
            mask = get_available_mask(n, frozenset(env.offered), nodes, eval_dev)
            nidx, disc, _ = policy.select_and_price(data.x, data.edge_index, mask, greedy=True)
            if nidx not in avail:
                nidx = avail[0]
            _, rew, done, _ = env.step(nidx, disc)
            acc = bool(rew > 0)
            discounts.append(float(disc))
            accepted.append(acc)
            if acc:
                total += float(rew)
                cum_rev.append(total)
            policy.update_sequence_state(float(disc), acc, float(rew))
            if done:
                break

    policy.to(device)
    return cum_rev, discounts, accepted


# ── Budget-curve aggregator ────────────────────────────────────────────────────

def revenue_at_k(cum_rev_by_S: List[float], k: int) -> float:
    """Lookup revenue when |S| = k. Clamp to last value if k > len."""
    if not cum_rev_by_S:
        return 0.0
    ki = min(k - 1, len(cum_rev_by_S) - 1)
    return cum_rev_by_S[ki] if ki >= 0 else 0.0


def run_budget_sweep_4methods(
    lstm_pol: SequentialJointPolicy,
    im_pol: JointPolicy,
    graph_fn,                         # callable(seed) → nx.Graph
    cfg,
    device: torch.device,
    k_values: List[int],
    n_seeds: int = 5,
    cfg_override: Optional[dict] = None,  # e.g. {"influence_model": "non_monotone"}
    out_path: Optional[str] = None,
) -> Dict:
    """Run 4-method budget sweep; mean ± std at each K (acceptance-based)."""
    if cfg_override:
        cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        for k_o, v_o in cfg_override.items():
            OmegaConf.update(cfg, k_o, v_o)

    METHOD_NAMES = ["Greedy-Discount", "S2V-DQN (dec.)", "Rev-GNN-IM-RL", "Rev-GNN-LSTM"]
    curves_by_method: Dict[str, List[List[float]]] = {m: [] for m in METHOD_NAMES}
    all_discounts: Dict[str, List[List[float]]] = {m: [] for m in METHOD_NAMES}
    all_accepted: Dict[str, List[List[bool]]] = {m: [] for m in METHOD_NAMES}

    for seed in range(n_seeds):
        graph = graph_fn(seed)
        g_curve, g_disc, g_acc = ac_greedy(graph, cfg)
        s_curve, s_disc, s_acc = ac_s2v_dec(graph, cfg)
        i_curve, i_disc, i_acc = ac_im_rl(im_pol, graph, cfg, device)
        l_curve, l_disc, l_acc = ac_lstm(lstm_pol, graph, cfg, device)
        for m, c, d, a in zip(
            METHOD_NAMES,
            [g_curve, s_curve, i_curve, l_curve],
            [g_disc, s_disc, i_disc, l_disc],
            [g_acc, s_acc, i_acc, l_acc],
        ):
            curves_by_method[m].append(c)
            all_discounts[m].append(d)
            all_accepted[m].append(a)

    # Build result dict
    result = {"k_values": k_values, "n_seeds": n_seeds}
    for m in METHOD_NAMES:
        revs = np.array([[revenue_at_k(c, k) for k in k_values]
                         for c in curves_by_method[m]])   # (n_seeds, len_k)
        result[m] = {
            "mean": revs.mean(axis=0).tolist(),
            "std": revs.std(axis=0).tolist(),
            "curves": [c for c in curves_by_method[m]],
        }

    if out_path:
        ensure_dir(os.path.dirname(out_path))
        with open(out_path, "w") as f:
            json.dump({k: v for k, v in result.items()
                       if k not in ["k_values", "n_seeds"] or True}, f)
    return result


def run_20seed_comparison(
    lstm_pol: SequentialJointPolicy,
    im_pol: JointPolicy,
    graph_fn,
    cfg,
    device: torch.device,
    n_seeds: int = 20,
    out_path: Optional[str] = None,
) -> Dict:
    """Run 20-seed full-episode comparison; include IE-Str, µ-Disc, σ-Disc, ToupleGDD."""
    METHOD_4 = ["Greedy-Discount", "S2V-DQN (dec.)", "Rev-GNN-IM-RL", "Rev-GNN-LSTM"]
    ALL_7 = ["IE-Strategy", "µ-Discount", "σ-Discount",
             "Greedy-Discount", "S2V-DQN (dec.)", "Rev-GNN-IM-RL", "Rev-GNN-LSTM"]
    revs: Dict[str, List[float]] = {m: [] for m in ALL_7}

    for seed in range(n_seeds):
        graph = graph_fn(seed)
        # Hand-crafted baselines
        revs["IE-Strategy"].append(float(ie_strategy(graph, cfg) or 0.0))
        revs["µ-Discount"].append(float(mu_discount(graph, cfg) or 0.0))
        revs["σ-Discount"].append(float(sigma_discount(graph, cfg) or 0.0))
        # Greedy & S2V
        g_c, _, _ = ac_greedy(graph, cfg)
        revs["Greedy-Discount"].append(revenue_at_k(g_c, len(g_c)) if g_c else 0.0)
        s_c, _, _ = ac_s2v_dec(graph, cfg)
        revs["S2V-DQN (dec.)"].append(revenue_at_k(s_c, len(s_c)) if s_c else 0.0)
        # Joint methods
        i_c, _, _ = ac_im_rl(im_pol, graph, cfg, device)
        revs["Rev-GNN-IM-RL"].append(revenue_at_k(i_c, len(i_c)) if i_c else 0.0)
        l_c, _, _ = ac_lstm(lstm_pol, graph, cfg, device)
        revs["Rev-GNN-LSTM"].append(revenue_at_k(l_c, len(l_c)) if l_c else 0.0)

    result = {}
    greedy_mean = np.mean(revs["Greedy-Discount"])
    for m in ALL_7:
        vals = np.array(revs[m])
        delta = (vals.mean() - greedy_mean) / greedy_mean * 100 if greedy_mean > 0 else 0
        result[m] = {"mean": float(vals.mean()), "std": float(vals.std()),
                     "all": revs[m], "delta_pct": float(delta)}

    if out_path:
        ensure_dir(os.path.dirname(out_path))
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
    return result


def run_generalization_all(
    lstm_pol: SequentialJointPolicy,
    im_pol: JointPolicy,
    cfg,
    device: torch.device,
    n_seeds: int = 5,
    data_dir: str = "data/raw",
    out_path: Optional[str] = None,
) -> Dict:
    """5-network generalization: IE-Str, Greedy, S2V-Dec, IM-RL, LSTM."""
    p, pb = cfg.graph.p, cfg.graph.pb
    METHODS = ["IE-Strategy", "Greedy-Discount", "S2V-DQN (dec.)", "Rev-GNN-IM-RL", "Rev-GNN-LSTM"]
    NETWORKS = ["FF n=500", "FF n=1000", "FF n=2000", "Modular FF", "Rice-FB n=443"]
    GRAPH_FNS = [
        lambda s: generate_forest_fire(500,  p, pb, seed=s),
        lambda s: generate_forest_fire(1000, p, pb, seed=s),
        lambda s: generate_forest_fire(2000, p, pb, seed=s),
        lambda s: generate_modular_forest_fire([200, 300, 500], p, pb, 0.01, seed=s),
    ]
    try:
        rf_graph = load_rice_facebook(data_dir=data_dir)
        GRAPH_FNS.append(lambda s: rf_graph)
    except FileNotFoundError:
        GRAPH_FNS.append(None)
        NETWORKS[-1] = "Rice-FB (missing)"

    result: Dict[str, Dict] = {}
    for net_name, graph_fn in zip(NETWORKS, GRAPH_FNS):
        if graph_fn is None:
            result[net_name] = {m: {"mean": float("nan"), "std": float("nan")} for m in METHODS}
            continue
        net_revs: Dict[str, List[float]] = {m: [] for m in METHODS}
        for seed in range(n_seeds):
            graph = graph_fn(seed)
            net_revs["IE-Strategy"].append(float(ie_strategy(graph, cfg) or 0.0))
            g_c = ac_greedy(graph, cfg)[0]
            net_revs["Greedy-Discount"].append(g_c[-1] if g_c else 0.0)
            s_c = ac_s2v_dec(graph, cfg)[0]
            net_revs["S2V-DQN (dec.)"].append(s_c[-1] if s_c else 0.0)
            i_c = ac_im_rl(im_pol, graph, cfg, device)[0]
            net_revs["Rev-GNN-IM-RL"].append(i_c[-1] if i_c else 0.0)
            l_c = ac_lstm(lstm_pol, graph, cfg, device)[0]
            net_revs["Rev-GNN-LSTM"].append(l_c[-1] if l_c else 0.0)
        result[net_name] = {}
        for m in METHODS:
            v = np.array(net_revs[m])
            result[net_name][m] = {"mean": float(v.mean()), "std": float(v.std()),
                                   "all": net_revs[m]}

    if out_path:
        ensure_dir(os.path.dirname(out_path))
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
    return result


def run_discount_trajectory(
    lstm_pol: SequentialJointPolicy,
    im_pol: JointPolicy,
    graph,
    cfg,
    device: torch.device,
) -> Dict:
    """3-method discount trajectories for 1 episode (Fig 5)."""
    g_curve, g_disc, g_acc = ac_greedy(graph, cfg)
    i_curve, i_disc, i_acc = ac_im_rl(im_pol, graph, cfg, device)
    l_curve, l_disc, l_acc = ac_lstm(lstm_pol, graph, cfg, device)
    return {
        "Greedy-Discount": {"discounts": g_disc, "accepted": g_acc},
        "Rev-GNN-IM-RL":   {"discounts": i_disc, "accepted": i_acc},
        "Rev-GNN-LSTM":    {"discounts": l_disc, "accepted": l_acc},
    }
