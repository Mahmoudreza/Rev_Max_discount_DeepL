"""
src/evaluation/idea1_eval.py

Four evaluation tasks for robustness evidence before paper submission.

Task 1: Multi-seed robustness (20 seeds, FF n=1000)
Task 2: Cross-network generalisation (5 networks × 5 seeds, zero-shot)
Task 3: Non-monotone influence model (5 seeds, FF n=1000)
Task 4: LSTM vs no-LSTM ablation (20 seeds, FF n=1000, extended metrics)
"""

import copy, csv, json, os
from typing import Dict, List, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy import stats as scipy_stats

from src.evaluation.baselines import greedy_discount_trajectory, _make_env
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


# ── Policy loaders ────────────────────────────────────────────────────────────

def load_lstm_policy(ckpt_path: str, cfg, device: torch.device) -> SequentialJointPolicy:
    """Reconstruct and load SequentialJointPolicy from checkpoint."""
    enc = GraphSAGEEncoder(
        cfg.features.dim, cfg.encoder.hidden_dim,
        cfg.encoder.n_layers, cfg.encoder.dropout,
    )
    lstm = EpisodeLSTM(
        graph_dim=cfg.encoder.hidden_dim,
        lstm_hidden=cfg.sequence_model.lstm_hidden,
        n_layers=cfg.sequence_model.lstm_n_layers,
    )
    policy = SequentialJointPolicy(
        enc, lstm,
        gnn_dim=cfg.encoder.hidden_dim,
        context_dim=cfg.sequence_model.lstm_hidden,
    )
    policy.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    policy.to(device).eval()
    return policy


def load_im_policy(ckpt_path: str, cfg, device: torch.device) -> JointPolicy:
    """Reconstruct and load JointPolicy (no LSTM) from checkpoint."""
    enc = GraphSAGEEncoder(
        cfg.features.dim, cfg.encoder.hidden_dim,
        cfg.encoder.n_layers, cfg.encoder.dropout,
    )
    policy = JointPolicy(enc, hidden_dim=cfg.encoder.hidden_dim)
    policy.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    policy.to(device).eval()
    return policy


# ── Episode evaluators ────────────────────────────────────────────────────────

def _eval_lstm_detailed(
    policy: SequentialJointPolicy,
    graph,
    cfg,
    device: torch.device,
) -> Dict:
    """Run one greedy LSTM episode, return extended metrics dict."""
    eval_dev = torch.device("cpu")
    policy.to(eval_dev)
    static = compute_static_features(graph)
    cache = build_graph_feature_cache(graph, static)
    n, nodes = graph.number_of_nodes(), list(graph.nodes())
    discounts_all: List[float] = []
    accepted_flags: List[bool] = []
    revenues_ordered: List[float] = []

    with torch.no_grad():
        env = _make_env(graph, cfg)
        env.reset()
        policy.reset_episode(eval_dev)
        for _ in range(n):
            available = env.available_nodes
            if not available:
                break
            feats = compute_node_features_fast(
                cache=cache, S=frozenset(env.S), offered=frozenset(env.offered),
                t=env.t, k=n, env=env,
            )
            data = graph_to_pyg_data(graph, feats, eval_dev)
            mask = get_available_mask(n, frozenset(env.offered), nodes, eval_dev)
            nidx, disc, _ = policy.select_and_price(data.x, data.edge_index, mask, greedy=True)
            if nidx not in available:
                nidx = available[0]
            _, rew, done, _ = env.step(nidx, disc)
            accept = rew > 0
            discounts_all.append(float(disc))
            accepted_flags.append(bool(accept))
            if accept:
                revenues_ordered.append(float(rew))
            policy.update_sequence_state(disc, bool(accept), float(rew))
            if done:
                break

    policy.to(device)
    n_acc = len(revenues_ordered)
    front100 = sum(revenues_ordered[:100]) if n_acc >= 100 else sum(revenues_ordered)
    back100  = sum(revenues_ordered[max(0, n_acc - 100):]) if n_acc >= 100 else sum(revenues_ordered)
    return {
        "revenue": float(env.total_revenue),
        "avg_discount": float(np.mean(discounts_all)) if discounts_all else 0.0,
        "acceptance_rate": float(np.mean(accepted_flags)) if accepted_flags else 0.0,
        "n_accepted": n_acc,
        "front100_rev": front100,
        "back100_rev": back100,
    }


def _eval_im_detailed(
    policy: JointPolicy,
    graph,
    cfg,
    device: torch.device,
) -> Dict:
    """Run one greedy JointPolicy (no LSTM) episode, return extended metrics dict."""
    static = compute_static_features(graph)
    cache = build_graph_feature_cache(graph, static)
    n, nodes = graph.number_of_nodes(), list(graph.nodes())
    discounts_all: List[float] = []
    accepted_flags: List[bool] = []
    revenues_ordered: List[float] = []

    with torch.no_grad():
        env = _make_env(graph, cfg)
        env.reset()
        for _ in range(n):
            available = env.available_nodes
            if not available:
                break
            feats = compute_node_features_fast(
                cache=cache, S=frozenset(env.S), offered=frozenset(env.offered),
                t=env.t, k=n, env=env,
            )
            data = graph_to_pyg_data(graph, feats, device)
            mask = get_available_mask(n, frozenset(env.offered), nodes, device)
            nidx, disc, _ = policy.select_and_price(data.x, data.edge_index, mask, greedy=True)
            if nidx not in available:
                nidx = available[0]
            _, rew, done, _ = env.step(nidx, disc)
            accept = rew > 0
            discounts_all.append(float(disc))
            accepted_flags.append(bool(accept))
            if accept:
                revenues_ordered.append(float(rew))
            if done:
                break

    n_acc = len(revenues_ordered)
    front100 = sum(revenues_ordered[:100]) if n_acc >= 100 else sum(revenues_ordered)
    back100  = sum(revenues_ordered[max(0, n_acc - 100):]) if n_acc >= 100 else sum(revenues_ordered)
    return {
        "revenue": float(env.total_revenue),
        "avg_discount": float(np.mean(discounts_all)) if discounts_all else 0.0,
        "acceptance_rate": float(np.mean(accepted_flags)) if accepted_flags else 0.0,
        "n_accepted": n_acc,
        "front100_rev": front100,
        "back100_rev": back100,
    }


def _eval_greedy_discount(graph, cfg) -> float:
    """Run greedy-discount baseline through env; return total revenue."""
    traj = greedy_discount_trajectory(graph, cfg)
    env = _make_env(graph, cfg)
    env.reset()
    nodes = list(graph.nodes())
    n = graph.number_of_nodes()
    for item in traj:
        nidx = item["node_idx"]
        disc = item["discount"]
        node = nodes[nidx] if nidx < n else nodes[0]
        if node not in env.offered:
            env.step(nidx, disc)
    return float(env.total_revenue)


# ── Task 1: Multi-seed robustness ─────────────────────────────────────────────

def task1_robustness(
    lstm_policy: SequentialJointPolicy,
    cfg,
    device: torch.device,
    logger,
    n_seeds: int = 20,
    out_path: str = "results/logs/robustness_20seeds.csv",
) -> Dict:
    logger.info(f"Task 1: Multi-seed robustness ({n_seeds} seeds, FF n=1000)")
    p, pb = cfg.graph.p, cfg.graph.pb
    lstm_revs, greedy_revs = [], []
    rows = []

    for seed in range(n_seeds):
        graph = generate_forest_fire(1000, p, pb, seed=seed)
        # Use non-overlapping seeds: graph structure from seed, env weights vary
        lstm_r  = _eval_lstm_detailed(lstm_policy, graph, cfg, device)["revenue"]
        greedy_r = _eval_greedy_discount(graph, cfg)
        lstm_revs.append(lstm_r)
        greedy_revs.append(greedy_r)
        rows.append({"seed": seed, "lstm": lstm_r, "greedy": greedy_r,
                     "wins": "yes" if lstm_r > greedy_r else "no"})
        logger.info(f"  seed={seed:2d}  LSTM={lstm_r:.2f}  Greedy={greedy_r:.2f}  "
                    f"{'LSTM wins' if lstm_r > greedy_r else 'Greedy wins'}")

    # Statistics
    t_stat, p_val = scipy_stats.ttest_rel(lstm_revs, greedy_revs)
    wins = sum(1 for r in rows if r["wins"] == "yes")
    lstm_mean, lstm_std = float(np.mean(lstm_revs)), float(np.std(lstm_revs))
    g_mean, g_std = float(np.mean(greedy_revs)), float(np.std(greedy_revs))

    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "lstm", "greedy", "wins"])
        w.writeheader()
        w.writerows(rows)
        w.writerow({"seed": "mean", "lstm": f"{lstm_mean:.2f}±{lstm_std:.2f}",
                    "greedy": f"{g_mean:.2f}±{g_std:.2f}", "wins": f"{wins}/{n_seeds}"})

    logger.info(f"  LSTM: {lstm_mean:.2f} ± {lstm_std:.2f}")
    logger.info(f"  Greedy: {g_mean:.2f} ± {g_std:.2f}")
    logger.info(f"  p-value: {p_val:.4f}   Win rate: {wins}/{n_seeds}")
    return {"lstm_mean": lstm_mean, "lstm_std": lstm_std, "greedy_mean": g_mean,
            "greedy_std": g_std, "p_value": float(p_val), "wins": wins, "n_seeds": n_seeds}


# ── Task 2: Cross-network generalisation ──────────────────────────────────────

def task2_generalisation(
    lstm_policy: SequentialJointPolicy,
    cfg,
    device: torch.device,
    logger,
    n_seeds: int = 5,
    data_dir: str = "data/raw",
    out_path: str = "results/logs/generalization_eval.csv",
) -> Dict:
    logger.info(f"Task 2: Cross-network generalisation ({n_seeds} seeds each)")
    p, pb = cfg.graph.p, cfg.graph.pb

    def _run_seeds(graph_fn, label):
        lr, gr = [], []
        for s in range(n_seeds):
            g = graph_fn(s)
            lr.append(_eval_lstm_detailed(lstm_policy, g, cfg, device)["revenue"])
            gr.append(_eval_greedy_discount(g, cfg))
        lm, ls = float(np.mean(lr)), float(np.std(lr))
        gm, gs = float(np.mean(gr)), float(np.std(gr))
        delta = (lm - gm) / gm * 100 if gm > 0 else 0.0
        logger.info(f"  {label:30s}  LSTM={lm:.2f}±{ls:.2f}  Greedy={gm:.2f}±{gs:.2f}  Δ={delta:+.1f}%")
        return {"network": label, "lstm_mean": lm, "lstm_std": ls,
                "greedy_mean": gm, "greedy_std": gs, "delta_pct": delta}

    rows = []
    rows.append(_run_seeds(lambda s: generate_forest_fire(500,  p, pb, seed=s), "FF n=500"))
    rows.append(_run_seeds(lambda s: generate_forest_fire(1000, p, pb, seed=s), "FF n=1000"))
    rows.append(_run_seeds(lambda s: generate_forest_fire(2000, p, pb, seed=s), "FF n=2000"))
    rows.append(_run_seeds(
        lambda s: generate_modular_forest_fire([200, 300, 500], p, pb, 0.01, seed=s),
        "Modular FF n=1000",
    ))

    # Rice-Facebook (same graph, 5 seed evals for weight variability)
    try:
        rf_graph = load_rice_facebook(data_dir=data_dir)
        rows.append(_run_seeds(lambda s: rf_graph, "rice_facebook n=443"))
    except FileNotFoundError as exc:
        logger.info(f"  [SKIP] rice_facebook: {exc}")
        rows.append({"network": "rice_facebook n=443", "lstm_mean": float("nan"),
                     "lstm_std": float("nan"), "greedy_mean": float("nan"),
                     "greedy_std": float("nan"), "delta_pct": float("nan")})

    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["network", "lstm_mean", "lstm_std",
                                           "greedy_mean", "greedy_std", "delta_pct"])
        w.writeheader()
        w.writerows(rows)

    return {"networks": rows}


# ── Task 3: Non-monotone influence model ─────────────────────────────────────

def task3_nonmonotone(
    lstm_policy: SequentialJointPolicy,
    cfg,
    device: torch.device,
    logger,
    n_seeds: int = 5,
    out_path: str = "results/logs/nonmonotone_eval.csv",
) -> Dict:
    logger.info(f"Task 3: Non-monotone influence (FF n=1000, {n_seeds} seeds)")
    p, pb = cfg.graph.p, cfg.graph.pb

    # Override influence model to non-monotone
    cfg_nm = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    with OmegaConf.open_dict(cfg_nm):
        cfg_nm.influence_model = "non_monotone"

    lstm_revs, greedy_revs = [], []
    rows = []
    for seed in range(n_seeds):
        graph = generate_forest_fire(1000, p, pb, seed=seed)
        lr = _eval_lstm_detailed(lstm_policy, graph, cfg_nm, device)["revenue"]
        gr = _eval_greedy_discount(graph, cfg_nm)
        lstm_revs.append(lr)
        greedy_revs.append(gr)
        rows.append({"seed": seed, "lstm": lr, "greedy": gr,
                     "wins": "yes" if lr > gr else "no"})
        logger.info(f"  seed={seed}  LSTM={lr:.2f}  Greedy={gr:.2f}")

    lm, ls = float(np.mean(lstm_revs)), float(np.std(lstm_revs))
    gm, gs = float(np.mean(greedy_revs)), float(np.std(greedy_revs))
    wins = sum(1 for r in rows if r["wins"] == "yes")

    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "lstm", "greedy", "wins"])
        w.writeheader()
        w.writerows(rows)

    logger.info(f"  LSTM: {lm:.2f} ± {ls:.2f}")
    logger.info(f"  Greedy: {gm:.2f} ± {gs:.2f}")
    logger.info(f"  Win rate: {wins}/{n_seeds}")
    return {"lstm_mean": lm, "lstm_std": ls, "greedy_mean": gm,
            "greedy_std": gs, "wins": wins, "n_seeds": n_seeds}


# ── Task 4: LSTM vs no-LSTM ablation ─────────────────────────────────────────

def task4_ablation(
    lstm_policy: SequentialJointPolicy,
    im_policy: JointPolicy,
    cfg,
    device: torch.device,
    logger,
    n_seeds: int = 20,
    out_path: str = "results/logs/ablation_lstm_vs_no_lstm.csv",
) -> Dict:
    logger.info(f"Task 4: LSTM vs no-LSTM ablation ({n_seeds} seeds, FF n=1000)")
    p, pb = cfg.graph.p, cfg.graph.pb

    lstm_rows, im_rows = [], []
    for seed in range(n_seeds):
        graph = generate_forest_fire(1000, p, pb, seed=seed)
        lr = _eval_lstm_detailed(lstm_policy, graph, cfg, device)
        ir = _eval_im_detailed(im_policy, graph, cfg, device)
        lstm_rows.append(lr)
        im_rows.append(ir)
        logger.info(
            f"  seed={seed:2d}  LSTM={lr['revenue']:.2f}"
            f" (disc={lr['avg_discount']:.3f} acc={lr['acceptance_rate']:.3f})"
            f"  IM-RL={ir['revenue']:.2f}"
            f" (disc={ir['avg_discount']:.3f} acc={ir['acceptance_rate']:.3f})"
        )

    def _agg(rows, key):
        vals = [r[key] for r in rows]
        return float(np.mean(vals)), float(np.std(vals))

    lm_rev, ls_rev = _agg(lstm_rows, "revenue")
    im_rev, is_rev = _agg(im_rows, "revenue")
    lm_disc, _ = _agg(lstm_rows, "avg_discount")
    im_disc, _ = _agg(im_rows, "avg_discount")
    lm_acc, _ = _agg(lstm_rows, "acceptance_rate")
    im_acc, _ = _agg(im_rows, "acceptance_rate")
    lm_f100, _ = _agg(lstm_rows, "front100_rev")
    im_f100, _ = _agg(im_rows, "front100_rev")
    lm_b100, _ = _agg(lstm_rows, "back100_rev")
    im_b100, _ = _agg(im_rows, "back100_rev")

    t_stat, p_val = scipy_stats.ttest_rel(
        [r["revenue"] for r in lstm_rows], [r["revenue"] for r in im_rows]
    )

    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "lstm_rev", "im_rev", "lstm_disc", "im_disc",
                         "lstm_acc", "im_acc", "lstm_front100", "im_front100",
                         "lstm_back100", "im_back100"])
        for i, (lr, ir) in enumerate(zip(lstm_rows, im_rows)):
            writer.writerow([i, lr["revenue"], ir["revenue"],
                             lr["avg_discount"], ir["avg_discount"],
                             lr["acceptance_rate"], ir["acceptance_rate"],
                             lr["front100_rev"], ir["front100_rev"],
                             lr["back100_rev"], ir["back100_rev"]])

    gap = lm_rev - im_rev
    pct = gap / im_rev * 100 if im_rev > 0 else 0.0
    logger.info(f"  LSTM:  rev={lm_rev:.2f}±{ls_rev:.2f}  disc={lm_disc:.3f}  acc={lm_acc:.3f}")
    logger.info(f"  IM-RL: rev={im_rev:.2f}±{is_rev:.2f}  disc={im_disc:.3f}  acc={im_acc:.3f}")
    logger.info(f"  Δ revenue: {gap:+.2f} ({pct:+.1f}%)  p-value: {p_val:.4f}")
    logger.info(f"  LSTM front100={lm_f100:.2f}  back100={lm_b100:.2f}  (front-loading ratio={lm_f100/(lm_b100+1e-9):.2f}x)")
    logger.info(f"  IM   front100={im_f100:.2f}  back100={im_b100:.2f}  (front-loading ratio={im_f100/(im_b100+1e-9):.2f}x)")

    return {"lstm_mean": lm_rev, "lstm_std": ls_rev, "im_mean": im_rev, "im_std": is_rev,
            "delta_revenue": gap, "delta_pct": pct, "p_value": float(p_val),
            "lstm_avg_discount": lm_disc, "im_avg_discount": im_disc,
            "lstm_acceptance_rate": lm_acc, "im_acceptance_rate": im_acc,
            "lstm_front100": lm_f100, "lstm_back100": lm_b100,
            "im_front100": im_f100, "im_back100": im_b100}


# ── Budget-K sweep ────────────────────────────────────────────────────────────

def eval_budget_curve_lstm(
    policy: SequentialJointPolicy,
    graph,
    cfg,
    device: torch.device,
) -> List[float]:
    """Run full LSTM episode; return cumulative revenue at each offer step k."""
    eval_dev = torch.device("cpu")
    policy.to(eval_dev)
    static = compute_static_features(graph)
    cache = build_graph_feature_cache(graph, static)
    n, nodes = graph.number_of_nodes(), list(graph.nodes())
    cum_rev: List[float] = []
    total = 0.0

    with torch.no_grad():
        env = _make_env(graph, cfg)
        env.reset()
        policy.reset_episode(eval_dev)
        for _ in range(n):
            available = env.available_nodes
            if not available:
                break
            feats = compute_node_features_fast(
                cache=cache, S=frozenset(env.S), offered=frozenset(env.offered),
                t=env.t, k=n, env=env,
            )
            data = graph_to_pyg_data(graph, feats, eval_dev)
            mask = get_available_mask(n, frozenset(env.offered), nodes, eval_dev)
            nidx, disc, _ = policy.select_and_price(data.x, data.edge_index, mask, greedy=True)
            if nidx not in available:
                nidx = available[0]
            _, rew, done, _ = env.step(nidx, disc)
            total += float(rew)
            cum_rev.append(total)
            policy.update_sequence_state(disc, bool(rew > 0), float(rew))
            if done:
                break

    policy.to(device)
    return cum_rev


def eval_budget_curve_greedy(graph, cfg) -> List[float]:
    """Run greedy-discount full episode; return cumulative revenue at each step k."""
    traj = greedy_discount_trajectory(graph, cfg)
    env = _make_env(graph, cfg)
    env.reset()
    nodes = list(graph.nodes())
    n = graph.number_of_nodes()
    cum_rev: List[float] = []
    total = 0.0
    for item in traj:
        nidx = item["node_idx"]
        disc = item["discount"]
        node = nodes[nidx] if nidx < n else nodes[0]
        if node not in env.offered:
            _, rew, _, _ = env.step(nidx, disc)
            total += float(rew)
        cum_rev.append(total)
    return cum_rev


def run_budget_sweep(
    lstm_policy: SequentialJointPolicy,
    cfg,
    device: torch.device,
    logger,
    n_seeds: int = 5,
    k_values: List[int] = None,
    out_path: str = "results/logs/budget_sweep_lstm_vs_greedy.csv",
) -> Dict:
    """Run budget-K sweep for LSTM vs Greedy; mean ± std at each K."""
    if k_values is None:
        k_values = [10, 25, 50, 100, 150, 200, 300, 400, 500, 750, 1000]
    p, pb = cfg.graph.p, cfg.graph.pb
    logger.info(f"Budget-K sweep ({n_seeds} seeds, K={k_values})")

    lstm_curves: List[List[float]] = []
    greedy_curves: List[List[float]] = []

    for seed in range(n_seeds):
        graph = generate_forest_fire(1000, p, pb, seed=seed)
        lc = eval_budget_curve_lstm(lstm_policy, graph, cfg, device)
        gc = eval_budget_curve_greedy(graph, cfg)
        # Pad to n=1000 if shorter
        lc += [lc[-1]] * (1000 - len(lc)) if lc else [0.0] * 1000
        gc += [gc[-1]] * (1000 - len(gc)) if gc else [0.0] * 1000
        lstm_curves.append(lc)
        greedy_curves.append(gc)
        logger.info(f"  seed={seed}  LSTM@K=1000: {lc[-1]:.2f}  Greedy@K=1000: {gc[-1]:.2f}")

    lc_arr = np.array(lstm_curves)   # (n_seeds, 1000)
    gc_arr = np.array(greedy_curves)

    rows = []
    for k in k_values:
        ki = min(k - 1, 999)
        lm = float(np.mean(lc_arr[:, ki]))
        ls = float(np.std(lc_arr[:, ki]))
        gm = float(np.mean(gc_arr[:, ki]))
        gs = float(np.std(gc_arr[:, ki]))
        rows.append({"k": k, "lstm_mean": lm, "lstm_std": ls,
                     "greedy_mean": gm, "greedy_std": gs})
        logger.info(f"  K={k:4d}  LSTM={lm:.2f}±{ls:.2f}  Greedy={gm:.2f}±{gs:.2f}")

    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["k", "lstm_mean", "lstm_std", "greedy_mean", "greedy_std"])
        w.writeheader()
        w.writerows(rows)

    # Also save full curves for smooth plotting
    full_path = out_path.replace(".csv", "_full.npy")
    np.save(full_path, {"lstm": lc_arr, "greedy": gc_arr})

    return {"k_values": k_values, "rows": rows,
            "lstm_curves": lc_arr.tolist(), "greedy_curves": gc_arr.tolist()}


def run_budget_sweep_on_graph(
    lstm_policy: SequentialJointPolicy,
    graph_fn,
    n_graph: int,
    cfg,
    device: torch.device,
    logger,
    n_seeds: int = 5,
    k_values: List[int] = None,
    graph_label: str = "graph",
    out_path: str = "results/logs/budget_sweep_custom.csv",
) -> Dict:
    """Budget-K sweep on a custom graph (e.g., rice_facebook).

    Args:
        graph_fn: callable(seed: int) -> nx.Graph.  Use `lambda s: real_graph` for fixed graphs.
        n_graph:  number of nodes in graph (for padding/K-axis).
        graph_label: label string for logging.
    """
    if k_values is None:
        n = n_graph
        k_values = sorted(set(
            [5, 10, 20, 30, 50, 75, 100, 150, 200, 300] + [n]
        ))
        k_values = [k for k in k_values if k <= n]
    logger.info(f"Budget-K on {graph_label} ({n_seeds} seeds, n={n_graph}, K={k_values})")

    lstm_curves_list: List[List[float]] = []
    greedy_curves_list: List[List[float]] = []

    for seed in range(n_seeds):
        graph = graph_fn(seed)
        n = graph.number_of_nodes()
        lc = eval_budget_curve_lstm(lstm_policy, graph, cfg, device)
        gc = eval_budget_curve_greedy(graph, cfg)
        # Pad / trim to n_graph
        lc = lc[:n_graph] + [lc[-1]] * max(0, n_graph - len(lc)) if lc else [0.0] * n_graph
        gc = gc[:n_graph] + [gc[-1]] * max(0, n_graph - len(gc)) if gc else [0.0] * n_graph
        lstm_curves_list.append(lc)
        greedy_curves_list.append(gc)
        logger.info(f"  seed={seed}  LSTM@K={n}: {lc[n-1]:.2f}  Greedy@K={n}: {gc[n-1]:.2f}")

    lc_arr = np.array(lstm_curves_list)   # (n_seeds, n_graph)
    gc_arr = np.array(greedy_curves_list)

    rows = []
    for k in k_values:
        ki = min(k - 1, n_graph - 1)
        lm = float(np.mean(lc_arr[:, ki]))
        ls = float(np.std(lc_arr[:, ki]))
        gm = float(np.mean(gc_arr[:, ki]))
        gs = float(np.std(gc_arr[:, ki]))
        rows.append({"k": k, "lstm_mean": lm, "lstm_std": ls,
                     "greedy_mean": gm, "greedy_std": gs})
        logger.info(f"  K={k:4d}  LSTM={lm:.2f}±{ls:.2f}  Greedy={gm:.2f}±{gs:.2f}")

    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["k", "lstm_mean", "lstm_std", "greedy_mean", "greedy_std"])
        w.writeheader()
        w.writerows(rows)

    full_path = out_path.replace(".csv", "_full.npy")
    np.save(full_path, {"lstm": lc_arr, "greedy": gc_arr})

    return {"k_values": k_values, "rows": rows,
            "lstm_curves": lc_arr.tolist(), "greedy_curves": gc_arr.tolist(),
            "n_graph": n_graph, "graph_label": graph_label}


# ── Summary ───────────────────────────────────────────────────────────────────

def print_and_save_summary(
    t1: Dict, t2: Dict, t3: Dict, t4: Dict,
    logger,
    out_path: str = "results/logs/idea1_evaluation_summary.json",
) -> None:
    logger.info("\n" + "=" * 50)
    logger.info("IDEA 1 — EVALUATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"\nRobustness (FF n=1000, {t1['n_seeds']} seeds):")
    logger.info(f"  Rev-GNN-LSTM:    {t1['lstm_mean']:.2f} ± {t1['lstm_std']:.2f}")
    logger.info(f"  Greedy-Discount: {t1['greedy_mean']:.2f} ± {t1['greedy_std']:.2f}")
    logger.info(f"  p-value: {t1['p_value']:.4f}   Win rate: {t1['wins']}/{t1['n_seeds']}")

    logger.info("\nGeneralisation (zero-shot):")
    for row in t2["networks"]:
        logger.info(
            f"  {row['network']:30s}  LSTM={row['lstm_mean']:.2f}±{row['lstm_std']:.2f}"
            f"  Greedy={row['greedy_mean']:.2f}±{row['greedy_std']:.2f}"
            f"  Δ={row['delta_pct']:+.1f}%"
        )

    logger.info(f"\nNon-monotone (FF n=1000, {t3['n_seeds']} seeds):")
    logger.info(f"  Rev-GNN-LSTM:    {t3['lstm_mean']:.2f} ± {t3['lstm_std']:.2f}")
    logger.info(f"  Greedy-Discount: {t3['greedy_mean']:.2f} ± {t3['greedy_std']:.2f}")
    logger.info(f"  Win rate: {t3['wins']}/{t3['n_seeds']}")

    logger.info("\nLSTM Ablation (FF n=1000, 20 seeds):")
    logger.info(f"  With LSTM: {t4['lstm_mean']:.2f} ± {t4['lstm_std']:.2f}")
    logger.info(f"  Without:   {t4['im_mean']:.2f} ± {t4['im_std']:.2f}")
    logger.info(f"  Δ: {t4['delta_revenue']:+.2f} ({t4['delta_pct']:+.1f}%)  p={t4['p_value']:.4f}")
    logger.info(f"  LSTM avg_disc={t4['lstm_avg_discount']:.3f}  IM avg_disc={t4['im_avg_discount']:.3f}")
    logger.info(f"  LSTM acc_rate={t4['lstm_acceptance_rate']:.3f}  IM acc_rate={t4['im_acceptance_rate']:.3f}")
    logger.info(f"  LSTM front-loading: {t4['lstm_front100']:.2f} (first 100 acc) vs {t4['lstm_back100']:.2f} (last 100 acc)")

    summary = {"task1_robustness": t1, "task2_generalisation": t2,
               "task3_nonmonotone": t3, "task4_ablation": t4}
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary saved → {out_path}")
