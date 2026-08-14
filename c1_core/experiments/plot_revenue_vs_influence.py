#!/usr/bin/env python3
"""
c1_core/experiments/plot_revenue_vs_influence.py
=================================================
Babaei-style figure: X = cumulative nodes influenced (|S| accepted seeds),
Y = cumulative seller revenue.

One full episode on FF_1000 seed=42 for each method.
Output: results/figures/c1_revenue_vs_influence.pdf + .png

Run on server:
  python c1_core/experiments/plot_revenue_vs_influence.py
"""
import sys, os
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
os.chdir(_REPO)

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from src.utils.helpers import load_config_with_base, ensure_dir, graph_to_pyg_data, get_available_mask
from src.evaluation.idea1_eval import load_lstm_policy, load_im_policy
from src.evaluation.baselines import _make_env, greedy_discount_trajectory, ie_strategy, mu_discount
from src.env.graph_generators import generate_forest_fire
from src.utils.features import compute_static_features, build_graph_feature_cache, compute_node_features_fast

LSTM_CKPT = "results/checkpoints/rev_gnn_lstm.pt"
IM_CKPT   = "results/checkpoints/rev_gnn_im_rl.pt"
OUT_DIR   = "results/figures"


def lstm_trajectory(policy, graph, cfg, is_lstm=True):
    """Run one greedy episode.
    Returns (x=step_t, y=cum_revenue) — one point per offer (submodular shape).
    """
    device = torch.device("cpu")
    policy.to(device)
    static = compute_static_features(graph)
    cache  = build_graph_feature_cache(graph, static)
    n, nodes = graph.number_of_nodes(), list(graph.nodes())
    xs, ys = [0], [0.0]
    cum_rev = 0.0
    step = 0

    with torch.no_grad():
        env = _make_env(graph, cfg)
        env.reset()
        if is_lstm:
            policy.reset_episode(device)
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
            cum_rev += max(0.0, float(rew))
            step += 1
            xs.append(step)
            ys.append(cum_rev)            # record EVERY step (submodular curve)
            if is_lstm:
                policy.update_sequence_state(disc, rew > 0, float(rew))
            if done:
                break

    return np.array(xs), np.array(ys)


def gd_trajectory_curve(graph, cfg):
    """Run GD via env.step(); return (step_t, cum_revenue) at every step."""
    traj = greedy_discount_trajectory(graph, cfg)
    env = _make_env(graph, cfg)
    env.reset()
    nodes = list(graph.nodes())
    n = graph.number_of_nodes()
    xs, ys = [0], [0.0]
    cum_rev = 0.0
    step = 0
    for item in traj:
        nidx, disc = item["node_idx"], item["discount"]
        node = nodes[nidx] if nidx < n else nodes[0]
        if node in env.offered:
            continue
        _, rew, done, _ = env.step(nidx, disc)
        cum_rev += max(0.0, float(rew))
        step += 1
        xs.append(step)
        ys.append(cum_rev)
        if done:
            break
    return np.array(xs), np.array(ys)


def main():
    cfg    = load_config_with_base("configs/experiments/rev_gnn_lstm.yaml")
    device = torch.device("cpu")
    graph  = generate_forest_fire(1000, p=cfg.graph.p, pb=cfg.graph.pb, seed=42)

    print("Loading checkpoints...")
    lstm_policy = load_lstm_policy(LSTM_CKPT, cfg, device)
    im_policy   = load_im_policy(IM_CKPT,   cfg, device)
    lstm_policy.eval(); im_policy.eval()

    print("Running LSTM episode...")
    x_lstm, y_lstm = lstm_trajectory(lstm_policy, graph, cfg, is_lstm=True)
    print(f"  LSTM: accepted={len(x_lstm)-1}  total_revenue={y_lstm[-1]:.2f}")

    print("Running IM-RL episode...")
    x_im, y_im = lstm_trajectory(im_policy, graph, cfg, is_lstm=False)
    print(f"  IM-RL: accepted={len(x_im)-1}  total_revenue={y_im[-1]:.2f}")

    print("Running GD episode...")
    x_gd, y_gd = gd_trajectory_curve(graph, cfg)
    print(f"  GD: accepted={len(x_gd)-1}  total_revenue={y_gd[-1]:.2f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    ax.plot(x_lstm, y_lstm, color="#1f77b4", lw=2.0,
            label=f"Rev-GNN-LSTM  (R={y_lstm[-1]:.0f})")
    ax.plot(x_im,   y_im,   color="#ff7f0e", lw=1.8, ls="--",
            label=f"Rev-GNN-IM-RL (R={y_im[-1]:.0f})")
    ax.plot(x_gd,   y_gd,   color="#2ca02c", lw=1.8, ls="-.",
            label=f"Greedy-Discount (Babaei et al. 2013) (R={y_gd[-1]:.0f})")

    ax.set_xlabel("Number of Buyers Offered  (episode step $t$)", fontsize=12)
    ax.set_ylabel("Cumulative Seller Revenue", fontsize=12)
    ax.set_title("Revenue vs. Buyers Offered  (submodular curve, FF-1000 seed=42)", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    ensure_dir(OUT_DIR)
    for ext in ("pdf", "png"):
        p = f"{OUT_DIR}/c1_revenue_vs_influence.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved → {p}")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
