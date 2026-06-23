"""
experiments/run_benchmark.py

Multi-graph benchmark: compare all methods across four graph topologies.

Graphs:
  rice_facebook  — Real Rice-Facebook undergrad network, age 18–20 (Ali et al. 2023)
                   n ≈ 441, real community structure (V1: age 18-19, V2: age 20)
  ba500          — Barabási-Albert preferential-attachment (n=500, m=3)
  sbm500         — Stochastic Block Model (n=500, 5 blocks, p_in=0.30, p_out=0.01)
  plow500        — Holme-Kim Power-Law Cluster / PLow (n=500, m=3, p=0.6)

Usage:
  cd revmax-aaai2027
  python experiments/run_benchmark.py --graph rice_facebook
  python experiments/run_benchmark.py --graph ba500
  python experiments/run_benchmark.py --graph sbm500
  python experiments/run_benchmark.py --graph plow500
  python experiments/run_benchmark.py --graph all        # run all 4 sequentially
"""

import sys
import copy
import argparse
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.helpers import load_config_with_base, set_seed
from src.utils.logging import ExperimentLogger
from src.utils.features import compute_static_features
from src.env.graph_generators import (
    generate_forest_fire, generate_ba, generate_sbm,
    generate_power_law_cluster, load_rice_facebook,
)
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.policies.joint_policy import JointPolicy
from src.evaluation.baselines import run_all_baselines, ie_strategy_trajectory

# Re-use training helpers from run_all_experiments
from experiments.run_all_experiments import (
    imitation_pretrain, train_gail_rl_rich, eval_joint,
    _rollout_joint, _reward_to_go,
)


# ── Graph registry ────────────────────────────────────────────────────────────

def _build_graphs(graph_name: str, seed: int):
    """Build test graph and training pool for the requested benchmark graph.

    Training strategy:
      - rice_facebook: test = real RF graph; train = 15 Forest Fire graphs (n=441)
        (train on synthetic to test zero-shot transfer to real topology)
      - ba500/sbm500/plow500: test = one instance; train = 15 same-type instances
        (in-distribution generalisation across random seeds)

    Returns (test_graph, train_graphs, graph_label, n).
    """
    if graph_name == "rice_facebook":
        test_graph   = load_rice_facebook(data_dir="data/raw")
        n            = test_graph.number_of_nodes()
        # Train on FF graphs of the same size (zero-shot transfer setting)
        train_graphs = [generate_forest_fire(n, p=0.37, pb=0.32, seed=seed + i)
                        for i in range(15)]
        label = f"Rice-Facebook (n={n}, real)"

    elif graph_name == "ba500":
        n            = 500
        test_graph   = generate_ba(n, m=3, seed=9999)
        train_graphs = [generate_ba(n, m=3, seed=seed + i) for i in range(15)]
        label = f"BA (n={n}, m=3)"

    elif graph_name == "sbm500":
        n            = 500
        test_graph   = generate_sbm(n, n_blocks=5, p_in=0.30, p_out=0.01, seed=9999)
        train_graphs = [generate_sbm(n, n_blocks=5, p_in=0.30, p_out=0.01, seed=seed + i)
                        for i in range(15)]
        label = f"SBM (n={n}, 5 blocks)"

    elif graph_name == "plow500":
        n            = 500
        test_graph   = generate_power_law_cluster(n, m=3, p=0.6, seed=9999)
        train_graphs = [generate_power_law_cluster(n, m=3, p=0.6, seed=seed + i)
                        for i in range(15)]
        label = f"PLow (n={n}, m=3, p=0.6)"

    else:
        raise ValueError(f"Unknown graph: {graph_name}. "
                         f"Choose from: rice_facebook, ba500, sbm500, plow500, all")

    return test_graph, train_graphs, label, n


# ── Single-graph benchmark run ────────────────────────────────────────────────

def run_benchmark_graph(graph_name: str):
    """Run the full comparison on one graph type.

    Returns results dict: {method_name: revenue}.
    """
    # Config: demo overrides (n_nodes overridden by actual graph size in generators)
    overrides = [
        "influence.n_mc_samples=5",
        "training.reinforce_epochs=50",
        "training.n_train_graphs=15",
    ]
    cfg     = load_config_with_base("configs/experiments/rev_gnn_im_rl.yaml",
                                    overrides=overrides)
    set_seed(cfg.project.seed)
    device  = torch.device("cpu")
    logger  = ExperimentLogger(cfg, run_name=f"benchmark_{graph_name}")

    test_graph, train_graphs, label, n = _build_graphs(graph_name, cfg.project.seed)
    test_static = compute_static_features(test_graph)

    results = {}
    logger.info(f"\n{'='*60}")
    logger.info(f"BENCHMARK: {label}")
    logger.info(f"  Test graph: n={test_graph.number_of_nodes()}, "
                f"m={test_graph.number_of_edges()}")
    logger.info(f"  Train pool: {len(train_graphs)} graphs")
    logger.info('='*60)

    # 1. Baselines ─────────────────────────────────────────────────────────
    logger.info("\n--- Baselines (Babaei et al. 2013) ---")
    bl = run_all_baselines(test_graph, cfg, n_trials=3)
    for k, v in bl.items():
        results[k] = v
        logger.info(f"  {k:25s}: {v:.4f}")

    # 2. Shared IM warm-start (100 epochs, IE teacher, traj-caching) ───────
    logger.info("\n--- Shared IM warm-start (100 epochs, IE teacher) ---")
    enc = GraphSAGEEncoder(
        in_dim=cfg.features.dim, hidden_dim=cfg.encoder.hidden_dim,
        n_layers=cfg.encoder.n_layers, dropout=cfg.encoder.dropout,
    ).to(device)
    im_policy  = JointPolicy(enc, hidden_dim=cfg.encoder.hidden_dim).to(device)
    im_statics = {id(g): compute_static_features(g) for g in train_graphs}
    traj_cache = {id(g): ie_strategy_trajectory(g, cfg) for g in train_graphs}
    n_imitation = max(10, cfg.training.reinforce_epochs * 2)
    opt_im = torch.optim.Adam(im_policy.parameters(), lr=cfg.training.reinforce_lr)
    imitation_pretrain(im_policy, opt_im, cfg, train_graphs, im_statics, device,
                       n_imitation, traj_cache=traj_cache)
    logger.info(f"  IM warm-start complete ({n_imitation} epochs)")

    # 3. Rev-GNN (IM+RL) ───────────────────────────────────────────────────
    logger.info("\n--- Rev-GNN (IM+RL): REINFORCE fine-tuning ---")
    gnn_policy   = copy.deepcopy(im_policy)
    rl_epochs    = max(cfg.training.reinforce_epochs, 20)
    rl_optimizer = torch.optim.Adam(gnn_policy.parameters(),
                                    lr=cfg.training.reinforce_lr * 0.3)
    rl_baseline  = 0.0
    for epoch in range(rl_epochs):
        graph  = train_graphs[epoch % len(train_graphs)]
        static = im_statics[id(graph)]
        lps, rews, _ = _rollout_joint(gnn_policy, graph, static, cfg, device,
                                      greedy=False)
        returns = _reward_to_go(rews, gamma=0.99, device=device, normalize=True)
        rl_baseline = 0.95 * rl_baseline + 0.05 * float(returns.mean())
        loss = torch.stack(
            [-lp * (R - rl_baseline) for lp, R in zip(lps, returns)]
        ).mean()
        rl_optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(gnn_policy.parameters(), cfg.training.grad_clip)
        rl_optimizer.step()
    results["Rev-GNN (IM+RL)"] = eval_joint(gnn_policy, test_graph, test_static,
                                             cfg, device)
    logger.info(f"  {'Rev-GNN (IM+RL)':25s}: {results['Rev-GNN (IM+RL)']:.4f}")

    # 4. GAIL-RL-Rich (IM checkpoint → adversarial fine-tuning) ───────────
    logger.info("\n--- GAIL-RL-Rich: adversarial fine-tuning ---")
    gail_policy = train_gail_rl_rich(
        im_policy, cfg, train_graphs, traj_cache, im_statics,
        device, n_epochs=rl_epochs)
    results["GAIL-RL-Rich"] = eval_joint(gail_policy, test_graph, test_static,
                                         cfg, device)
    logger.info(f"  {'GAIL-RL-Rich':25s}: {results['GAIL-RL-Rich']:.4f}")

    # ── Summary table ──────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"  {label}")
    logger.info(f"  {'Method':<28} {'Revenue':>10}")
    logger.info("-" * 60)
    for method, rev in sorted(results.items(), key=lambda x: -x[1]):
        marker = " ◀ BEST" if rev == max(results.values()) else ""
        logger.info(f"  {method:<28} {rev:>10.4f}{marker}")
    logger.info('='*60)

    logger.log({f"benchmark_{graph_name}/{k}": v for k, v in results.items()})
    logger.finish()
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-graph benchmark for RevMax methods")
    parser.add_argument("--graph", type=str, default="all",
        choices=["rice_facebook", "ba500", "sbm500", "plow500", "all"],
        help="Graph to benchmark (default: all)")
    args = parser.parse_args()

    graphs = (["rice_facebook", "ba500", "sbm500", "plow500"]
              if args.graph == "all" else [args.graph])

    all_results = {}
    for g in graphs:
        print(f"\n\n{'#'*65}")
        print(f"# Running benchmark: {g}")
        print(f"{'#'*65}")
        all_results[g] = run_benchmark_graph(g)

    # Cross-graph summary (if --graph all)
    if args.graph == "all":
        print(f"\n{'='*65}")
        print("  CROSS-GRAPH SUMMARY")
        print(f"  {'Method':<22}", end="")
        for g in graphs:
            print(f"  {g:>14}", end="")
        print()
        print("-" * 65)
        all_methods = sorted({m for r in all_results.values() for m in r})
        for method in all_methods:
            print(f"  {method:<22}", end="")
            for g in graphs:
                val = all_results[g].get(method, float("nan"))
                print(f"  {val:>14.4f}", end="")
            print()
        print('='*65)


if __name__ == "__main__":
    main()
