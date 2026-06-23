"""
src/evaluation/evaluate.py

Evaluation utilities: run policy on test graphs, compute revenue metrics.
"""

import torch
import numpy as np
from typing import Dict, List

from src.utils.helpers import graph_to_pyg_data, get_available_mask
from src.utils.features import compute_static_features, compute_node_features
from src.evaluation.baselines import _make_env


def evaluate_policy(
    policy,
    test_graphs: list,
    cfg,
    device: torch.device,
    n_trials: int = 1,
    greedy: bool = True,
) -> Dict:
    """Evaluate policy on a list of test graphs.

    Runs greedy rollout on each graph and records total revenue.

    Args:
        policy: JointPolicy or subclass with select_and_price().
        test_graphs: List of NetworkX test graphs.
        cfg: OmegaConf DictConfig.
        device: PyTorch device.
        n_trials: Number of MC trials per graph (to average over link weights).
        greedy: If True, use greedy (deterministic) policy.

    Returns:
        Dict with: mean_revenue, std_revenue, per_graph_revenues.
    """
    policy.eval()
    all_revenues = []

    with torch.no_grad():
        for graph in test_graphs:
            graph_revenues = []
            for trial in range(n_trials):
                from src.evaluation.baselines import _override_seed
                trial_cfg = _override_seed(cfg, cfg.project.seed + trial)
                env = _make_env(graph, trial_cfg)
                env.reset()

                static = compute_static_features(graph)
                n = graph.number_of_nodes()
                nodes = list(graph.nodes())

                for step in range(n):
                    available = env.available_nodes
                    if not available:
                        break

                    features = compute_node_features(
                        graph=graph,
                        static_features=static,
                        S=frozenset(env.S),
                        offered=frozenset(env.offered),
                        t=env.t,
                        n=n,
                        k=n,
                        env=env,
                    )
                    data = graph_to_pyg_data(graph, features, device)
                    mask = get_available_mask(n, frozenset(env.offered), nodes, device)

                    node_idx, discount, _ = policy.select_and_price(
                        data.x, data.edge_index, mask, greedy=greedy
                    )

                    if node_idx not in available:
                        node_idx = available[0]

                    _, _, done, _ = env.step(node_idx, discount)
                    if done:
                        break

                graph_revenues.append(env.total_revenue)
            all_revenues.append(float(np.mean(graph_revenues)))

    policy.train()
    return {
        "mean_revenue": float(np.mean(all_revenues)),
        "std_revenue": float(np.std(all_revenues)),
        "per_graph_revenues": all_revenues,
    }


def evaluate_and_compare(
    policies: Dict[str, object],
    test_graphs: list,
    cfg,
    device: torch.device,
    n_trials: int = 5,
) -> Dict[str, Dict]:
    """Evaluate multiple policies and return comparison dict.

    Args:
        policies: Dict of name → policy.
        test_graphs: List of test graphs.
        cfg: OmegaConf config.
        device: PyTorch device.
        n_trials: MC trials per graph.

    Returns:
        Dict of name → evaluate_policy result.
    """
    results = {}
    for name, policy in policies.items():
        results[name] = evaluate_policy(policy, test_graphs, cfg, device, n_trials)
    return results
