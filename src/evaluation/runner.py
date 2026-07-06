"""
src/evaluation/runner.py

Lightweight policy evaluation utilities used by experiment scripts.

These are thin orchestration wrappers — all logic lives in env, features, and trainer.
"""

import torch

from src.evaluation.baselines import _make_env
from src.utils.features import (compute_static_features, compute_node_features,
                                  build_graph_feature_cache, compute_node_features_fast)
from src.utils.helpers import graph_to_pyg_data, get_available_mask

# Module-level caches — static features and graph feature cache are O(nm) and
# must not be recomputed every evaluation call (especially for n=1000 test graphs).
_STATIC_CACHE: dict = {}
_FEAT_CACHE: dict = {}


def eval_greedy_revenue(
    policy: torch.nn.Module,
    graph,
    cfg,
    device: torch.device,
) -> float:
    """Run one greedy episode with the joint policy; return total flat revenue.

    Uses estimate-based pricing and true-valuation acceptance (same as env.step).
    Called after Phase 1 (imitation) and every 10 Phase 2 (REINFORCE) epochs.

    Args:
        policy: JointPolicy with select_and_price() interface.
        graph:  NetworkX graph (typically the held-out test graph).
        cfg:    OmegaConf DictConfig.
        device: PyTorch device.

    Returns:
        Total flat revenue from the greedy episode.
    """
    env = _make_env(graph, cfg)
    env.reset()
    gid = id(graph)
    if gid not in _STATIC_CACHE:
        _STATIC_CACHE[gid] = compute_static_features(graph)
    if gid not in _FEAT_CACHE:
        _FEAT_CACHE[gid] = build_graph_feature_cache(graph, _STATIC_CACHE[gid])
    fcache = _FEAT_CACHE[gid]
    n = graph.number_of_nodes()
    nodes = list(graph.nodes())

    policy.eval()
    with torch.no_grad():
        for _ in range(n):
            if not env.available_nodes:
                break
            feats = compute_node_features_fast(
                cache=fcache, S=frozenset(env.S), offered=frozenset(env.offered),
                t=env.t, k=n, env=env,
            )
            data = graph_to_pyg_data(graph, feats, device)
            mask = get_available_mask(n, frozenset(env.offered), nodes, device)
            idx, disc, _ = policy.select_and_price(
                data.x, data.edge_index, mask, greedy=True
            )
            _, _, done, _ = env.step(idx, disc)
            if done:
                break
    policy.train()
    return env.total_revenue
