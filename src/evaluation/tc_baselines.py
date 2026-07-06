"""
src/evaluation/tc_baselines.py — Time-Critical Revenue Evaluation (Idea 2).

Sequential model only (Babaei et al.). No IC cascade. No TimeCriticalRevenueEnv.

Wraps the Idea 1 acceptance-curve collectors from paper_eval.py to produce
cumulative revenue curves indexed by the number of ACCEPTANCES (|S|).
Revenue at deadline τ = cumrev[τ-1], identical to Idea 1's revenue_at_k().

The SAME trajectories from Idea 1 can be evaluated under Idea 2 deadlines.
We just read off different positions in the existing acceptance curves.

Do NOT import TimeCriticalRevenueEnv or run_cascade anywhere in this file.
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import networkx as nx
from omegaconf import OmegaConf

from src.evaluation.baselines import (
    _make_env,
    _greedy_seed_selection,
    _invalidate_caches,
)
from src.evaluation.paper_eval import (
    ac_greedy,
    ac_lstm,
    ac_im_rl,
)
from src.evaluation.tc_evaluation import (
    revenue_at_checkpoints,
    evaluate_tc_comparison,
    revenue_area_under_curve,
)
from src.utils.helpers import ensure_dir


# ── IE-Strategy acceptance curve ─────────────────────────────────────────────

def ac_ie_strategy(
    graph: nx.Graph,
    cfg,
    precomputed_seeds: Optional[list] = None,
) -> List[float]:
    """IE-Strategy acceptance curve (sequential, acceptance-indexed).

    Phase 1: k=cfg.budget.k free seeds (always accepted, revenue=0).
             Adds k entries of 0.0 to the curve.
    Phase 2: offer remaining nodes at estimated valuation (myopic).
             Only append to curve when buyer ACCEPTS (true_val >= price).

    The curve is indexed by |S| so that revenue_at_checkpoints(curve, τ)
    correctly reads revenue after τ ACCEPTANCES.

    Expected: curve is flat at 0.0 for first k entries (free seeds).
    Deadline collapse: IE earns 0 at τ ≤ cfg.budget.k.

    Args:
        graph:             NetworkX social network graph.
        cfg:               OmegaConf config (uses cfg.budget.k).
        precomputed_seeds: Optional pre-computed seed list from
                           _greedy_seed_selection. Pass this when calling
                           across multiple trials for the same graph to avoid
                           rerunning the expensive greedy-IM selection
                           (seeds depend only on graph structure, not on the
                           link-weight sample). Computed internally if None.

    Returns:
        cum_rev_by_S: Acceptance-indexed cumulative revenue list.
    """
    env = _make_env(graph, cfg)
    env.reset()

    k = int(cfg.budget.k)
    cum_rev: List[float] = []
    total = 0.0

    # Phase 1: k free seeds — reuse precomputed seeds if available,
    # otherwise run greedy-IM (expensive: O(k × n × n_mc_samples))
    seed_set = (precomputed_seeds
                if precomputed_seeds is not None
                else _greedy_seed_selection(graph, env, k))
    for node in seed_set:
        cum_rev.append(total)        # revenue stays 0.0 for each free seed
        env.offered.add(node)
        env.S.add(node)
        env.t += 1

    # Phase 2: offer remaining nodes at full estimated valuation (myopic)
    remaining = sorted(
        [v for v in env.nodes if v not in env.offered],
        key=lambda v: -env._estimate_valuation(v),
    )
    for node in remaining:
        est_val = env._estimate_valuation(node)
        if est_val > 0:
            true_val = env._true_valuation(node)
            if true_val >= est_val:
                env.S.add(node)
                _invalidate_caches(env, node)
                total += est_val
                cum_rev.append(total)   # only on acceptance
        env.offered.add(node)
        env.t += 1

    return cum_rev


# ── Multi-trial curve collection ─────────────────────────────────────────────

def collect_tc_curves(
    graph: nx.Graph,
    cfg,
    lstm_pol=None,
    im_pol=None,
    device: Optional[torch.device] = None,
    n_trials: int = 10,
) -> Dict[str, List[List[float]]]:
    """Collect acceptance-indexed cum_rev curves for all 4 methods.

    Each trial uses a different random seed (different link-weight sample).
    The curves are indexed by |S| (consecutive acceptances in the
    sequential Babaei model — no separate cascade phase).

    Args:
        graph:    NetworkX social network graph.
        cfg:      OmegaConf config.
        lstm_pol: Loaded SequentialJointPolicy (Rev-GNN-LSTM). None to skip.
        im_pol:   Loaded JointPolicy (Rev-GNN-IM-RL). None to skip.
        device:   PyTorch device.
        n_trials: Number of link-weight trials (MC variance).

    Returns:
        Dict: method_name → list of cum_rev_by_S curves (one per trial).
    """
    if device is None:
        device = torch.device("cpu")

    base_seed = int(getattr(cfg.project, "seed", 42))

    curves: Dict[str, List[List[float]]] = {
        "IE-Strategy":     [],
        "Greedy-Discount": [],
    }
    if lstm_pol is not None:
        curves["Rev-GNN-LSTM"] = []
    if im_pol is not None:
        curves["Rev-GNN-IM-RL"] = []

    # Pre-compute greedy-IM seeds ONCE per graph.
    # The seed set depends only on graph structure (expected influence),
    # not on the link-weight sample used per revenue trial.
    # This avoids rerunning O(k × n × n_mc_samples) greedy-IM per trial.
    _env0 = _make_env(graph, cfg)
    _env0.reset()
    ie_seeds = _greedy_seed_selection(graph, _env0, int(cfg.budget.k))

    for trial in range(n_trials):
        # New link-weight sample for each trial
        cfg_t = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        OmegaConf.update(cfg_t, "project.seed", base_seed + trial)

        # IE-Strategy — reuse precomputed seeds across trials
        curves["IE-Strategy"].append(ac_ie_strategy(graph, cfg_t, precomputed_seeds=ie_seeds))

        # Greedy-Discount
        g_cum, _, _ = ac_greedy(graph, cfg_t)
        curves["Greedy-Discount"].append(g_cum)

        # RL policies
        if lstm_pol is not None:
            l_cum, _, _ = ac_lstm(lstm_pol, graph, cfg_t, device)
            curves["Rev-GNN-LSTM"].append(l_cum)

        if im_pol is not None:
            i_cum, _, _ = ac_im_rl(im_pol, graph, cfg_t, device)
            curves["Rev-GNN-IM-RL"].append(i_cum)

    return curves


# ── Full TC comparison on one graph ──────────────────────────────────────────

def run_tc_comparison(
    graph: nx.Graph,
    cfg,
    checkpoints: List[int],
    lstm_pol=None,
    im_pol=None,
    tc_lstm_pol=None,
    device: Optional[torch.device] = None,
    n_trials: int = 10,
) -> Dict[str, Dict]:
    """Compare all methods on one graph at multiple τ checkpoints.

    Args:
        graph:       NetworkX social network graph.
        cfg:         OmegaConf config.
        checkpoints: List of τ values.
        lstm_pol:    Loaded SequentialJointPolicy (Rev-GNN-LSTM). Optional.
        im_pol:      Loaded JointPolicy (Rev-GNN-IM-RL). Optional.
        tc_lstm_pol: Optional TC-fine-tuned LSTM (Rev-GNN-LSTM-TC).
        device:      PyTorch device.
        n_trials:    Number of link-weight trials.

    Returns:
        Dict: method_name → {checkpoints, checkpoints_std, area, area_std, n_trials}
        (Same format as src.evaluation.tc_evaluation.evaluate_tc_comparison)
    """
    if device is None:
        device = torch.device("cpu")

    curves = collect_tc_curves(
        graph, cfg,
        lstm_pol=lstm_pol, im_pol=im_pol,
        device=device, n_trials=n_trials,
    )

    # Optional TC-fine-tuned LSTM as an extra method
    if tc_lstm_pol is not None:
        tc_curves: List[List[float]] = []
        base_seed = int(getattr(cfg.project, "seed", 42))
        for trial in range(n_trials):
            cfg_t = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            OmegaConf.update(cfg_t, "project.seed", base_seed + trial)
            l_cum, _, _ = ac_lstm(tc_lstm_pol, graph, cfg_t, device)
            tc_curves.append(l_cum)
        curves["Rev-GNN-LSTM-TC"] = tc_curves

    return evaluate_tc_comparison(curves, checkpoints)


# ── Multi-graph TC comparison ─────────────────────────────────────────────────

def run_tc_comparison_multi_graph(
    graph_fn,
    cfg,
    checkpoints: List[int],
    n_graphs: int = 10,
    lstm_pol=None,
    im_pol=None,
    tc_lstm_pol=None,
    device: Optional[torch.device] = None,
    n_trials: int = 5,
    out_path: Optional[str] = None,
) -> Dict[str, Dict]:
    """TC comparison averaged across n_graphs graph instances.

    For the paper: n_graphs=10 (different graph structures), n_trials=5
    (different link-weight samples per graph) → 50 total data points per method.

    Args:
        graph_fn:    Callable(seed: int) → nx.Graph.
        cfg:         OmegaConf config.
        checkpoints: List of τ values.
        n_graphs:    Number of graph structure seeds.
        lstm_pol:    Loaded LSTM policy.
        im_pol:      Loaded IM-RL policy.
        tc_lstm_pol: Optional TC-trained LSTM.
        device:      Torch device.
        n_trials:    Link-weight trials per graph.
        out_path:    Optional JSON cache path.

    Returns:
        Dict compatible with evaluate_tc_comparison output, averaged over
        all n_graphs × n_trials combinations.
    """
    if device is None:
        device = torch.device("cpu")

    # Collect ALL curves from all graph seeds
    all_methods_curves: Dict[str, List[List[float]]] = {}

    for gseed in range(n_graphs):
        graph = graph_fn(gseed)
        curves = collect_tc_curves(
            graph, cfg,
            lstm_pol=lstm_pol, im_pol=im_pol,
            device=device, n_trials=n_trials,
        )
        if tc_lstm_pol is not None:
            base_seed = int(getattr(cfg.project, "seed", 42))
            tc_cs: List[List[float]] = []
            for t2 in range(n_trials):
                cfg_t2 = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
                OmegaConf.update(cfg_t2, "project.seed", base_seed + t2)
                lc, _, _ = ac_lstm(tc_lstm_pol, graph, cfg_t2, device)
                tc_cs.append(lc)
            curves["Rev-GNN-LSTM-TC"] = tc_cs

        for method, cs in curves.items():
            all_methods_curves.setdefault(method, []).extend(cs)

        print(f"    graph seed {gseed+1}/{n_graphs} done", flush=True)

    results = evaluate_tc_comparison(all_methods_curves, checkpoints)

    if out_path:
        ensure_dir(os.path.dirname(out_path))
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  [✓] saved {out_path}")

    return results


# ── Load / save helpers ───────────────────────────────────────────────────────

def save_tc_results(results: Dict, path: str) -> None:
    """Save TC comparison results to JSON cache.

    Args:
        results: Output of run_tc_comparison or evaluate_tc_comparison.
        path:    Output JSON path.
    """
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def load_tc_results(path: str) -> Dict:
    """Load TC results from JSON cache.

    Args:
        path: Path to JSON file.

    Returns:
        Dict from evaluate_tc_comparison.
    """
    with open(path) as f:
        return json.load(f)
