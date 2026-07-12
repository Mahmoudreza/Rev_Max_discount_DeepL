#!/usr/bin/env python
"""
experiments/eval_transformer_gate_a.py — Gate A evaluation for Rev-GNN-Transformer.

Runs same protocol as eval_idea1.py but for TransformerJointPolicy:
  Task 1: FF n=1000  (20 seeds)          — Gate A primary metric
  Task 2: FF n=500 / n=2000, Modular FF, Rice-FB (5 seeds each) — OOD

GATE A verdict (printed at end):
  PASS  iff  TFM FF n=1000 mean >= (LSTM_BASELINE - 5.0)
          OR TFM beats LSTM on >= 2 of the 3 OOD networks
  FAIL  → Transformer = one future-work sentence; stop here.

Usage (after training completes):
  cd revmax-aaai2027 && source venv/bin/activate
  python experiments/eval_transformer_gate_a.py \
    --config configs/experiments/rev_gnn_transformer_300ep.yaml \
    --ckpt   results/checkpoints/rev_gnn_transformer.pt
"""
import argparse, json, os, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.utils.helpers import (load_config_with_base, set_seed, ensure_dir,
                                graph_to_pyg_data, get_available_mask)
from src.utils.logging import ExperimentLogger
from src.utils.features import (compute_static_features, build_graph_feature_cache,
                                 compute_node_features_fast)
from src.env.graph_generators import (generate_forest_fire,
                                       generate_modular_forest_fire,
                                       load_rice_facebook)
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.episode_transformer import EpisodeTransformerSliding
from src.models.policies.transformer_joint_policy import TransformerJointPolicy
from src.evaluation.baselines import _make_env

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Gate A reference values ───────────────────────────────────────────────────
LSTM_FF1000        = 462.6    # LSTM mean over 20 seeds (from training log)
GATE_A_FF_THRESH   = LSTM_FF1000 - 5.0   # = 457.6

# LSTM OOD means from paper_gen_updated.json (single-seed, indicative)
LSTM_OOD = {
    "FF n=500":       217.0,
    "FF n=2000":      915.0,
    "Modular FF":     414.4,
    "Rice-FB n=443":  214.1,
}


# ── Policy loader ─────────────────────────────────────────────────────────────

def load_transformer_policy(ckpt_path: str, cfg, device: torch.device) -> TransformerJointPolicy:
    enc = GraphSAGEEncoder(
        int(cfg.features.dim), int(cfg.encoder.hidden_dim),
        int(cfg.encoder.n_layers), float(cfg.encoder.dropout),
    )
    tfm = EpisodeTransformerSliding.from_config(cfg.transformer)
    policy = TransformerJointPolicy(
        enc, tfm,
        gnn_dim=int(cfg.encoder.hidden_dim),
        context_dim=tfm.context_dim,
    )
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    policy.load_state_dict(state)
    policy.to(device).eval()
    return policy


# ── Episode evaluator ─────────────────────────────────────────────────────────

def _eval_one(policy, graph, cfg, device) -> float:
    """Greedy revenue on a single graph. Returns float revenue."""
    n     = graph.number_of_nodes()
    nodes = list(graph.nodes())
    statics = compute_static_features(graph)
    cache   = build_graph_feature_cache(graph, statics)
    env = _make_env(graph, cfg)
    env.reset()
    policy.reset_episode(device)
    S, off = frozenset(), frozenset()
    revenue = 0.0
    with torch.no_grad():
        for _ in range(n):
            feats = compute_node_features_fast(
                cache=cache, S=S, offered=off, t=len(off), k=n, env=env)
            data = graph_to_pyg_data(graph, feats, device)
            mask = get_available_mask(n, off, nodes, device)
            if mask.sum() == 0:
                break
            ni, disc, _ = policy.select_and_price(
                data.x, data.edge_index, mask, greedy=True)
            node = nodes[int(ni)]
            val   = env._true_valuation(node)
            d     = float(disc)
            price = val * (1 - d)
            acc   = val >= price
            off = frozenset(off | {node}); env.offered.add(node); env.t += 1
            if acc:
                S = frozenset(S | {node}); env.S.add(node)
                env._influence_cache = {}
                revenue += price
            policy.update_sequence_state(d, acc, price if acc else 0.0)
    return revenue


def _run_seeds(policy, graph_fn, n_seeds, cfg, device, label, logger) -> dict:
    revs = []
    for s in range(n_seeds):
        g = graph_fn(s)
        r = _eval_one(policy, g, cfg, device)
        revs.append(float(r))
        logger.info(f"  {label}  seed={s}  rev={r:.2f}")
    m, std = float(np.mean(revs)), float(np.std(revs))
    logger.info(f"  → {label}  mean={m:.2f} ± {std:.2f}")
    return {"network": label, "mean": m, "std": std, "all": [float(x) for x in revs]}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Transformer Gate A evaluation")
    parser.add_argument("--config", default="configs/experiments/rev_gnn_transformer_300ep.yaml")
    parser.add_argument("--ckpt",   default="results/checkpoints/rev_gnn_transformer.pt")
    parser.add_argument("--seeds-t1", type=int, default=20)
    parser.add_argument("--seeds-t2", type=int, default=5)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--out", default="results/logs/transformer_gate_a.json")
    args = parser.parse_args()

    cfg    = load_config_with_base(args.config)
    set_seed(cfg.project.seed)
    device = torch.device("cpu")   # CPU for reproducibility
    logger = ExperimentLogger(cfg, run_name="transformer_gate_a")
    ensure_dir("results/logs")

    logger.info(f"Loading checkpoint: {args.ckpt}")
    policy = load_transformer_policy(args.ckpt, cfg, device)
    logger.info(f"Policy: {sum(p.numel() for p in policy.parameters()):,} params on {device}")
    logger.info(f"Gate A threshold: TFM FF n=1000 >= {GATE_A_FF_THRESH:.1f}"
                f" OR beats LSTM on >=2/3 OOD networks")

    results = {}

    # ── Task 1: FF n=1000 (20 seeds) ─────────────────────────────────────────
    logger.info(f"\n=== Task 1: FF n=1000 ({args.seeds_t1} seeds) ===")
    p, pb = cfg.graph.p, cfg.graph.pb
    t1 = _run_seeds(
        policy,
        lambda s: generate_forest_fire(1000, p, pb, seed=s),
        args.seeds_t1, cfg, device, "FF n=1000", logger,
    )
    results["FF n=1000"] = t1
    ff1000_mean = t1["mean"]

    # ── Task 2: OOD networks (5 seeds each) ───────────────────────────────────
    logger.info(f"\n=== Task 2: OOD generalisation ({args.seeds_t2} seeds each) ===")

    t2_ff500 = _run_seeds(
        policy,
        lambda s: generate_forest_fire(500, p, pb, seed=s),
        args.seeds_t2, cfg, device, "FF n=500", logger,
    )
    results["FF n=500"] = t2_ff500

    t2_ff2000 = _run_seeds(
        policy,
        lambda s: generate_forest_fire(2000, p, pb, seed=s),
        args.seeds_t2, cfg, device, "FF n=2000", logger,
    )
    results["FF n=2000"] = t2_ff2000

    t2_modff = _run_seeds(
        policy,
        lambda s: generate_modular_forest_fire([200, 300, 500], p, pb, 0.01, seed=s),
        args.seeds_t2, cfg, device, "Modular FF", logger,
    )
    results["Modular FF"] = t2_modff

    # Rice-FB: same graph topology, 5 seed evals
    try:
        rf_graph = load_rice_facebook(data_dir=args.data_dir)
        t2_rice = _run_seeds(
            policy,
            lambda s: rf_graph,
            args.seeds_t2, cfg, device, "Rice-FB n=443", logger,
        )
        results["Rice-FB n=443"] = t2_rice
    except FileNotFoundError as e:
        logger.info(f"  [SKIP] Rice-FB: {e}")
        results["Rice-FB n=443"] = {"mean": float("nan"), "std": float("nan")}

    # ── Gate A verdict ────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("GATE A EVALUATION")
    logger.info("=" * 70)

    # Criterion 1: FF n=1000 within 5 pts of LSTM
    crit1 = ff1000_mean >= GATE_A_FF_THRESH
    logger.info(f"Criterion 1 — FF n=1000 >= {GATE_A_FF_THRESH:.1f}: "
                f"TFM={ff1000_mean:.2f}  →  {'PASS ✓' if crit1 else 'fail'}")

    # Criterion 2: beats LSTM on >= 2 of 3 OOD networks
    ood_wins = 0
    for net, lstm_v in [("FF n=500",  LSTM_OOD["FF n=500"]),
                         ("FF n=2000", LSTM_OOD["FF n=2000"]),
                         ("Modular FF", LSTM_OOD["Modular FF"])]:
        tfm_v = results.get(net, {}).get("mean", float("nan"))
        win   = bool(not np.isnan(tfm_v) and tfm_v > lstm_v)
        ood_wins += int(win)
        logger.info(f"  OOD {net:20s}: TFM={tfm_v:.2f}  LSTM={lstm_v:.2f}  "
                    f"{'TFM wins ✓' if win else 'LSTM wins'}")
    crit2 = ood_wins >= 2
    logger.info(f"Criterion 2 — OOD wins >= 2: {ood_wins}/3  →  {'PASS ✓' if crit2 else 'fail'}")

    gate_a_pass = crit1 or crit2
    verdict_str = "GATE A: PASS ✓ — Transformer is competitive" if gate_a_pass \
                  else "GATE A: FAIL — Transformer is future work"
    logger.info(f"\n{'='*70}")
    logger.info(verdict_str)
    logger.info(f"{'='*70}")

    # Save results
    results["_gate_a"] = {
        "ff1000_mean":    float(ff1000_mean),
        "lstm_baseline":  float(LSTM_FF1000),
        "gate_a_thresh":  float(GATE_A_FF_THRESH),
        "crit1_pass":     bool(crit1),
        "crit2_ood_wins": int(ood_wins),
        "crit2_pass":     bool(crit2),
        "gate_a_pass":    bool(gate_a_pass),
        "verdict":        str(verdict_str),
    }
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved → {args.out}")
    logger.finish()


if __name__ == "__main__":
    main()
