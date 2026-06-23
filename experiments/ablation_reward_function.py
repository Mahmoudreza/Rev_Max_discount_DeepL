"""
experiments/ablation_reward_function.py  (<80 lines)

Ablation: Flat revenue (γ=1) vs NPV reward (γ=0.9).
Compares greedy-discount baseline under each reward type.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from src.utils.helpers import load_config_with_base, set_seed
from src.utils.logging import ExperimentLogger
from src.evaluation.baselines import greedy_discount
from src.env.graph_generators import generate_graph


def main() -> None:
    cfg = load_config_with_base("configs/experiments/rev_gnn_im_rl.yaml")
    set_seed(cfg.project.seed)
    logger = ExperimentLogger(cfg, run_name="ablation_reward_function")

    graph = generate_graph(cfg, graph_type="ba", n=cfg.graph.n_nodes)

    reward_types = [
        ("flat",  1.0),
        ("npv",   0.9),
        ("npv",   0.95),
        ("npv",   0.99),
    ]

    for reward_type, gamma in reward_types:
        variant_cfg = OmegaConf.merge(cfg, OmegaConf.create(
            {"reward": {"type": reward_type, "gamma": gamma}}
        ))
        revenues = []
        for seed_offset in range(5):
            from src.evaluation.baselines import _override_seed
            trial_cfg = _override_seed(variant_cfg, cfg.project.seed + seed_offset)
            rev = greedy_discount(graph, trial_cfg)
            revenues.append(rev)
        mean_rev = sum(revenues) / len(revenues)
        logger.info(f"  reward={reward_type} γ={gamma}: revenue={mean_rev:.4f}")
        logger.log({"reward_type": reward_type, "gamma": gamma, "revenue": mean_rev})

    logger.finish()


if __name__ == "__main__":
    main()
