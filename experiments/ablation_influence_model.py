"""
experiments/ablation_influence_model.py  (<80 lines)

Ablation: Monotone vs Non-Monotone influence model.
Trains Rev-GNN-IM-RL with each influence model, reports revenue comparison.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from src.utils.helpers import load_config_with_base, set_seed, get_device
from src.utils.logging import ExperimentLogger
from src.evaluation.baselines import run_all_baselines
from src.env.graph_generators import generate_graph


def main() -> None:
    cfg = load_config_with_base("configs/experiments/rev_gnn_im_rl.yaml")
    set_seed(cfg.project.seed)
    logger = ExperimentLogger(cfg, run_name="ablation_influence_model")

    graph = generate_graph(cfg, graph_type="ba", n=cfg.graph.n_nodes)

    for model_type in ["monotone", "non_monotone"]:
        variant_cfg = OmegaConf.merge(cfg, OmegaConf.create(
            {"influence": {"model": model_type}}
        ))
        results = run_all_baselines(graph, variant_cfg, n_trials=5)
        logger.info(f"\n=== influence_model={model_type} ===")
        for strategy, rev in results.items():
            logger.log({"model": model_type, "strategy": strategy, "revenue": rev})
            logger.info(f"  {strategy}: {rev:.4f}")

    logger.finish()


if __name__ == "__main__":
    main()
