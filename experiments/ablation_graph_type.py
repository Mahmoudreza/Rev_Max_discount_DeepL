"""
experiments/ablation_graph_type.py  (<80 lines)

Ablation: BA vs WS vs ER graphs.
Runs all 4 baselines on each graph type, reports comparison table.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.helpers import load_config_with_base, set_seed
from src.utils.logging import ExperimentLogger
from src.evaluation.baselines import run_all_baselines
from src.env.graph_generators import generate_graph


def main() -> None:
    cfg = load_config_with_base("configs/experiments/rev_gnn_im_rl.yaml")
    set_seed(cfg.project.seed)
    logger = ExperimentLogger(cfg, run_name="ablation_graph_type")

    graph_types = ["ba", "ws", "er"]
    n = cfg.graph.n_nodes

    for graph_type in graph_types:
        graph = generate_graph(cfg, graph_type=graph_type, n=n)
        results = run_all_baselines(graph, cfg, n_trials=5)
        logger.info(f"\n=== graph_type={graph_type} (n={n}) ===")
        for strategy, rev in results.items():
            logger.log({"graph_type": graph_type, "strategy": strategy, "revenue": rev})
            logger.info(f"  {strategy}: {rev:.4f}")

    logger.finish()


if __name__ == "__main__":
    main()
