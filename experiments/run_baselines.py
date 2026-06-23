"""
experiments/run_baselines.py  (<80 lines)

Run Babaei et al. (2013) baselines on all configured graph types.
Logs results via ExperimentLogger. Output: results/logs/baselines_*.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx
from src.utils.helpers import load_config_with_base, set_seed, ensure_dir
from src.utils.logging import ExperimentLogger
from src.evaluation.baselines import run_all_baselines
from src.env.graph_generators import generate_graph


def main(cfg_path: str = "configs/experiments/rev_gnn_im_rl.yaml") -> None:
    cfg = load_config_with_base(cfg_path)
    set_seed(cfg.project.seed)
    logger = ExperimentLogger(cfg, run_name="baselines")

    graphs = {
        "ba_500": generate_graph(cfg, graph_type="ba", n=500),
        "ws_500": generate_graph(cfg, graph_type="ws", n=500),
        "er_500": generate_graph(cfg, graph_type="er", n=500),
    }

    for graph_name, graph in graphs.items():
        logger.info(f"Running baselines on {graph_name} (n={graph.number_of_nodes()})")
        results = run_all_baselines(graph, cfg, n_trials=cfg.evaluation.n_trials)
        for strategy, revenue in results.items():
            logger.log({
                "graph": graph_name,
                "strategy": strategy,
                "revenue": revenue,
            })
            logger.info(f"  {strategy}: {revenue:.4f}")

    logger.finish()


if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/experiments/rev_gnn_im_rl.yaml"
    main(cfg_path)
