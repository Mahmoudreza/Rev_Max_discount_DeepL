"""
experiments/ablation_encoder_type.py  (<80 lines)

Ablation: GraphSAGE vs Graph Transformer encoder.
Compares forward pass shapes and parameter counts.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from omegaconf import OmegaConf
from src.utils.helpers import load_config_with_base, set_seed, get_device, graph_to_pyg_data
from src.utils.logging import ExperimentLogger
from src.utils.features import compute_static_features, compute_node_features
from src.env.graph_generators import generate_graph
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.graph_transformer import GraphTransformerEncoder


def main() -> None:
    cfg = load_config_with_base("configs/experiments/rev_gnn_im_rl.yaml")
    set_seed(cfg.project.seed)
    device = get_device()
    logger = ExperimentLogger(cfg, run_name="ablation_encoder_type")

    graph = generate_graph(cfg, graph_type="ba", n=100)
    static = compute_static_features(graph)
    from src.evaluation.baselines import _make_env
    env = _make_env(graph, cfg)
    env.reset()
    features = compute_node_features(
        graph=graph, static_features=static, S=frozenset(),
        offered=frozenset(), t=0, n=100, k=100, env=env,
    )
    data = graph_to_pyg_data(graph, features, device)

    encoders = {
        "graphsage": GraphSAGEEncoder(in_dim=20, hidden_dim=64, n_layers=2).to(device),
        "graph_transformer": GraphTransformerEncoder(in_dim=20, hidden_dim=64, n_layers=2, n_heads=4).to(device),
    }

    for name, enc in encoders.items():
        n_params = sum(p.numel() for p in enc.parameters())
        h = enc(data.x, data.edge_index)
        logger.info(f"{name}: params={n_params:,}, output_shape={tuple(h.shape)}")
        logger.log({"encoder": name, "n_params": n_params, "output_dim": h.shape[-1]})

    logger.finish()


if __name__ == "__main__":
    main()
