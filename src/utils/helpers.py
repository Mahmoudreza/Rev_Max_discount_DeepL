"""
src/utils/helpers.py

Utility functions for reproducibility, device management, config loading,
and PyG data conversion.

All experiment scripts must call set_seed(cfg.seed) at the top.
"""

import os
import random
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import networkx as nx

from omegaconf import OmegaConf, DictConfig


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set all random seeds for full reproducibility.

    Sets seeds for Python random, NumPy, PyTorch (CPU + CUDA).
    Call this at the top of every experiment script.

    Args:
        seed: Integer random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # For full determinism on GPU (may slow training):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU).

    Never hardcode "cuda" — always call this instead.

    Returns:
        torch.device for the best available accelerator.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Config loading ────────────────────────────────────────────────────────────

def load_config(
    path: Union[str, Path],
    overrides: Optional[List[str]] = None,
) -> DictConfig:
    """Load an OmegaConf config from a YAML file with optional overrides.

    Args:
        path: Path to the base YAML config file.
        overrides: Optional list of dot-notation overrides, e.g.
                   ["training.lr=0.001", "graph.n_nodes=500"].

    Returns:
        Merged OmegaConf DictConfig.
    """
    cfg = OmegaConf.load(path)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
    return cfg


def load_config_with_base(
    experiment_path: Union[str, Path],
    base_path: Union[str, Path] = "configs/base_config.yaml",
    overrides: Optional[List[str]] = None,
) -> DictConfig:
    """Load experiment config merged onto base config.

    Args:
        experiment_path: Path to experiment-specific YAML.
        base_path: Path to base config (default: configs/base_config.yaml).
        overrides: Optional CLI overrides.

    Returns:
        Merged DictConfig (experiment overrides base).
    """
    base_cfg = OmegaConf.load(base_path)
    exp_cfg = OmegaConf.load(experiment_path)
    cfg = OmegaConf.merge(base_cfg, exp_cfg)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
    return cfg


# ── Path utilities ────────────────────────────────────────────────────────────

def get_project_root() -> Path:
    """Return the absolute path to the project root (revmax-aaai2027/).

    Assumes this file is at src/utils/helpers.py.

    Returns:
        Path to project root directory.
    """
    return Path(__file__).parent.parent.parent.resolve()


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory (and parents) if it doesn't exist.

    Args:
        path: Directory path to create.

    Returns:
        The (now existing) Path object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── PyG data conversion ───────────────────────────────────────────────────────

def graph_to_pyg_data(
    graph: nx.Graph,
    features: np.ndarray,
    device: torch.device,
):
    """Convert a NetworkX graph + feature matrix to a PyG Data object.

    For undirected graphs, edges are added in both directions (bidirectional
    edge_index), as expected by PyG's message-passing layers.

    Args:
        graph: Undirected NetworkX graph with n nodes.
        features: Node feature matrix, shape (n, d).
        device: Target PyTorch device.

    Returns:
        torch_geometric.data.Data with fields:
          - x: float32 tensor, shape (n, d)
          - edge_index: long tensor, shape (2, 2|E|) for undirected graphs
    """
    from torch_geometric.data import Data

    x = torch.tensor(features, dtype=torch.float32, device=device)

    # Build bidirectional edge_index
    edges = list(graph.edges())
    if edges:
        src = [u for u, v in edges] + [v for u, v in edges]
        dst = [v for u, v in edges] + [u for u, v in edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)

    return Data(x=x, edge_index=edge_index)


def get_available_mask(
    n: int,
    offered: set,
    nodes: list,
    device: torch.device,
) -> torch.Tensor:
    """Build a boolean availability mask for node selection.

    Args:
        n: Total number of nodes.
        offered: Set of node identifiers (not indices) already offered.
        nodes: Ordered list mapping index → node identifier.
        device: PyTorch device.

    Returns:
        Boolean tensor of shape (n,): True for nodes not in ``offered``.
    """
    mask = torch.ones(n, dtype=torch.bool, device=device)
    for i, node in enumerate(nodes):
        if node in offered:
            mask[i] = False
    return mask
