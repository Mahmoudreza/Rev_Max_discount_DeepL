"""
src/env/graph_generators.py

Graph generators for the revenue maximization experiments.

Implements:
  - Forest Fire model (Leskovec et al. 2007)
  - Modular Forest Fire model (multi-module with inter-module rewiring)
  - Real network loaders (SNAP datasets)
  - build_graph_from_config(): factory function driven by OmegaConf config
"""

import random
import pickle
import os
from pathlib import Path
from typing import List, Optional

import networkx as nx
import numpy as np


# ── Forest Fire Model ─────────────────────────────────────────────────────────

def generate_forest_fire(
    n: int,
    p: float,
    pb: float,
    seed: Optional[int] = None,
) -> nx.Graph:
    """Generate a Forest Fire graph (Leskovec et al. 2007).

    Each new node v selects a random "ambassador" ambassador w already in the
    graph, then "burns" forward with probability p and backward with probability
    pb.  The result is an undirected graph with power-law degree and diameter
    shrinkage properties matching real social networks.

    Args:
        n: Number of nodes.
        p: Forward burning probability.
        pb: Backward burning probability.
        seed: Random seed for reproducibility.

    Returns:
        Undirected NetworkX graph with n nodes.
    """
    rng = random.Random(seed)

    # Internal directed graph (converted to undirected on return)
    G = nx.DiGraph()
    G.add_node(0)

    for v in range(1, n):
        G.add_node(v)

        # Step 1: choose a random ambassador from already-added nodes
        existing = list(range(v))
        ambassador = rng.choice(existing)

        # Step 2: BFS-like burning process
        visited = set()
        queue = [ambassador]

        while queue:
            w = queue.pop(0)
            if w in visited or w == v:
                continue
            visited.add(w)
            G.add_edge(v, w)

            # Burn forward links (out-neighbors of w)
            out_neighbors = list(G.successors(w))
            n_forward = np.random.default_rng(
                rng.randint(0, 2**31)
            ).geometric(p) - 1
            n_forward = min(n_forward, len(out_neighbors))
            if n_forward > 0:
                chosen = rng.sample(out_neighbors, n_forward)
                queue.extend(chosen)

            # Burn backward links (in-neighbors of w)
            in_neighbors = list(G.predecessors(w))
            n_backward = np.random.default_rng(
                rng.randint(0, 2**31)
            ).geometric(pb) - 1
            n_backward = min(n_backward, len(in_neighbors))
            if n_backward > 0:
                chosen = rng.sample(in_neighbors, n_backward)
                queue.extend(chosen)

    # Convert to undirected, take largest connected component
    UG = G.to_undirected()
    if not nx.is_connected(UG):
        largest_cc = max(nx.connected_components(UG), key=len)
        UG = UG.subgraph(largest_cc).copy()
        UG = nx.convert_node_labels_to_integers(UG)

    return UG


# ── Modular Forest Fire ───────────────────────────────────────────────────────

def generate_modular_forest_fire(
    module_sizes: List[int],
    p: float,
    pb: float,
    inter_prob: float,
    seed: Optional[int] = None,
) -> nx.Graph:
    """Generate a Modular Forest Fire graph.

    Builds n_modules independent Forest Fire graphs, then adds inter-module
    edges by rewiring with probability inter_prob between module pairs.

    Args:
        module_sizes: List of node counts per module (e.g. [200, 300, 500]).
        p: Forward burning probability (same for all modules).
        pb: Backward burning probability (same for all modules).
        inter_prob: Probability of an inter-module edge between any two nodes
                    from different modules.
        seed: Random seed.

    Returns:
        Undirected NetworkX graph (union of all modules + inter-module edges).
    """
    rng = random.Random(seed)

    modules = []
    offset = 0
    for i, size in enumerate(module_sizes):
        module_seed = rng.randint(0, 2**31) if seed is not None else None
        G_module = generate_forest_fire(size, p, pb, seed=module_seed)
        # Re-label nodes to avoid index collisions across modules
        mapping = {v: v + offset for v in G_module.nodes()}
        G_module = nx.relabel_nodes(G_module, mapping)
        modules.append(G_module)
        offset += G_module.number_of_nodes()

    # Combine all modules into one graph
    G = nx.Graph()
    for mod in modules:
        G = nx.compose(G, mod)

    # Add inter-module edges with probability inter_prob
    for i in range(len(modules)):
        nodes_i = list(modules[i].nodes())
        for j in range(i + 1, len(modules)):
            nodes_j = list(modules[j].nodes())
            for u in nodes_i:
                for v in nodes_j:
                    if rng.random() < inter_prob:
                        G.add_edge(u, v)

    return G


# ── Real network loaders ──────────────────────────────────────────────────────

# Mapping from friendly name → filename in data/raw/
_NETWORK_FILES = {
    "facebook": "facebook.txt",
    "yeast": "yeast.txt",
    "wiki": "wiki.txt",
    "newman": "newman.txt",
    "hep": "hep.txt",
}


def load_real_network(name: str, data_dir: str = "data/raw") -> nx.Graph:
    """Load a real-world network from an edge-list file.

    Reads data/raw/{name}.txt, strips comment lines (starting with '#'),
    parses each remaining line as 'u v' and adds undirected edges.

    Args:
        name: Network name. One of: "facebook", "yeast", "wiki", "newman", "hep".
        data_dir: Directory containing raw edge-list files.

    Returns:
        Undirected NetworkX Graph, node labels are integers.

    Raises:
        ValueError: If name is not a supported network.
        FileNotFoundError: If the edge-list file does not exist.
    """
    if name not in _NETWORK_FILES:
        raise ValueError(
            f"Unknown network '{name}'. Supported: {list(_NETWORK_FILES.keys())}"
        )

    filepath = Path(data_dir) / _NETWORK_FILES[name]
    if not filepath.exists():
        raise FileNotFoundError(
            f"Network file not found: {filepath}. "
            f"Run experiments/download_networks.py first."
        )

    # Try preprocessed pickle first
    pkl_path = Path(data_dir).parent / "processed" / f"{name}.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    G = nx.Graph()
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                if u != v:  # skip self-loops
                    G.add_edge(u, v)

    # Convert to 0-indexed integers
    G = nx.convert_node_labels_to_integers(G)

    # Take largest connected component
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        G = nx.convert_node_labels_to_integers(G)

    return G


# ── Config-driven factory ─────────────────────────────────────────────────────

def build_graph_from_config(cfg) -> nx.Graph:
    """Factory function: build the right graph based on cfg.graph settings.

    Reads ``cfg.graph.type`` and dispatches to the appropriate generator.

    Supported types:
      - "forest_fire"   → generate_forest_fire(cfg.graph.n_nodes, cfg.graph.p, cfg.graph.pb)
      - "modular_ff"    → generate_modular_forest_fire(module_sizes, p, pb, inter_prob)
      - "facebook" / "yeast" / "wiki" / "newman" / "hep" → load_real_network(type)

    Args:
        cfg: OmegaConf DictConfig with a ``graph`` sub-config.

    Returns:
        Undirected NetworkX graph.

    Raises:
        ValueError: If cfg.graph.type is not recognized.
    """
    graph_type = cfg.graph.type
    seed = getattr(cfg, "seed", 42)

    if graph_type == "forest_fire":
        return generate_forest_fire(
            n=cfg.graph.n_nodes,
            p=cfg.graph.p,
            pb=cfg.graph.pb,
            seed=seed,
        )
    elif graph_type == "modular_ff":
        return generate_modular_forest_fire(
            module_sizes=list(cfg.graph.module_sizes),
            p=cfg.graph.p,
            pb=cfg.graph.pb,
            inter_prob=cfg.graph.inter_module_prob,
            seed=seed,
        )
    elif graph_type in _NETWORK_FILES:
        data_dir = str(Path(cfg.project.output_dir).parent / "data" / "raw")
        # Try relative path first
        if not Path(data_dir).exists():
            data_dir = "data/raw"
        return load_real_network(graph_type, data_dir)
    else:
        raise ValueError(
            f"Unknown graph type '{graph_type}'. "
            f"Supported: forest_fire, modular_ff, {', '.join(_NETWORK_FILES.keys())}"
        )
