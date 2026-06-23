"""
experiments/download_networks.py

Download the 5 real-world networks from SNAP and preprocess them.

Downloads to data/raw/, preprocesses to data/processed/ as NetworkX pickles.
Usage:
    python experiments/download_networks.py
"""

import gzip
import pickle
import shutil
import urllib.request
from pathlib import Path

import networkx as nx

# ── Network registry ──────────────────────────────────────────────────────────
# name → (url, raw_filename)
NETWORKS = {
    "facebook": (
        "https://snap.stanford.edu/data/CollegeMsg.txt.gz",
        "facebook.txt",
    ),
    "yeast": (
        "https://snap.stanford.edu/data/bio-yeast.txt.gz",
        "yeast.txt",
    ),
    "wiki": (
        "https://snap.stanford.edu/data/wiki-Vote.txt.gz",
        "wiki.txt",
    ),
    "newman": (
        "https://snap.stanford.edu/data/ca-CondMat.txt.gz",
        "newman.txt",
    ),
    "hep": (
        "https://snap.stanford.edu/data/cit-HepTh.txt.gz",
        "hep.txt",
    ),
}

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def download_and_unzip(url: str, dest_path: Path) -> None:
    """Download a .gz file from url and decompress to dest_path.

    Args:
        url: URL of the .gz file.
        dest_path: Destination path for the decompressed file.
    """
    gz_path = dest_path.with_suffix(".txt.gz")
    print(f"  Downloading {url} ...")
    urllib.request.urlretrieve(url, gz_path)
    print(f"  Decompressing to {dest_path} ...")
    with gzip.open(gz_path, "rb") as f_in:
        with open(dest_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()


def parse_edgelist(filepath: Path) -> nx.Graph:
    """Parse a SNAP edge-list file into an undirected NetworkX graph.

    Args:
        filepath: Path to the edge-list text file.

    Returns:
        Undirected NetworkX graph (largest connected component, 0-indexed).
    """
    G = nx.Graph()
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                if u != v:
                    G.add_edge(u, v)

    # Convert to 0-indexed integers
    G = nx.convert_node_labels_to_integers(G)

    # Take largest connected component
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        G = nx.convert_node_labels_to_integers(G)

    return G


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Downloading and preprocessing real-world networks")
    print("=" * 60)

    for name, (url, filename) in NETWORKS.items():
        raw_path = RAW_DIR / filename
        pkl_path = PROCESSED_DIR / f"{name}.pkl"

        print(f"\n[{name}]")

        if raw_path.exists():
            print(f"  Raw file already exists: {raw_path}")
        else:
            download_and_unzip(url, raw_path)

        print(f"  Parsing edge-list ...")
        G = parse_edgelist(raw_path)

        print(f"  Nodes: {G.number_of_nodes():,}  |  Edges: {G.number_of_edges():,}")

        with open(pkl_path, "wb") as f:
            pickle.dump(G, f)
        print(f"  Saved preprocessed graph → {pkl_path}")

    print("\n" + "=" * 60)
    print("All networks downloaded and preprocessed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
