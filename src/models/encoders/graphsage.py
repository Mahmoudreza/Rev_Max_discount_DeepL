"""
src/models/encoders/graphsage.py

GraphSAGE encoder from WSDM 2027 (Section 4.2), extended to handle
the new 20-dim feature vector (was 12/16-dim in WSDM).

Architecture (Eq. 12-13 in WSDM paper):
  h_v^(0) = ReLU(LayerNorm(W_proj * phi(v)))    ∈ R^64
  h_v^(l) = ReLU(LayerNorm(h^(l-1) + W_self*h^(l-1) + W_neigh * A_hat * h^(l-1)))
  score_v  = Linear(32→1)(ReLU(Linear(64→32)(h_v^(2))))

This file is a direct port of the WSDM backbone — only the input projection
dimension changes (d=20 instead of d=12 or d=16).
"""

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class GraphSAGEEncoder(nn.Module):
    """Two-layer GraphSAGE encoder with residual connections and LayerNorm.

    Identical to WSDM 2027 Section 4.2. Input dimension is set by `in_dim`
    (20 for this paper vs. 12/16 in WSDM).

    Args:
        in_dim: Input node feature dimension (default 20 for this paper).
        hidden_dim: Hidden + output dimension (default 64, same as WSDM).
        n_layers: Number of SAGEConv layers (default 2, same as WSDM).
        dropout: Dropout rate (default 0.0, same as WSDM).
    """

    def __init__(
        self,
        in_dim: int = 20,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout

        # Input projection (Eq. 12 in WSDM)
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # GraphSAGE layers (Eq. 13 in WSDM)
        self.conv_layers = nn.ModuleList([
            SAGEConv(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])

        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute node embeddings.

        Args:
            x: Node feature matrix, shape (n, in_dim).
            edge_index: Edge connectivity, shape (2, |E|).

        Returns:
            Node embeddings h, shape (n, hidden_dim).
        """
        # h^(0): input projection + LayerNorm + ReLU (Eq. 12)
        h = self.act(self.input_norm(self.input_proj(x)))

        # h^(l): residual SAGEConv + LayerNorm + ReLU (Eq. 13)
        for conv, norm in zip(self.conv_layers, self.layer_norms):
            h_new = conv(h, edge_index)           # W_self * h + W_neigh * A_hat * h
            h = self.act(norm(h + h_new))         # residual connection
            h = self.drop(h)

        return h
