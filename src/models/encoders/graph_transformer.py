"""
src/models/encoders/graph_transformer.py

Graph Transformer encoder — same interface as GraphSAGEEncoder but uses
PyG's TransformerConv instead of SAGEConv.

Architecture:
  - Input projection: Linear(in_dim, hidden_dim) + LayerNorm + ReLU
  - 2x TransformerConv(hidden_dim, hidden_dim, heads=4, concat=False) + residual
  - Same LayerNorm and residual pattern as GraphSAGEEncoder
"""

import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv


class GraphTransformerEncoder(nn.Module):
    """Two-layer Graph Transformer encoder with residual connections and LayerNorm.

    Drop-in replacement for GraphSAGEEncoder — same __init__ args and
    forward signature.  Uses TransformerConv (Shi et al., 2021) for
    attention-based neighbourhood aggregation.

    Args:
        in_dim: Input node feature dimension (default 20).
        hidden_dim: Hidden + output dimension (default 64).
        n_layers: Number of TransformerConv layers (default 2).
        n_heads: Number of attention heads (default 4).
        dropout: Dropout rate (default 0.1).
    """

    def __init__(
        self,
        in_dim: int = 20,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Input projection (mirrors GraphSAGEEncoder)
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # TransformerConv layers: concat=False → output is hidden_dim (not n_heads*head_dim)
        self.conv_layers = nn.ModuleList([
            TransformerConv(hidden_dim, hidden_dim, heads=n_heads, concat=False)
            for _ in range(n_layers)
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
        # h^(0): input projection + LayerNorm + ReLU
        h = self.act(self.input_norm(self.input_proj(x)))

        # h^(l): residual TransformerConv + LayerNorm + ReLU
        for conv, norm in zip(self.conv_layers, self.layer_norms):
            h_new = conv(h, edge_index)
            h = self.act(norm(h + h_new))
            h = self.drop(h)

        return h
