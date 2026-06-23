"""
src/models/sequence/transformer_policy.py

Transformer sequence model layered on top of the GNN encoder.

Architecture:
  h_t = GNN(G, x_t)                   # node embeddings at step t
  s_t = mean_pool(h_t)                # global graph state
  context = TransformerEncoder([s_0, ..., s_t])  # all past states
  z_t = context[-1]                   # current context
  score_v = MLP(cat(h_v, z_t))        # scoring each node
  discount_v = Sigmoid(MLP(cat(h_v, z_t)))

Unlike LSTM, the Transformer attends over ALL previous timesteps simultaneously
(avoids vanishing gradients over long selling sequences).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class TransformerJointPolicy(nn.Module):
    """Transformer-augmented joint policy for sequential revenue maximization.

    Maintains a buffer of global states s_0, ..., s_t and applies
    a causal Transformer to compute the current episode context.

    Args:
        encoder: GNN encoder (GraphSAGEEncoder or GraphTransformerEncoder).
        hidden_dim: GNN embedding dimension (default 64).
        n_heads: Number of attention heads (default 4).
        n_layers: Number of Transformer encoder layers (default 2).
        ff_dim: Feed-forward hidden size (default 128).
        dropout: Dropout in Transformer (default 0.1).
        max_seq_len: Max number of steps before truncation (default 2000).
    """

    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 2000,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # Positional encoding
        self._build_pe(max_seq_len, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Scoring and pricing heads
        self.scoring_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.pricing_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Episode state buffer
        self._state_buffer: List[torch.Tensor] = []

    def _build_pe(self, max_len: int, d_model: int) -> None:
        """Build sinusoidal positional encoding buffer."""
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def reset_buffer(self) -> None:
        """Clear episode state buffer at the start of a new episode."""
        self._state_buffer = []

    def _get_context(self, global_state: torch.Tensor) -> torch.Tensor:
        """Append new state and compute Transformer context.

        Args:
            global_state: Current global state, shape (hidden_dim,).

        Returns:
            Context vector z_t, shape (hidden_dim,).
        """
        self._state_buffer.append(global_state.detach())

        # Truncate if too long
        if len(self._state_buffer) > self.max_seq_len:
            self._state_buffer = self._state_buffer[-self.max_seq_len:]

        T = len(self._state_buffer)
        seq = torch.stack(self._state_buffer, dim=0).unsqueeze(0)  # (1, T, d)

        # Add positional encoding
        seq = seq + self.pe[:, :T, :]

        # Apply Transformer (no mask needed — we use all past states)
        context = self.transformer(seq)  # (1, T, d)
        return context[:, -1, :].squeeze(0)  # (d,) — last timestep

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        available_mask: torch.Tensor,
        return_embeddings: bool = False,
    ) -> Tuple:
        """Compute scores and discounts for one step.

        Args:
            x: Node features (n, in_dim).
            edge_index: Edge index (2, |E|).
            available_mask: Boolean mask (n,).
            return_embeddings: If True, return h too.

        Returns:
            scores, masked_scores, [h]
        """
        h = self.encoder(x, edge_index)               # (n, hidden_dim)
        global_state = h.mean(dim=0)                  # (hidden_dim,)
        context = self._get_context(global_state)      # (hidden_dim,)

        # Expand context to all nodes
        ctx_expanded = context.unsqueeze(0).expand(h.size(0), -1)  # (n, hidden_dim)
        combined = torch.cat([h, ctx_expanded], dim=-1)             # (n, 2*hidden_dim)

        scores = self.scoring_head(combined).squeeze(-1)            # (n,)
        masked_scores = scores.masked_fill(~available_mask, float("-inf"))

        if return_embeddings:
            return scores, masked_scores, h
        return scores, masked_scores

    def select_and_price(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        available_mask: torch.Tensor,
        greedy: bool = True,
    ) -> Tuple[int, float, torch.Tensor]:
        """Select node and compute discount.

        Args:
            x: Node features (n, in_dim).
            edge_index: Edge index (2, |E|).
            available_mask: Boolean mask (n,).
            greedy: Argmax if True.

        Returns:
            node_idx, discount, log_prob
        """
        scores, masked_scores, h = self.forward(
            x, edge_index, available_mask, return_embeddings=True
        )

        if greedy:
            node_idx = int(masked_scores.argmax().item())
            log_prob_node = F.log_softmax(masked_scores, dim=0)[node_idx]
        else:
            probs = F.softmax(masked_scores, dim=0)
            dist = torch.distributions.Categorical(probs)
            node_idx_t = dist.sample()
            node_idx = int(node_idx_t.item())
            log_prob_node = dist.log_prob(node_idx_t)

        # Pricing using Transformer context
        global_state = h.mean(dim=0)
        context = self._state_buffer[-1] if self._state_buffer else global_state.detach()
        context_t = context.to(x.device) if not context.requires_grad else context
        combined_node = torch.cat([h[node_idx], context_t], dim=-1)
        discount_tensor = self.pricing_head(combined_node.unsqueeze(0)).squeeze()
        discount = float(discount_tensor.item())

        disc_dist = torch.distributions.Normal(discount_tensor, 0.1)
        log_prob_discount = disc_dist.log_prob(discount_tensor)

        return node_idx, discount, log_prob_node + log_prob_discount
