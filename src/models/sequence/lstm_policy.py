"""
src/models/sequence/lstm_policy.py

LSTM sequence model layered on top of the GNN encoder.

Architecture:
  h_t = GNN(G, x_t)           # node embeddings at step t
  s_t = mean_pool(h_t)        # global graph state
  z_t = LSTM(z_{t-1}, s_t)   # sequential context
  score_v = MLP(cat(h_v, z_t))  # scoring each node
  discount_v = Sigmoid(MLP(cat(h_v, z_t)))  # pricing each node
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class LSTMJointPolicy(nn.Module):
    """LSTM-augmented joint policy for sequential revenue maximization.

    The LSTM maintains a hidden state across the T steps of one episode,
    allowing the policy to remember which nodes were previously offered
    and how the seed set has evolved.

    Args:
        encoder: GNN encoder (GraphSAGEEncoder or GraphTransformerEncoder).
        hidden_dim: GNN embedding dimension (default 64).
        lstm_hidden: LSTM hidden state size (default 64).
        lstm_n_layers: Number of LSTM layers (default 1).
    """

    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int = 64,
        lstm_hidden: int = 64,
        lstm_n_layers: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.lstm_hidden = lstm_hidden

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_n_layers,
            batch_first=True,
        )

        # Scoring head: cat(node_emb, lstm_state) → score
        self.scoring_head = nn.Sequential(
            nn.Linear(hidden_dim + lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Pricing head: cat(node_emb, lstm_state) → discount ∈ [0,1]
        self.pricing_head = nn.Sequential(
            nn.Linear(hidden_dim + lstm_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self._hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def reset_hidden(self) -> None:
        """Reset LSTM hidden state at the start of a new episode."""
        self._hidden = None

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
            return_embeddings: If True, also return h (node embeddings).

        Returns:
            scores: Raw node scores (n,).
            masked_scores: Scores with unavailable set to -inf (n,).
            h (optional): Node embeddings (n, hidden_dim).
        """
        # GNN encoding
        h = self.encoder(x, edge_index)             # (n, hidden_dim)
        global_state = h.mean(dim=0, keepdim=True)  # (1, hidden_dim)

        # LSTM step: update context with global state
        lstm_in = global_state.unsqueeze(0)  # (1, 1, hidden_dim)
        lstm_out, self._hidden = self.lstm(lstm_in, self._hidden)
        lstm_context = lstm_out.squeeze(0).squeeze(0)  # (lstm_hidden,)

        # Expand context to match node count
        context_expanded = lstm_context.unsqueeze(0).expand(h.size(0), -1)  # (n, lstm_hidden)

        # Node-level scoring and pricing
        combined = torch.cat([h, context_expanded], dim=-1)  # (n, hidden_dim + lstm_hidden)
        scores = self.scoring_head(combined).squeeze(-1)      # (n,)

        # Apply availability mask
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
            greedy: Argmax if True; sample if False.

        Returns:
            node_idx, discount, log_prob
        """
        scores, masked_scores, h = self.forward(
            x, edge_index, available_mask, return_embeddings=True
        )

        # Node selection
        if greedy:
            node_idx = int(masked_scores.argmax().item())
            log_prob_node = F.log_softmax(masked_scores, dim=0)[node_idx]
        else:
            probs = F.softmax(masked_scores, dim=0)
            dist = torch.distributions.Categorical(probs)
            node_idx_t = dist.sample()
            node_idx = int(node_idx_t.item())
            log_prob_node = dist.log_prob(node_idx_t)

        # Discount: use LSTM context + node embedding
        global_state = h.mean(dim=0, keepdim=True)
        lstm_in = global_state.unsqueeze(0)
        with torch.no_grad():
            lstm_out, _ = self.lstm(lstm_in, self._hidden)
        lstm_context = lstm_out.squeeze(0).squeeze(0)
        combined_node = torch.cat([h[node_idx], lstm_context], dim=-1)
        discount_tensor = self.pricing_head(combined_node.unsqueeze(0)).squeeze()
        discount = float(discount_tensor.item())

        # Discount distribution for log-prob
        disc_dist = torch.distributions.Normal(discount_tensor, 0.1)
        log_prob_discount = disc_dist.log_prob(discount_tensor)

        return node_idx, discount, log_prob_node + log_prob_discount
