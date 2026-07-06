"""
src/models/policies/sequential_joint_policy.py

Joint policy with episode-level sequence memory (LSTM or Transformer).

Extends JointPolicy by concatenating the GNN node embeddings with
a context vector from an LSTM or Transformer over the step history.

At each step t:
  1. GNN encoder     → node embeddings H_t ∈ R^(n×64)
  2. Global mean pool → graph state g_t ∈ R^64
  3. Sequence model  → context c_t ∈ R^64  (from LSTM hidden state or Transformer CLS)
  4. For each node v:
       score_v   = scoring_head([H_t[v] ‖ c_t])    ← 128-dim input
       v*        = argmax(masked scores)
       discount  = pricing_head([H_t[v*] ‖ c_t])   ← 128-dim input

Why this matters:
  The GNN captures spatial structure (who is connected).
  The sequence model captures temporal structure:
    - Whether past buyers rejected offers (and at what discount)
    - How quickly influence is spreading through the network
    - Price sensitivity patterns in this particular graph instance

This is the strongest model in the family and the key architectural contribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

from src.models.encoders.sequence_models import EpisodeLSTM, EpisodeTransformer


class SequentialJointPolicy(nn.Module):
    """GNN + sequence model joint policy for revenue maximization.

    Args:
        encoder: GNN encoder (GraphSAGE or GraphTransformer), output dim = 64.
        sequence_model: EpisodeLSTM or EpisodeTransformer instance.
        gnn_dim: GNN output dimension (default 64).
        context_dim: Sequence model context dimension (default 64).
        scoring_hidden: Hidden dim of scoring MLP (default 32).
        pricing_hidden: Hidden dim of pricing MLP (default 32).
    """

    def __init__(
        self,
        encoder: nn.Module,
        sequence_model: nn.Module,   # EpisodeLSTM or EpisodeTransformer
        gnn_dim: int = 64,
        context_dim: int = 64,
        scoring_hidden: int = 32,
        pricing_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.sequence_model = sequence_model
        self.gnn_dim = gnn_dim
        self.context_dim = context_dim

        combined_dim = gnn_dim + context_dim   # 128

        # Scoring head: [H_v ‖ c_t] → score
        self.scoring_head = nn.Sequential(
            nn.Linear(combined_dim, scoring_hidden),
            nn.ReLU(),
            nn.Linear(scoring_hidden, 1),
        )

        # Pricing head: [H_{v*} ‖ c_t] → Beta(α, β) distribution over discount
        # Outputs 2 raw values → softplus + 1 → α > 1, β > 1 (unimodal Beta)
        self.pricing_head = nn.Sequential(
            nn.Linear(combined_dim, pricing_hidden),
            nn.ReLU(),
            nn.Linear(pricing_hidden, 2),   # 2 outputs: raw_alpha, raw_beta
        )

        # Last entropy for trainer entropy regularisation
        self._last_entropy: torch.Tensor = torch.tensor(0.0)

        # Detect which sequence model we have
        self._is_lstm = isinstance(sequence_model, EpisodeLSTM)
        self._is_transformer = isinstance(sequence_model, EpisodeTransformer)

        # Episode state (reset each episode)
        self._lstm_hidden = None
        self._token_history: List[torch.Tensor] = []

    def get_discount_distribution(self, combined: torch.Tensor) -> torch.distributions.Beta:
        """Get Beta distribution for the discount given a combined [h ‖ c] vector.

        Args:
            combined: Concatenated node+context embedding, shape (combined_dim,)
                      or (1, combined_dim).

        Returns:
            Beta distribution with α > 1 and β > 1 (guaranteed unimodal).
        """
        if combined.dim() == 1:
            combined = combined.unsqueeze(0)
        raw = self.pricing_head(combined).squeeze(0)   # (2,)
        alpha = F.softplus(raw[0]) + 1.0
        beta  = F.softplus(raw[1]) + 1.0
        return torch.distributions.Beta(alpha, beta)

    def reset_episode(self, device: torch.device) -> None:
        """Reset internal sequence model state at the start of each episode.

        Call this at the beginning of every episode (env.reset()).

        Args:
            device: Target device.
        """
        if self._is_lstm:
            self._lstm_hidden = self.sequence_model.init_hidden(device)
        self._token_history = []
        self._last_discount = 0.0
        self._last_accepted = False
        self._last_revenue = 0.0

    def update_sequence_state(
        self,
        discount: float,
        accepted: bool,
        revenue: float,
    ) -> None:
        """Update the sequence model state after each step.

        Call this AFTER env.step() returns the outcome.

        Args:
            discount: Discount offered this step.
            accepted: Whether buyer accepted.
            revenue: Revenue collected this step.
        """
        self._last_discount = discount
        self._last_accepted = accepted
        self._last_revenue = revenue

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        available_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: compute scores and discount for current step.

        Args:
            x: Node features (n, 20).
            edge_index: Edge index (2, |E|).
            available_mask: Boolean mask (n,), True for available nodes.

        Returns:
            masked_scores: Scores with unavailable nodes → -inf, shape (n,).
            embeddings: GNN node embeddings (n, 64).
            context: Sequence model context c_t (context_dim,).
            graph_embedding: Mean-pooled graph state (gnn_dim,).
        """
        # ── GNN encoder ──────────────────────────────────────────────────────
        h = self.encoder(x, edge_index)             # (n, 64)
        graph_emb = h.mean(dim=0)                   # (64,) — global graph state

        # ── Sequence model: compute context c_t ──────────────────────────────
        if self._is_lstm and self._lstm_hidden is not None:
            context, _new_h = self.sequence_model.step(
                graph_emb,
                self._last_discount,
                self._last_accepted,
                self._last_revenue,
                self._lstm_hidden,
            )
            # Detach hidden state to prevent unbounded BPTT chain across steps.
            # MPS (Apple Metal) fails on long RNN backward chains (~100+ steps).
            # Gradient still flows through the current step's output (context),
            # which is sufficient to train all LSTM parameters ("online LSTM").
            self._lstm_hidden = (_new_h[0].detach(), _new_h[1].detach())
            # Store token for transformer fallback / logging
            token = self.sequence_model.token_proj(
                torch.cat([
                    graph_emb,
                    torch.tensor(
                        [self._last_discount, float(self._last_accepted), self._last_revenue],
                        device=x.device
                    )
                ])
            )
            self._token_history.append(token.detach())

        elif self._is_transformer:
            # Build current token and add to history
            token = self.sequence_model.build_token(
                graph_emb,
                self._last_discount,
                self._last_accepted,
                self._last_revenue,
            )
            self._token_history.append(token)

            if len(self._token_history) == 0:
                # First step: zero context
                context = torch.zeros(self.context_dim, device=x.device)
            else:
                token_seq = torch.stack(self._token_history, dim=0)  # (t, 67)
                context = self.sequence_model(token_seq)              # (64,)
        else:
            # Fallback: zero context (should not happen after reset_episode)
            context = torch.zeros(self.context_dim, device=x.device)

        # ── Scoring head: [H_v ‖ c_t] for all nodes ──────────────────────────
        c_expanded = context.unsqueeze(0).expand(h.shape[0], -1)  # (n, 64)
        combined = torch.cat([h, c_expanded], dim=1)              # (n, 128)
        scores = self.scoring_head(combined).squeeze(-1)           # (n,)

        masked_scores = scores.clone()
        masked_scores[~available_mask] = float('-inf')

        return masked_scores, h, context, graph_emb

    def select_and_price(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        available_mask: torch.Tensor,
        greedy: bool = True,
    ) -> Tuple[int, float, torch.Tensor]:
        """Select next buyer and compute discount.

        Args:
            x: Node features (n, 20).
            edge_index: Edge index (2, |E|).
            available_mask: Boolean mask (n,).
            greedy: Greedy selection if True; stochastic if False.

        Returns:
            node_idx: Selected buyer index.
            discount: Discount in [0,1].
            log_prob: Log probability of the joint action.
        """
        masked_scores, h, context, _ = self.forward(x, edge_index, available_mask)

        # ── Node selection ────────────────────────────────────────────────────
        if greedy:
            node_idx = int(masked_scores.argmax().item())
            log_prob_node = F.log_softmax(masked_scores, dim=0)[node_idx]
        else:
            probs = F.softmax(masked_scores, dim=0)
            dist = torch.distributions.Categorical(probs)
            node_idx = int(dist.sample().item())
            log_prob_node = dist.log_prob(torch.tensor(node_idx, device=probs.device))

        # ── Discount pricing via Beta distribution: [H_{v*} ‖ c_t] ──────────
        combined_selected = torch.cat([h[node_idx], context], dim=0)  # (128,)
        dist = self.get_discount_distribution(combined_selected)

        # Store entropy for trainer entropy regularisation
        self._last_entropy = dist.entropy()

        if greedy:
            discount_t = dist.mean         # E[Beta] = α/(α+β)
            log_prob = log_prob_node
        else:
            discount_t = dist.rsample().clamp(1e-6, 1.0 - 1e-6)
            log_prob_discount = dist.log_prob(discount_t)
            log_prob = log_prob_node + log_prob_discount

        discount = float(discount_t.item())
        return node_idx, discount, log_prob
