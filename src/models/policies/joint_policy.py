"""
src/models/policies/joint_policy.py

Joint Policy: combines GNN encoder + node scoring head + pricing head.

At each step:
  1. GNN encoder → node embeddings h_v for all v
  2. Scoring head → score_v for all v ∉ offered
  3. v* = argmax(score_v)  [masked to available nodes only]
  4. Pricing head applied to h_{v*} → Beta(α, β) → discount ∈ (0, 1)

Pricing head change (vs Gaussian): Beta distribution is:
  - Naturally bounded in (0, 1) — no clamping needed
  - Unimodal with α, β > 1 — prevents degenerate endpoint mass
  - Gradient flows cleanly via rsample() (reparameterisation)
  - Entropy is well-defined and can be used for regularisation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class JointPolicy(nn.Module):
    """Joint seed selection + discount policy for revenue maximization.

    Wraps a GNN encoder with a scoring head (node selection, as in WSDM)
    and a pricing head (Beta distribution over discount, new for this paper).

    Args:
        encoder: GNN encoder module (GraphSAGE or Graph Transformer).
        hidden_dim: Hidden dimension of the encoder output (default 64).
        scoring_hidden: Hidden dim of scoring MLP (default 32).
        pricing_hidden: Hidden dim of pricing MLP (default 32).
    """

    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int = 64,
        scoring_hidden: int = 32,
        pricing_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.encoder = encoder

        # ── Scoring head (identical to WSDM Eq.14) ───────────────────────────
        # 64 → 32 → 1
        self.scoring_head = nn.Sequential(
            nn.Linear(hidden_dim, scoring_hidden),
            nn.ReLU(),
            nn.Linear(scoring_hidden, 1),
        )

        # ── Pricing head (Beta distribution — FIX for pricing collapse) ───────
        # Outputs (log_alpha_raw, log_beta_raw); softplus + 1 → α > 1, β > 1
        # This forces unimodal Beta: no probability mass at degenerate endpoints.
        self.pricing_head = nn.Sequential(
            nn.Linear(hidden_dim, pricing_hidden),
            nn.ReLU(),
            nn.Linear(pricing_hidden, 2),   # 2 outputs: raw_alpha, raw_beta
        )

        # Last entropy computed during select_and_price (read by trainer)
        self._last_entropy: torch.Tensor = torch.tensor(0.0)

    # ── Beta distribution helper ──────────────────────────────────────────────

    def get_discount_distribution(self, h_node: torch.Tensor) -> torch.distributions.Beta:
        """Get Beta distribution for the discount at a node.

        Args:
            h_node: Node embedding, shape (hidden_dim,) or (1, hidden_dim).

        Returns:
            Beta distribution with α > 1 and β > 1 (guaranteed unimodal).
        """
        if h_node.dim() == 1:
            h_node = h_node.unsqueeze(0)
        raw = self.pricing_head(h_node).squeeze(0)   # (2,)
        # Softplus ensures α, β > 0;  +1 ensures > 1 (unimodal, no endpoint mass)
        alpha = F.softplus(raw[0]) + 1.0
        beta  = F.softplus(raw[1]) + 1.0
        return torch.distributions.Beta(alpha, beta)

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        available_mask: torch.Tensor,
        return_embeddings: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass: compute node scores.

        Args:
            x: Node feature matrix, shape (n, d).
            edge_index: Edge index tensor, shape (2, |E|).
            available_mask: Boolean mask, shape (n,), True for nodes not yet offered.
            return_embeddings: If True, also return GNN embeddings.

        Returns:
            scores: Raw scores for all nodes, shape (n,).
            masked_scores: Scores with unavailable nodes set to -inf, shape (n,).
            embeddings: Node embeddings (n, 64) if return_embeddings else None.
        """
        h = self.encoder(x, edge_index)   # (n, hidden_dim)

        scores = self.scoring_head(h).squeeze(-1)   # (n,)

        masked_scores = scores.clone()
        masked_scores[~available_mask] = float('-inf')

        embeddings = h if return_embeddings else None
        return scores, masked_scores, embeddings

    # ── Inference + training ──────────────────────────────────────────────────

    def select_and_price(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        available_mask: torch.Tensor,
        greedy: bool = True,
    ) -> Tuple[int, float, torch.Tensor]:
        """Select a node and compute its discount.

        Uses Beta(α, β) distribution for the discount.  This replaces the old
        Gaussian(mean, 0.1) sampler which collapsed to discount≈1.0 because
        clamping at 1.0 killed ~50% of gradients.

        Args:
            x: Node features (n, d).
            edge_index: Edge index (2, |E|).
            available_mask: Boolean mask for available nodes (n,).
            greedy: If True, use distribution mean (inference); else rsample.

        Returns:
            node_idx: Selected node index (int).
            discount: Discount value in (0, 1) (float).
            log_prob: Joint log probability of (node, discount) action.
        """
        scores, masked_scores, h = self.forward(
            x, edge_index, available_mask, return_embeddings=True
        )

        # ── Node selection ────────────────────────────────────────────────────
        if greedy:
            node_idx = int(masked_scores.argmax().item())
            log_prob_node = F.log_softmax(masked_scores, dim=0)[node_idx]
        else:
            probs = F.softmax(masked_scores, dim=0)
            dist_node = torch.distributions.Categorical(probs)
            node_idx = int(dist_node.sample().item())
            log_prob_node = dist_node.log_prob(
                torch.tensor(node_idx, device=probs.device)
            )

        # ── Discount pricing via Beta distribution ────────────────────────────
        dist = self.get_discount_distribution(h[node_idx])

        # Store entropy for trainer entropy regularisation
        self._last_entropy = dist.entropy()

        if greedy:
            # Deterministic inference: use distribution mean E[Beta] = α/(α+β)
            discount_t = dist.mean
            log_prob = log_prob_node
        else:
            # Training: reparameterised sample — gradient flows through rsample()
            # No clamping needed: Beta support is naturally (0, 1)
            discount_t = dist.rsample().clamp(1e-6, 1.0 - 1e-6)
            log_prob_discount = dist.log_prob(discount_t)
            log_prob = log_prob_node + log_prob_discount

        discount = float(discount_t.item())
        return node_idx, discount, log_prob

    def get_discount_for_node(self, h: torch.Tensor, node_idx: int) -> torch.Tensor:
        """Get expected discount for a node given all embeddings.

        Args:
            h: All node embeddings (n, hidden_dim).
            node_idx: Target node index.

        Returns:
            Expected discount (Beta mean) as a scalar tensor.
        """
        return self.get_discount_distribution(h[node_idx]).mean
