"""
src/models/policies/joint_policy.py

Joint Policy: combines GNN encoder + node scoring head + pricing head.

At each step:
  1. GNN encoder → node embeddings h_v for all v
  2. Scoring head → score_v for all v ∉ offered
  3. v* = argmax(score_v)  [masked to available nodes only]
  4. Pricing head applied to h_{v*} → discount ∈ [0, 1]

This is the core architectural contribution over WSDM 2027:
the WSDM policy outputs only node selection (step 3);
we add step 4 (the pricing head) to make it a joint policy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class JointPolicy(nn.Module):
    """Joint seed selection + discount policy for revenue maximization.

    Wraps a GNN encoder with a scoring head (node selection, as in WSDM)
    and a pricing head (discount in [0,1], new for this paper).

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

        # ── Pricing head (NEW for this paper) ────────────────────────────────
        # 64 → 32 → 1, with Sigmoid to output discount ∈ [0,1]
        self.pricing_head = nn.Sequential(
            nn.Linear(hidden_dim, pricing_hidden),
            nn.ReLU(),
            nn.Linear(pricing_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        available_mask: torch.Tensor,
        return_embeddings: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass: compute node scores and discount for selected node.

        Args:
            x: Node feature matrix, shape (n, d).
            edge_index: Edge index tensor, shape (2, |E|).
            available_mask: Boolean mask, shape (n,), True for nodes not yet offered.
            return_embeddings: If True, also return GNN embeddings.

        Returns:
            scores: Raw scores for all nodes, shape (n,). Use for selection.
            masked_scores: Scores with unavailable nodes set to -inf, shape (n,).
            embeddings: Node embeddings (n, 64) if return_embeddings else None.
        """
        # ── GNN encoder ──────────────────────────────────────────────────────
        h = self.encoder(x, edge_index)   # (n, 64)

        # ── Scoring head ─────────────────────────────────────────────────────
        scores = self.scoring_head(h).squeeze(-1)   # (n,)

        # Mask unavailable nodes
        masked_scores = scores.clone()
        masked_scores[~available_mask] = float('-inf')

        embeddings = h if return_embeddings else None
        return scores, masked_scores, embeddings

    def select_and_price(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        available_mask: torch.Tensor,
        greedy: bool = True,
    ) -> Tuple[int, float, torch.Tensor]:
        """Select a node and compute its discount.

        This is the inference-time method called by the RL agent at each step.

        Args:
            x: Node features (n, d).
            edge_index: Edge index (2, |E|).
            available_mask: Boolean mask for available nodes (n,).
            greedy: If True, take argmax; if False, sample from softmax.

        Returns:
            node_idx: Selected node index (int).
            discount: Discount value in [0,1] (float).
            log_prob: Log probability of the action (for REINFORCE gradient).
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
            dist = torch.distributions.Categorical(probs)
            node_idx = int(dist.sample().item())
            log_prob_node = dist.log_prob(torch.tensor(node_idx, device=probs.device))

        # ── Discount pricing ──────────────────────────────────────────────────
        # Pricing head applied to the selected node's embedding
        discount_tensor = self.pricing_head(h[node_idx].unsqueeze(0)).squeeze()
        discount = float(discount_tensor.item())

        # Log prob for continuous discount (approximate as Gaussian for REINFORCE)
        # For SAC/PPO, this is handled in their respective wrappers
        log_prob = log_prob_node

        return node_idx, discount, log_prob

    def get_discount_for_node(self, h: torch.Tensor, node_idx: int) -> torch.Tensor:
        """Get discount for a specific node given its embedding.

        Args:
            h: All node embeddings (n, 64).
            node_idx: Target node index.

        Returns:
            Discount value as a tensor (scalar).
        """
        return self.pricing_head(h[node_idx].unsqueeze(0)).squeeze()
