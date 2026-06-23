"""
src/models/policies/ppo_policy.py

PPO actor-critic policy wrapper.

Wraps a JointPolicy (actor) with an additional value head (critic).
The value head shares the GNN encoder with the actor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class PPOPolicy(nn.Module):
    """PPO actor-critic for joint seed selection + discount assignment.

    Args:
        joint_policy: JointPolicy instance (actor — scoring head + pricing head).
        hidden_dim: GNN hidden dimension (default 64).
        value_hidden: Value MLP hidden dimension (default 32).
    """

    def __init__(
        self,
        joint_policy: nn.Module,
        hidden_dim: int = 64,
        value_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.joint_policy = joint_policy

        # Value head: mean-pool node embeddings → scalar V(s)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        available_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute policy outputs + state value for PPO update.

        Args:
            x: Node features (n, d).
            edge_index: Edge index (2, |E|).
            available_mask: Boolean mask (n,).

        Returns:
            masked_scores: Node scores with unavailable set to -inf (n,).
            discount_params: Discount output (n, 1) — one per node.
            value: Scalar state value V(s).
        """
        scores, masked_scores, h = self.joint_policy.forward(
            x, edge_index, available_mask, return_embeddings=True
        )

        # Value: mean pool of node embeddings
        value = self.value_head(h.mean(dim=0, keepdim=True)).squeeze()

        # Discount params for all nodes
        discount_params = self.joint_policy.pricing_head(h)  # (n, 1)

        return masked_scores, discount_params, value

    def get_value(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Compute state value only (for advantage computation).

        Args:
            x: Node features (n, d).
            edge_index: Edge index (2, |E|).

        Returns:
            Scalar state value tensor.
        """
        h = self.joint_policy.encoder(x, edge_index)
        return self.value_head(h.mean(dim=0, keepdim=True)).squeeze()

    def select_and_price(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        available_mask: torch.Tensor,
        greedy: bool = False,
    ) -> Tuple[int, float, torch.Tensor]:
        """Delegate action selection to wrapped JointPolicy.

        Args:
            x: Node features (n, d).
            edge_index: Edge index (2, |E|).
            available_mask: Boolean mask (n,).
            greedy: Greedy selection if True; stochastic if False.

        Returns:
            node_idx, discount, log_prob
        """
        return self.joint_policy.select_and_price(
            x, edge_index, available_mask, greedy=greedy
        )

    def evaluate_actions(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        available_mask: torch.Tensor,
        node_idx: int,
        discount: float,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate log-prob, entropy, and value for a stored action.

        Used during PPO update step to compute ratio pi_new / pi_old.

        Args:
            x: Node features (n, d).
            edge_index: Edge index (2, |E|).
            available_mask: Boolean mask (n,).
            node_idx: Stored node action.
            discount: Stored discount action.

        Returns:
            Dict with: log_prob, entropy, value.
        """
        masked_scores, discount_params, value = self.forward(
            x, edge_index, available_mask
        )

        # Node selection log-prob
        log_prob_node = F.log_softmax(masked_scores, dim=0)[node_idx]
        entropy_node = -(F.softmax(masked_scores, dim=0) *
                         F.log_softmax(masked_scores, dim=0)).sum()

        # Discount log-prob (Gaussian with std=0.1)
        discount_mean = discount_params[node_idx].squeeze()
        discount_std = torch.tensor(0.1, device=x.device)
        discount_dist = torch.distributions.Normal(discount_mean, discount_std)
        log_prob_discount = discount_dist.log_prob(
            torch.tensor(discount, device=x.device)
        )
        entropy_discount = discount_dist.entropy()

        log_prob = log_prob_node + log_prob_discount
        entropy = entropy_node + entropy_discount

        return {
            "log_prob": log_prob,
            "entropy": entropy,
            "value": value,
        }
