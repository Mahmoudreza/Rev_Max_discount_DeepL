"""
src/models/policies/sac_policy.py

SAC actor-critic wrapper for the mixed discrete+continuous action space.

Node selection: discrete (treated via straight-through / log-sum-exp)
Discount:       continuous Gaussian, reparameterized
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class SACPolicy(nn.Module):
    """SAC actor: outputs node logits + discount distribution parameters.

    Shares the GNN encoder with the Q-networks (which are separate).

    Args:
        joint_policy: JointPolicy instance (shares scoring + pricing heads).
        hidden_dim: GNN hidden dimension (default 64).
        discount_log_std_range: (min, max) clamp for log std of discount dist.
    """

    def __init__(
        self,
        joint_policy: nn.Module,
        hidden_dim: int = 64,
        discount_log_std_range: Tuple[float, float] = (-4.0, 2.0),
    ) -> None:
        super().__init__()
        self.joint_policy = joint_policy
        self.log_std_min, self.log_std_max = discount_log_std_range

        # Learnable log-std for the discount distribution (per-node)
        self.discount_log_std_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        available_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute actor outputs for SAC update.

        Args:
            x: Node features (n, d).
            edge_index: Edge index (2, |E|).
            available_mask: Boolean mask (n,).

        Returns:
            node_logits: Raw node selection scores, shape (n,).
            discount_mean: Discount mean (Sigmoid output), shape (n,).
            discount_log_std: Clamped log-std, shape (n,).
        """
        scores, masked_scores, h = self.joint_policy.forward(
            x, edge_index, available_mask, return_embeddings=True
        )

        discount_mean = self.joint_policy.pricing_head(h).squeeze(-1)  # (n,) in [0,1]

        log_std = self.discount_log_std_head(h).squeeze(-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return masked_scores, discount_mean, log_std

    def sample_action(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        available_mask: torch.Tensor,
    ) -> Tuple[int, float, torch.Tensor]:
        """Sample node + discount using reparameterization trick.

        Args:
            x: Node features (n, d).
            edge_index: Edge index (2, |E|).
            available_mask: Boolean mask (n,).

        Returns:
            node_idx: Sampled node index.
            discount: Sampled discount in [0, 1].
            log_prob: Log-prob for SAC entropy term.
        """
        masked_scores, discount_mean, log_std = self.forward(
            x, edge_index, available_mask
        )

        # Node selection: Gumbel-softmax for discrete
        probs = F.softmax(masked_scores, dim=0)
        dist_node = torch.distributions.Categorical(probs)
        node_idx = int(dist_node.sample().item())
        log_prob_node = dist_node.log_prob(torch.tensor(node_idx, device=x.device))

        # Discount: reparameterized Gaussian → squashed to [0,1]
        std = log_std[node_idx].exp()
        eps = torch.randn_like(std)
        # Pre-squash sample
        raw = discount_mean[node_idx] + std * eps
        # Squash to [0,1] via sigmoid (already applied in pricing_head)
        # discount_mean is already in [0,1], so we perturb then re-clamp
        discount_tensor = torch.clamp(raw, 0.0, 1.0)
        discount = float(discount_tensor.item())

        # Log-prob with Gaussian approximation
        discount_dist = torch.distributions.Normal(discount_mean[node_idx], std)
        log_prob_discount = discount_dist.log_prob(
            torch.tensor(discount, device=x.device)
        )

        log_prob = log_prob_node + log_prob_discount
        return node_idx, discount, log_prob

    def select_and_price(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        available_mask: torch.Tensor,
        greedy: bool = False,
    ) -> Tuple[int, float, torch.Tensor]:
        """Unified interface: delegate to sample_action or greedy.

        Args:
            x: Node features (n, d).
            edge_index: Edge index (2, |E|).
            available_mask: Boolean mask (n,).
            greedy: If True, use argmax for node + mean for discount.

        Returns:
            node_idx, discount, log_prob
        """
        if greedy:
            return self.joint_policy.select_and_price(
                x, edge_index, available_mask, greedy=True
            )
        return self.sample_action(x, edge_index, available_mask)


class SACQNetwork(nn.Module):
    """Twin Q-network for SAC, sharing GNN encoder with the actor.

    Takes (graph state, node_idx, discount) → scalar Q-value.

    Args:
        encoder: Shared GNN encoder (GraphSAGEEncoder).
        hidden_dim: GNN output dimension (default 64).
        action_dim: Action encoding size (node embedding + discount = 65).
    """

    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = encoder

        # Q1 network: [global_state (64) + node_emb (64) + discount (1)] → Q
        action_input_dim = hidden_dim + hidden_dim + 1

        self.q1 = nn.Sequential(
            nn.Linear(action_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(action_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_idx: int,
        discount: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Q1 and Q2 values.

        Args:
            x: Node features (n, d).
            edge_index: Edge index (2, |E|).
            node_idx: Selected node index.
            discount: Discount value.

        Returns:
            (q1_val, q2_val) scalar tensors.
        """
        h = self.encoder(x, edge_index)          # (n, 64)
        global_state = h.mean(dim=0)             # (64,)
        node_emb = h[node_idx]                   # (64,)
        d = torch.tensor([discount], device=x.device)

        action_state = torch.cat([global_state, node_emb, d], dim=0)  # (129,)
        action_state = action_state.unsqueeze(0)  # (1, 129)

        q1_val = self.q1(action_state).squeeze()
        q2_val = self.q2(action_state).squeeze()
        return q1_val, q2_val
