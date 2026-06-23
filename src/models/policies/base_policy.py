"""
src/models/policies/base_policy.py

Abstract base class for all joint policies.
Defines the minimum interface every policy must implement.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple


class BasePolicy(nn.Module, ABC):
    """Abstract base class for revenue-maximization policies.

    All policies (JointPolicy, SequentialJointPolicy, PPOPolicy, SACPolicy)
    inherit from this class and implement select_and_price().
    """

    @abstractmethod
    def select_and_price(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        available_mask: torch.Tensor,
        greedy: bool = True,
    ) -> Tuple[int, float, torch.Tensor]:
        """Select a node and compute its discount.

        Args:
            x: Node features (n, d).
            edge_index: Edge index (2, |E|).
            available_mask: Boolean mask (n,), True for available nodes.
            greedy: If True, deterministic argmax; if False, sample.

        Returns:
            node_idx: Selected buyer index (int).
            discount: Discount in [0, 1] (float).
            log_prob: Log probability tensor for REINFORCE gradient.
        """
        raise NotImplementedError
