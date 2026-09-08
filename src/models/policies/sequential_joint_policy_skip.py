"""
sequential_joint_policy_skip.py — SequentialJointPolicy augmented with an explicit skip action.

forward() returns scores of shape (n+1,):
  scores[:n]  = per-buyer logits (unavailable buyers masked to -1e9)
  scores[n]   = skip logit (always available)

Greedy eval: action = argmax(scores). action==n → skip (waiting semantics:
all buyers remain available, step counter advances, LSTM sees null event).

SKIP_CAP = 50: if 50 consecutive skips → caller must terminate episode.
"""
import torch
import torch.nn as nn
from src.models.policies.sequential_joint_policy import SequentialJointPolicy

SKIP_CAP = 50  # max consecutive skips before episode termination


class SequentialJointPolicySkip(SequentialJointPolicy):
    """SequentialJointPolicy + learned skip action (no-offer / wait).

    The skip logit is produced by a small MLP from the episode context:
        skip_score = skip_head(ctx)   # scalar
    so the skip decision can be state-dependent (budget exhaustion, late
    episode, etc.) while adding only context_dim+1 parameters.
    """

    def __init__(self, encoder, lstm, gnn_dim: int = 64, context_dim: int = 64):
        super().__init__(encoder, lstm, gnn_dim=gnn_dim, context_dim=context_dim)
        # Skip head: context → scalar logit
        self.skip_head = nn.Sequential(
            nn.Linear(context_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        # Initialise skip head so skip starts neutral (log-odds ≈ 0)
        for m in self.skip_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, ei, av_mask):
        """
        Args:
            x:       (n, in_dim) node features
            ei:      PyG edge_index
            av_mask: (n,) bool — True = buyer is available

        Returns:
            scores:  (n+1,) raw logits; index n is the skip action
            h:       (n, gnn_dim) per-node GNN embeddings
            ctx:     (context_dim,) LSTM episode context
            enc_out: raw encoder output (pass-through)
        """
        sc, h, ctx, enc = super().forward(x, ei, av_mask)   # (n,)
        skip_sc = self.skip_head(ctx).squeeze(-1)            # scalar
        return torch.cat([sc, skip_sc.unsqueeze(0)]), h, ctx, enc

    # update_sequence_state, get_discount_distribution, reset_episode
    # are all inherited unchanged.
