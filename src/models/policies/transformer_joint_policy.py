"""src/models/policies/transformer_joint_policy.py — Rev-GNN-Transformer policy.

Mirror of SequentialJointPolicy with EpisodeLSTM replaced by
EpisodeTransformerSliding.  ALL other components are identical:
  • GraphSAGE encoder (d=64)
  • Scoring head: Linear(128, 1) → softmax over available nodes
  • Beta-distribution pricing head: Linear(128, 2) → Beta(α, β)
  • 20-dim node features (same as SequentialJointPolicy)
  • Same forward / reset_episode / select_and_price / update_sequence_state API

The ONLY change: `self.seq_module` is an EpisodeTransformerSliding
(ALiBi, sliding window=256, Pre-LN, 2L 4H d=64 ffn=128).

Drop-in replacement in all experiment scripts:
  Replace: policy = SequentialJointPolicy(enc, lstm, ...)
  With:    policy = TransformerJointPolicy(enc, tfm, ...)
  Everything else (Phase 1 imitation, Phase 2 REINFORCE, eval) is unchanged.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


class TransformerJointPolicy(nn.Module):
    """Joint node-selection + discount-pricing policy with Transformer context.

    Args:
        encoder:     GraphSAGE encoder → (n_nodes, gnn_dim) node embeddings.
        seq_module:  EpisodeTransformerSliding instance.
        gnn_dim:     GNN output dimension (default 64).
        context_dim: Transformer context dimension (= seq_module.context_dim).
    """

    def __init__(
        self,
        encoder:     nn.Module,
        seq_module:  nn.Module,
        gnn_dim:     int = 64,
        context_dim: int = 64,
    ) -> None:
        super().__init__()
        self.encoder     = encoder
        self.seq_module  = seq_module
        self.gnn_dim     = gnn_dim
        self.context_dim = context_dim
        combined_dim     = gnn_dim + context_dim   # 128

        # ── Scoring head ────────────────────────────────────────────────────────
        # Scores how attractive each node is to offer next.
        self.scorer = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.ReLU(),
            nn.Linear(combined_dim, 1),
        )

        # ── Pricing (discount) head ─────────────────────────────────────────────
        # Outputs two positive scalars → Beta(α, β) distribution over [0, 1].
        # α > 1 and β > 1 → unimodal, away from 0 and 1 extremes.
        self.pricing_head = nn.Linear(combined_dim, 2)

        # ── Episode state (reset every episode) ────────────────────────────────
        self._last_discount: float = 0.0
        self._last_accepted: bool  = False
        self._last_revenue:  float = 0.0
        self._hidden = None   # unused for Transformer; present for API parity

    # ── Episode management ─────────────────────────────────────────────────────

    def reset_episode(self, device: torch.device) -> None:
        """Reset Transformer token buffer and last-action state. Call before each episode."""
        self.seq_module.reset_episode(device)
        self._last_discount = 0.0
        self._last_accepted = False
        self._last_revenue  = 0.0
        self._hidden        = None

    def update_sequence_state(
        self,
        discount: float,
        accepted: bool,
        revenue:  float,
    ) -> None:
        """Record action outcome for the next step's token.

        Called AFTER env.step() with the discount used, whether it was accepted,
        and the revenue earned.  The Transformer's buffer was already updated
        inside step(); this method only updates the 'last action' scalars used
        to build the NEXT step's token.
        """
        self._last_discount = discount
        self._last_accepted = accepted
        self._last_revenue  = revenue
        # seq_module.update_sequence_state is a no-op for EpisodeTransformerSliding
        self.seq_module.update_sequence_state(discount, accepted, revenue)

    # ── Core forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        x:          torch.Tensor,   # (n, feature_dim)
        edge_index: torch.Tensor,   # (2, E)
        mask:       torch.Tensor,   # (n,) bool — True = available
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """Run one step through encoder → transformer → scorer.

        Args:
            x:          Node features at current step, shape (n, feature_dim).
            edge_index: Graph connectivity, shape (2, E).
            mask:       Boolean mask, True for nodes that can still be offered.

        Returns:
            scores:  Raw logits (n,), softmax over available nodes gives policy.
            h:       Node embeddings (n, gnn_dim), for use by pricing head.
            context: Episode context (context_dim,) from Transformer.
            None:    Placeholder (SequentialJointPolicy returns log_p here in some calls).
        """
        # ── GNN ────────────────────────────────────────────────────────────────
        h = self.encoder(x, edge_index)   # (n, gnn_dim)

        # Graph-level embedding: mean-pool for Transformer token
        graph_emb = h.mean(dim=0)         # (gnn_dim,)

        # ── Transformer step ───────────────────────────────────────────────────
        context, self._hidden = self.seq_module.step(
            graph_emb,
            self._last_discount,
            self._last_accepted,
            self._last_revenue,
            hidden=None,
        )   # context: (context_dim,)

        # ── Scoring ────────────────────────────────────────────────────────────
        # Broadcast context to all nodes: [h_v ‖ context]
        context_exp = context.unsqueeze(0).expand(h.shape[0], -1)  # (n, context_dim)
        combined    = torch.cat([h, context_exp], dim=-1)           # (n, 128)
        scores      = self.scorer(combined).squeeze(-1)             # (n,)

        # Mask unavailable nodes (set score → -inf before softmax)
        scores = scores.masked_fill(~mask, float("-inf"))

        return scores, h, context, None

    # ── Action sampling ─────────────────────────────────────────────────────────

    def get_discount_distribution(
        self,
        combined: torch.Tensor,   # (combined_dim,) = (128,)
    ) -> Beta:
        """Build Beta(α, β) discount distribution from node+context embedding.

        Uses softplus + 1.0 offset so α, β > 1 (unimodal, interior mode).

        Args:
            combined: Concatenated [h_v ‖ context] for the selected node, shape (128,).

        Returns:
            Beta distribution. Sample with .sample() or use .mean for greedy pricing.
        """
        raw = self.pricing_head(combined)                       # (2,)
        alpha = F.softplus(raw[0]) + 1.0                        # α > 1
        beta  = F.softplus(raw[1]) + 1.0                        # β > 1
        return Beta(alpha, beta)

    def select_and_price(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        mask:       torch.Tensor,
        greedy:     bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select a node and sample a discount, returning action + log probability.

        This is the main method called during rollout (both training and eval).

        Args:
            x:          Node features (n, feature_dim).
            edge_index: Graph edges (2, E).
            mask:       Bool mask, True = available node.
            greedy:     If True, take argmax/mean instead of sampling.

        Returns:
            node_idx: Selected node index (scalar tensor).
            discount: Discount in [0, 1] (scalar tensor).
            log_p:    log π(node, discount | state) (scalar tensor, for REINFORCE).
        """
        scores, h, context, _ = self.forward(x, edge_index, mask)

        # ── Node selection ─────────────────────────────────────────────────────
        node_probs = F.softmax(scores, dim=0)   # (n,)
        if greedy:
            node_idx = scores.argmax()
            log_p_node = torch.log(node_probs[node_idx] + 1e-8)
        else:
            node_dist  = torch.distributions.Categorical(probs=node_probs)
            node_idx   = node_dist.sample()
            log_p_node = node_dist.log_prob(node_idx)

        # ── Discount pricing ───────────────────────────────────────────────────
        combined = torch.cat([h[node_idx], context], dim=-1)   # (128,)
        dist     = self.get_discount_distribution(combined)
        if greedy:
            discount   = dist.mean
            log_p_disc = dist.log_prob(discount.clamp(1e-6, 1 - 1e-6))
        else:
            discount   = dist.rsample()
            discount   = discount.clamp(0.0, 1.0)
            log_p_disc = dist.log_prob(discount.clamp(1e-6, 1 - 1e-6))

        log_p = log_p_node + log_p_disc   # joint log prob

        return node_idx, discount, log_p
