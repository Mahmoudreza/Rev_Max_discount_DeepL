"""
src/training/tc_reinforce_trainer.py — REINFORCE for Time-Critical objective.

Sequential model only (Babaei et al.). No IC cascade. No TimeCriticalRevenueEnv.

Differs from reinforce_trainer.py in ONE way:
  • Reward = multi-checkpoint weighted sum: R = Σ_i w_i × Revenue(τ_i)
    where τ_i are acceptance-count deadlines (e.g., [100, 300, 1000]).
  • Revenue(τ) = cumulative revenue after τ ACCEPTANCES in the episode.
    (Same as revenue_at_k(cum_rev_by_S, τ) — no separate cascade phase.)

Uses the SAME policy architecture and env as Idea 1 (SequentialJointPolicy + RevenueEnv).
Does NOT import TimeCriticalRevenueEnv or any cascade code.

Warm-start from Idea 1 checkpoint recommended (skip_imitation=True flag).
"""

import torch
import torch.nn as nn
from typing import List, Optional

import numpy as np

from src.evaluation.tc_evaluation import compute_tc_reward
from src.utils.helpers import graph_to_pyg_data, get_available_mask
from src.utils.features import (
    compute_static_features,
    build_graph_feature_cache,
    compute_node_features_fast,
)


class TCREINFORCETrainer:
    """REINFORCE trainer with multi-checkpoint time-critical reward.

    Each episode:
      1. Run full sequential episode (all n offers, same as Idea 1).
      2. Build cum_rev_by_S: cumulative revenue indexed by |S| (acceptances).
      3. TC reward: R = Σ_i w_i × cum_rev_by_S[τ_i - 1]   (acceptance-indexed)
      4. REINFORCE update: loss = -mean(log_probs) × (R - baseline).

    Args:
        policy:  JointPolicy or SequentialJointPolicy (same as Idea 1).
        cfg:     OmegaConf config with time_critical and training sections.
        logger:  ExperimentLogger.
        device:  PyTorch device.
    """

    def __init__(self, policy: nn.Module, cfg, logger, device: torch.device) -> None:
        self.policy = policy
        self.cfg    = cfg
        self.logger = logger
        self.device = device

        self.optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=float(cfg.training.reinforce_lr),
            weight_decay=float(getattr(cfg.training, "weight_decay", 0.0)),
        )

        tc = cfg.time_critical
        self._train_checkpoints: List[int]   = list(tc.training_checkpoints)
        self._train_weights: List[float]     = list(tc.training_weights)
        self._entropy_coef: float            = float(
            getattr(cfg.training, "entropy_coef", 0.01))
        self._grad_clip: float               = float(cfg.training.grad_clip)

        self._baseline   = 0.0
        self._static_cache: dict = {}
        self._feat_cache:   dict = {}

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_caches(self, graph):
        gid = id(graph)
        if gid not in self._static_cache:
            self._static_cache[gid] = compute_static_features(graph)
        if gid not in self._feat_cache:
            self._feat_cache[gid]   = build_graph_feature_cache(
                graph, self._static_cache[gid])
        return self._feat_cache[gid]

    def _make_env(self, graph, trial: int = 0):
        """Create + reset RevenueEnv (same as Idea 1 — NOT TimeCriticalRevenueEnv)."""
        from omegaconf import OmegaConf
        from src.evaluation.baselines import _make_env
        cfg_t = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))
        base  = int(getattr(self.cfg.project, "seed", 42))
        OmegaConf.update(cfg_t, "project.seed", base + trial)
        env = _make_env(graph, cfg_t)
        env.reset()
        return env

    # ── Rollout collection ────────────────────────────────────────────────────

    def collect_rollout(self, graph) -> dict:
        """Run one full sequential episode under torch.no_grad(), store trajectory.

        Mirrors reinforce_trainer.collect_rollout exactly, but additionally
        tracks cum_rev_by_S and computes tc_reward.  log_probs are NOT stored
        here — they are recomputed with gradients in update().

        Args:
            graph: NetworkX training graph.

        Returns:
            Dict: states, actions, cum_rev_by_S, total_revenue, tc_reward.
        """
        n      = graph.number_of_nodes()
        nodes  = list(graph.nodes())
        fcache = self._get_caches(graph)
        env    = self._make_env(graph, trial=0)

        if hasattr(self.policy, "reset_episode"):
            self.policy.reset_episode(self.device)

        states:       list          = []
        actions:      list          = []
        cum_rev_by_S: List[float]   = []
        total_revenue = 0.0

        for _step in range(n):
            available = env.available_nodes
            if not available:
                break

            features   = compute_node_features_fast(
                cache=fcache, S=frozenset(env.S),
                offered=frozenset(env.offered), t=env.t, k=n, env=env,
            )
            data       = graph_to_pyg_data(graph, features, self.device)
            avail_mask = get_available_mask(n, frozenset(env.offered), nodes, self.device)

            # Action selection — no gradients needed here (replay in update)
            with torch.no_grad():
                node_idx, discount, _ = self.policy.select_and_price(
                    data.x, data.edge_index, avail_mask, greedy=False
                )

            if node_idx not in available:
                node_idx = available[0]

            # Store state for replay
            states.append((features.copy(), list(graph.edges()), list(env.offered)))
            actions.append((node_idx, float(discount)))

            _, rew, done, _ = env.step(node_idx, float(discount))
            accepted = bool(rew > 0)

            if hasattr(self.policy, "update_sequence_state"):
                self.policy.update_sequence_state(float(discount), accepted, float(rew))

            if accepted:
                total_revenue += float(rew)
                cum_rev_by_S.append(total_revenue)

            if done:
                break

        tc_reward = compute_tc_reward(
            cum_rev_by_S,
            self._train_checkpoints,
            self._train_weights,
        )

        return {
            "states":        states,
            "actions":       actions,
            "cum_rev_by_S":  cum_rev_by_S,
            "total_revenue": total_revenue,
            "tc_reward":     tc_reward,
            # keep 'reward' key for backward compat with train() logging
            "reward":        tc_reward,
        }

    # ── REINFORCE update ──────────────────────────────────────────────────────

    def update(self, graph, rollout: dict) -> float:
        """Replay the stored episode with gradients; use tc_reward as advantage.

        Mirrors reinforce_trainer.update exactly but with:
          advantage = normalize(tc_reward / n)   instead of total_revenue / n

        Args:
            graph:   NetworkX training graph (same as in collect_rollout).
            rollout: Output of collect_rollout().

        Returns:
            Policy loss scalar.
        """
        if not rollout["states"]:
            return 0.0

        n     = graph.number_of_nodes()
        nodes = list(graph.nodes())
        fcache = self._get_caches(graph)

        tc_rev   = rollout["tc_reward"] / max(n, 1)
        advantage = self._normalize_reward(tc_rev)

        policy_losses: List[torch.Tensor] = []
        env = self._make_env(graph, trial=0)

        for t, (step_data, (node_idx, discount)) in enumerate(
            zip(rollout["states"], rollout["actions"])
        ):
            _features, _edges, offered_list = step_data
            offered    = frozenset(offered_list)
            avail_mask = get_available_mask(n, offered, nodes, self.device)

            features_t = compute_node_features_fast(
                cache=fcache, S=frozenset(env.S), offered=offered,
                t=t, k=n, env=env,
            )
            data = graph_to_pyg_data(graph, features_t, self.device)

            # Recompute log_prob WITH gradients (select_and_price samples new
            # action; gradient flows through log_prob even if sample differs)
            _, _, log_prob = self.policy.select_and_price(
                data.x, data.edge_index, avail_mask, greedy=False
            )
            entropy = getattr(self.policy, "_last_entropy",
                              torch.tensor(0.0, device=self.device))
            policy_losses.append(
                -log_prob * advantage - self._entropy_coef * entropy
            )

            # Advance env to maintain correct sequence state for LSTM
            if node_idx in env.available_nodes:
                env.step(node_idx, discount)

        if policy_losses:
            loss = torch.stack(policy_losses).mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_clip)
            self.optimizer.step()
            self._baseline = 0.99 * self._baseline + 0.01 * tc_rev
            return float(loss.item())

        return 0.0

    def _normalize_reward(self, r: float) -> float:
        """EMA-normalised advantage (same formula as reinforce_trainer)."""
        return r - self._baseline

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(
        self,
        train_graphs: list,
        n_epochs: Optional[int] = None,
    ) -> dict:
        """TC-REINFORCE training loop.

        Warm-start from Idea 1 checkpoint before calling this (recommended):
            policy.load_state_dict(torch.load("results/checkpoints/rev_gnn_lstm.pt"))

        Args:
            train_graphs: List of NetworkX training graphs.
            n_epochs:     Epochs (default: cfg.training.reinforce_epochs_phase2).

        Returns:
            Dict: losses, rewards, best_reward.
        """
        if n_epochs is None:
            n_epochs = int(self.cfg.training.reinforce_epochs_phase2)
        log_every = int(getattr(self.cfg.logging, "log_every_n_steps", 10))

        losses, rewards = [], []
        best_reward     = -float("inf")
        self.policy.train()

        for epoch in range(n_epochs):
            graph   = train_graphs[epoch % len(train_graphs)]
            rollout = self.collect_rollout(graph)
            loss    = self.update(graph, rollout)
            r       = rollout["reward"]
            losses.append(loss)
            rewards.append(r)

            if r > best_reward:
                best_reward = r

            if epoch % log_every == 0 or epoch == n_epochs - 1:
                self.logger.log({
                    "tc_reinforce/epoch":         epoch,
                    "tc_reinforce/loss":          loss,
                    "tc_reinforce/tc_reward":     r,
                    "tc_reinforce/total_revenue": rollout["total_revenue"],
                    "tc_reinforce/best":          best_reward,
                })
                self.logger.info(
                    f"TC-REINFORCE ep {epoch:4d}/{n_epochs} | "
                    f"tc_reward={r:.4f} | total_rev={rollout['total_revenue']:.2f} | "
                    f"best={best_reward:.4f}"
                )

        self.logger.info(
            f"TC-REINFORCE: {n_epochs} epochs done | best_reward={best_reward:.4f}"
        )
        return {"losses": losses, "rewards": rewards, "best_reward": best_reward}
