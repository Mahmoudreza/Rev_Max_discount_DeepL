"""
src/training/reinforce_trainer.py

Phase-2 REINFORCE fine-tuning for the joint policy.

Loss: -log_pi(a_t|s_t) * (G_t - baseline), where G_t is the return.
Baseline: running mean of returns (no learned value function).
"""

import torch
import torch.nn as nn
from typing import List
import numpy as np

from src.utils.helpers import graph_to_pyg_data, get_available_mask
from src.utils.features import (compute_static_features, compute_node_features,
                                  build_graph_feature_cache, compute_node_features_fast)
from src.evaluation.baselines import _make_env


class REINFORCETrainer:
    """REINFORCE fine-tuning trainer.

    Args:
        policy: JointPolicy.
        cfg: OmegaConf DictConfig.
        logger: ExperimentLogger.
        device: PyTorch device.
    """

    def __init__(self, policy: nn.Module, cfg, logger, device: torch.device) -> None:
        self.policy = policy
        self.cfg = cfg
        self.logger = logger
        self.device = device

        self.optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=cfg.training.reinforce_lr,
            weight_decay=cfg.training.weight_decay,
        )
        self._baseline = 0.0          # legacy (kept for backward compat)
        self.reward_baseline = 0.0    # episode-level reward baseline (size-normalised)
        self._static_cache: dict = {} # graph id → static features (avoid O(nm) repeat)
        self._feat_cache: dict = {}   # graph id → build_graph_feature_cache result

        # ── Running reward statistics (Welford's online algorithm) ────────────
        # Used to normalise the advantage so that zero-reward episodes produce
        # a negative signal ("you did worse than average") rather than no signal.
        self._rw_mean:  float = 0.0
        self._rw_m2:    float = 1.0   # variance accumulator (init 1 → std=1 early)
        self._rw_count: int   = 0

    def collect_rollout(self, graph) -> dict:
        """Run one episode and collect (state, action, reward) trajectory.

        Args:
            graph: NetworkX graph for this episode.

        Returns:
            Dict with keys: states, actions, rewards, total_revenue.
        """
        env = _make_env(graph, self.cfg)
        obs = env.reset()
        # Reset LSTM/Transformer hidden state at episode start
        if hasattr(self.policy, "reset_episode"):
            self.policy.reset_episode(self.device)
        gid = id(graph)
        if gid not in self._static_cache:
            self._static_cache[gid] = compute_static_features(graph)
        if gid not in self._feat_cache:
            self._feat_cache[gid] = build_graph_feature_cache(graph, self._static_cache[gid])
        fcache = self._feat_cache[gid]
        n = graph.number_of_nodes()
        nodes = list(graph.nodes())

        states, actions, rewards = [], [], []

        for step in range(n):
            available = env.available_nodes
            if not available:
                break

            features = compute_node_features_fast(
                cache=fcache, S=frozenset(env.S), offered=frozenset(env.offered),
                t=env.t, k=n, env=env,
            )
            data = graph_to_pyg_data(graph, features, self.device)
            available_mask = get_available_mask(n, frozenset(env.offered), nodes, self.device)

            # Store tensors detached for now; recompute gradients at update
            states.append(
                (features.copy(), list(graph.edges()), list(env.offered))
            )

            with torch.no_grad():
                node_idx, discount, log_prob = self.policy.select_and_price(
                    data.x, data.edge_index, available_mask, greedy=False
                )

            if node_idx not in available:
                node_idx = available[0]  # fallback

            _, reward, done, info = env.step(node_idx, discount)
            # Notify LSTM/Transformer sequence model of step outcome
            if hasattr(self.policy, "update_sequence_state"):
                self.policy.update_sequence_state(
                    discount, bool(reward > 0), float(reward)
                )
            actions.append((node_idx, discount))
            rewards.append(reward)

            if done:
                break

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "total_revenue": env.total_revenue,
        }

    def _normalize_reward(self, reward: float) -> float:
        """Online Welford normalisation of per-episode reward.

        Keeps a running mean and variance of (total_revenue / n) values seen
        during training.  Returns z-scored value so that:
          - Episodes below the running mean → negative advantage → policy pushed away
          - Episodes above the running mean → positive advantage → policy reinforced
          - Scale is independent of graph size or reward magnitude
        """
        self._rw_count += 1
        delta = reward - self._rw_mean
        self._rw_mean += delta / self._rw_count
        delta2 = reward - self._rw_mean
        self._rw_m2 += delta * delta2
        std = max((self._rw_m2 / max(self._rw_count, 1)) ** 0.5, 1e-8)
        return (reward - self._rw_mean) / std

    def compute_returns(self, rewards: List[float]) -> List[float]:
        """Compute discounted returns G_t = sum_{k>=t} gamma^(k-t) * r_k.

        Args:
            rewards: Per-step rewards.

        Returns:
            List of returns G_t.
        """
        gamma = self.cfg.reward.gamma if self.cfg.reward.type == "npv" else 1.0
        G = 0.0
        returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.append(G)
        returns.reverse()
        return returns

    def update(self, graph, rollout: dict) -> float:
        """REINFORCE gradient update from one rollout.

        Args:
            graph: NetworkX graph.
            rollout: Output of collect_rollout().

        Returns:
            Policy loss value.
        """
        gid = id(graph)
        if gid not in self._static_cache:
            self._static_cache[gid] = compute_static_features(graph)
        if gid not in self._feat_cache:
            self._feat_cache[gid] = build_graph_feature_cache(graph, self._static_cache[gid])
        fcache = self._feat_cache[gid]
        n = graph.number_of_nodes()
        nodes = list(graph.nodes())

        # Episode-level advantage: normalised by running reward statistics.
        # This ensures zero-revenue episodes produce a NEGATIVE signal, which
        # is essential for breaking out of the "give everything for free" trap.
        total_rev = rollout["total_revenue"] / n
        advantage = self._normalize_reward(total_rev)

        policy_losses = []
        env = _make_env(graph, self.cfg)
        env.reset()

        for t, (step_data, (node_idx, discount)) in enumerate(
            zip(rollout["states"], rollout["actions"])
        ):
            features, edges, offered_list = step_data
            offered = frozenset(offered_list)
            available_mask = get_available_mask(n, offered, nodes, self.device)

            features_tensor = compute_node_features_fast(
                cache=fcache, S=frozenset(env.S), offered=offered,
                t=t, k=n, env=env,
            )
            data = graph_to_pyg_data(graph, features_tensor, self.device)

            _, _, log_prob = self.policy.select_and_price(
                data.x, data.edge_index, available_mask, greedy=False
            )

            # Entropy regularisation: bonus for non-degenerate Beta distributions.
            # High entropy (α ≈ β ≈ 1) means wide discount range is explored.
            # Low entropy (α or β >> 1) means distribution is concentrated.
            entropy = getattr(self.policy, '_last_entropy',
                              torch.tensor(0.0, device=self.device))
            entropy_coef = float(getattr(self.cfg.training, 'entropy_coef', 0.01))
            policy_losses.append(-log_prob * advantage - entropy_coef * entropy)

            # Update env state
            if node_idx in env.available_nodes:
                env.step(node_idx, discount)

        if policy_losses:
            loss = torch.stack(policy_losses).mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.cfg.training.grad_clip
            )
            self.optimizer.step()
            # Update legacy EMA baseline (kept for logging / backward compat)
            self.reward_baseline = 0.99 * self.reward_baseline + 0.01 * total_rev
            return float(loss.item())

        return 0.0

    def train(self, train_graphs: list) -> dict:
        """REINFORCE fine-tuning over n_rollouts per epoch.

        Args:
            train_graphs: List of NetworkX training graphs.

        Returns:
            Dict with keys: losses, revenues.
        """
        n_rollouts = self.cfg.training.reinforce_epochs
        log_every = self.cfg.logging.log_every_n_steps
        losses, revenues = [], []

        self.policy.train()

        for rollout_idx in range(n_rollouts):
            graph = train_graphs[rollout_idx % len(train_graphs)]
            rollout = self.collect_rollout(graph)
            loss = self.update(graph, rollout)
            losses.append(loss)
            revenues.append(rollout["total_revenue"])

            if rollout_idx % log_every == 0:
                self.logger.log({
                    "reinforce/rollout": rollout_idx,
                    "reinforce/loss": loss,
                    "reinforce/revenue": rollout["total_revenue"],
                })

        self.logger.info(
            f"REINFORCE: {n_rollouts} rollouts done, "
            f"final revenue={revenues[-1]:.4f}"
        )
        return {"losses": losses, "revenues": revenues}
