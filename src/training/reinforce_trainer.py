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
from src.utils.features import compute_static_features, compute_node_features
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
        self._baseline = 0.0   # running mean baseline

    def collect_rollout(self, graph) -> dict:
        """Run one episode and collect (state, action, reward) trajectory.

        Args:
            graph: NetworkX graph for this episode.

        Returns:
            Dict with keys: states, actions, rewards, total_revenue.
        """
        env = _make_env(graph, self.cfg)
        obs = env.reset()
        static = compute_static_features(graph)
        n = graph.number_of_nodes()
        nodes = list(graph.nodes())

        states, actions, rewards = [], [], []

        for step in range(n):
            available = env.available_nodes
            if not available:
                break

            features = compute_node_features(
                graph=graph,
                static_features=static,
                S=frozenset(env.S),
                offered=frozenset(env.offered),
                t=env.t,
                n=n,
                k=n,
                env=env,
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
        returns = self.compute_returns(rollout["rewards"])
        G_mean = float(np.mean(returns))

        # Update running baseline
        self._baseline = 0.95 * self._baseline + 0.05 * G_mean
        baseline = self._baseline

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=False)
        policy_losses = []

        static = compute_static_features(graph)
        n = graph.number_of_nodes()
        nodes = list(graph.nodes())

        env = _make_env(graph, self.cfg)
        env.reset()

        for t, (step_data, (node_idx, discount), G_t) in enumerate(
            zip(rollout["states"], rollout["actions"], returns)
        ):
            features, edges, offered_list = step_data
            offered = frozenset(offered_list)
            available_mask = get_available_mask(n, offered, nodes, self.device)

            features_tensor = compute_node_features(
                graph=graph,
                static_features=static,
                S=frozenset(env.S),
                offered=offered,
                t=t,
                n=n,
                k=n,
                env=env,
            )
            data = graph_to_pyg_data(graph, features_tensor, self.device)

            _, _, log_prob = self.policy.select_and_price(
                data.x, data.edge_index, available_mask, greedy=False
            )

            advantage = G_t - baseline
            policy_loss = -log_prob * advantage
            policy_losses.append(policy_loss)

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
