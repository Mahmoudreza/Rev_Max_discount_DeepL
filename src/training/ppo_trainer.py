"""
src/training/ppo_trainer.py

PPO Trainer with clipped surrogate objective.

Collects n_rollouts per update, then runs ppo_epochs minibatch updates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
import numpy as np

from src.utils.helpers import graph_to_pyg_data, get_available_mask
from src.utils.features import compute_static_features, compute_node_features
from src.evaluation.baselines import _make_env


class PPOTrainer:
    """PPO trainer for JointPolicy wrapped in PPOPolicy.

    Args:
        ppo_policy: PPOPolicy instance (actor + value head).
        cfg: OmegaConf DictConfig.
        logger: ExperimentLogger.
        device: PyTorch device.
    """

    def __init__(self, ppo_policy: nn.Module, cfg, logger, device: torch.device) -> None:
        self.ppo_policy = ppo_policy
        self.cfg = cfg
        self.logger = logger
        self.device = device

        self.optimizer = torch.optim.Adam(
            ppo_policy.parameters(),
            lr=cfg.training.ppo_lr,
            weight_decay=cfg.training.weight_decay,
        )
        self.clip_eps = cfg.training.ppo_clip
        self.ppo_epochs = cfg.training.ppo_epochs
        self.ent_coef = cfg.training.ppo_entropy_coef

    def collect_rollout(self, graph) -> Dict:
        """Run one episode and store all data for PPO update.

        Args:
            graph: NetworkX graph.

        Returns:
            Episode data dict.
        """
        env = _make_env(graph, self.cfg)
        env.reset()
        static = compute_static_features(graph)
        n = graph.number_of_nodes()
        nodes = list(graph.nodes())

        episode = {"x": [], "edge_index": [], "masks": [], "node_idxs": [],
                   "discounts": [], "log_probs": [], "rewards": [], "values": []}

        for step in range(n):
            available = env.available_nodes
            if not available:
                break

            features = compute_node_features(
                graph=graph, static_features=static,
                S=frozenset(env.S), offered=frozenset(env.offered),
                t=env.t, n=n, k=n, env=env,
            )
            data = graph_to_pyg_data(graph, features, self.device)
            mask = get_available_mask(n, frozenset(env.offered), nodes, self.device)

            with torch.no_grad():
                node_idx, discount, log_prob = self.ppo_policy.select_and_price(
                    data.x, data.edge_index, mask, greedy=False
                )
                value = self.ppo_policy.get_value(data.x, data.edge_index)

            if node_idx not in available:
                node_idx = available[0]

            _, reward, done, _ = env.step(node_idx, discount)

            episode["x"].append(data.x.detach())
            episode["edge_index"].append(data.edge_index)
            episode["masks"].append(mask.detach())
            episode["node_idxs"].append(node_idx)
            episode["discounts"].append(discount)
            episode["log_probs"].append(log_prob.detach())
            episode["rewards"].append(reward)
            episode["values"].append(value.detach())

            if done:
                break

        episode["total_revenue"] = env.total_revenue
        return episode

    def compute_gae(self, rewards: List[float], values: List[float],
                    last_val: float = 0.0) -> List[float]:
        """Generalized Advantage Estimation (GAE).

        Args:
            rewards: Step rewards.
            values: State values.
            last_val: Bootstrap value at terminal.

        Returns:
            List of advantages.
        """
        gamma = getattr(self.cfg.reward, "gamma", 1.0)
        gae_lambda = 0.95
        T = len(rewards)
        advantages = [0.0] * T
        gae = 0.0

        for t in reversed(range(T)):
            next_val = values[t + 1] if t + 1 < T else last_val
            delta = rewards[t] + gamma * next_val - values[t]
            gae = delta + gamma * gae_lambda * gae
            advantages[t] = gae

        return advantages

    def update(self, episodes: List[Dict]) -> Dict[str, float]:
        """PPO update over a batch of episodes.

        Args:
            episodes: List of episode dicts from collect_rollout().

        Returns:
            Dict of losses.
        """
        all_losses = {"policy": [], "value": [], "entropy": []}

        for episode in episodes:
            rewards = episode["rewards"]
            values = [float(v.item()) for v in episode["values"]]
            advantages = self.compute_gae(rewards, values)

            for _ in range(self.ppo_epochs):
                for t in range(len(rewards)):
                    x = episode["x"][t]
                    edge_idx = episode["edge_index"][t]
                    mask = episode["masks"][t]
                    node_idx = episode["node_idxs"][t]
                    discount = episode["discounts"][t]
                    old_log_prob = episode["log_probs"][t]
                    adv = advantages[t]
                    ret = rewards[t] + values[min(t + 1, len(values) - 1)]

                    eval_out = self.ppo_policy.evaluate_actions(
                        x, edge_idx, mask, node_idx, discount
                    )
                    new_log_prob = eval_out["log_prob"]
                    entropy = eval_out["entropy"]
                    value = eval_out["value"]

                    ratio = torch.exp(new_log_prob - old_log_prob)
                    adv_t = torch.tensor(adv, dtype=torch.float32, device=self.device)
                    surr1 = ratio * adv_t
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t
                    policy_loss = -torch.min(surr1, surr2)

                    ret_t = torch.tensor(ret, dtype=torch.float32, device=self.device)
                    value_loss = F.mse_loss(value, ret_t)
                    ent_loss = -self.ent_coef * entropy

                    loss = policy_loss + 0.5 * value_loss + ent_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.ppo_policy.parameters(), self.cfg.training.grad_clip
                    )
                    self.optimizer.step()

                    all_losses["policy"].append(float(policy_loss.item()))
                    all_losses["value"].append(float(value_loss.item()))
                    all_losses["entropy"].append(float(entropy.item()))

        return {k: float(np.mean(v)) if v else 0.0 for k, v in all_losses.items()}

    def train(self, train_graphs: list) -> Dict:
        """PPO training loop.

        Args:
            train_graphs: List of NetworkX training graphs.

        Returns:
            Dict with losses and revenues.
        """
        n_epochs = self.cfg.training.n_epochs
        log_every = self.cfg.logging.log_every_n_steps
        revenues = []

        self.ppo_policy.train()

        for epoch in range(n_epochs):
            episodes = [self.collect_rollout(g) for g in train_graphs[:2]]
            losses = self.update(episodes)
            rev = float(np.mean([e["total_revenue"] for e in episodes]))
            revenues.append(rev)

            if epoch % log_every == 0:
                self.logger.log({
                    "ppo/epoch": epoch,
                    "ppo/revenue": rev,
                    **{f"ppo/{k}_loss": v for k, v in losses.items()},
                })

        return {"revenues": revenues}
