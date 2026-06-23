"""
src/training/sac_trainer.py

SAC Trainer for the mixed discrete+continuous revenue MDP.

Uses twin Q-networks with a replay buffer (off-policy).
Temperature alpha adjusts entropy regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from typing import Dict, List

from src.utils.helpers import graph_to_pyg_data, get_available_mask
from src.utils.features import compute_static_features, compute_node_features
from src.evaluation.baselines import _make_env


class ReplayBuffer:
    """Simple replay buffer for SAC.

    Stores transitions as dicts; samples random minibatches.
    """

    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def push(self, transition: Dict) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Dict]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


class SACTrainer:
    """SAC trainer: alternate policy and Q-network updates.

    Args:
        sac_policy: SACPolicy instance.
        q_net: SACQNetwork (contains twin Q-networks).
        q_net_target: Target SACQNetwork (EMA-updated).
        cfg: OmegaConf DictConfig.
        logger: ExperimentLogger.
        device: PyTorch device.
    """

    def __init__(
        self,
        sac_policy: nn.Module,
        q_net: nn.Module,
        q_net_target: nn.Module,
        cfg,
        logger,
        device: torch.device,
    ) -> None:
        self.sac_policy = sac_policy
        self.q_net = q_net
        self.q_net_target = q_net_target
        self.cfg = cfg
        self.logger = logger
        self.device = device

        # Copy weights to target net
        self.q_net_target.load_state_dict(q_net.state_dict())

        self.policy_optimizer = torch.optim.Adam(
            sac_policy.parameters(), lr=cfg.training.sac_lr
        )
        self.q_optimizer = torch.optim.Adam(
            q_net.parameters(), lr=cfg.training.sac_lr
        )

        # Learnable temperature (log alpha)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-4)
        self.target_entropy = -1.0  # target entropy for discrete + continuous

        self.replay_buffer = ReplayBuffer(cfg.training.sac_buffer_size)
        self.batch_size = cfg.training.sac_batch_size
        self.tau = cfg.training.sac_tau
        self.alpha = cfg.training.sac_alpha

    def collect_episode(self, graph) -> float:
        """Run one episode, push transitions to replay buffer.

        Args:
            graph: NetworkX graph.

        Returns:
            Total episode revenue.
        """
        env = _make_env(graph, self.cfg)
        env.reset()
        static = compute_static_features(graph)
        n = graph.number_of_nodes()
        nodes = list(graph.nodes())

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
                node_idx, discount, log_prob = self.sac_policy.sample_action(
                    data.x, data.edge_index, mask
                )

            if node_idx not in available:
                node_idx = available[0]

            prev_features = features.copy()
            _, reward, done, _ = env.step(node_idx, discount)

            self.replay_buffer.push({
                "x": data.x.detach(),
                "edge_index": data.edge_index,
                "mask": mask.detach(),
                "node_idx": node_idx,
                "discount": discount,
                "reward": reward,
                "done": done,
            })

            if done:
                break

        return env.total_revenue

    def update_q_networks(self, batch: List[Dict]) -> float:
        """TD update for twin Q-networks.

        Args:
            batch: List of transition dicts.

        Returns:
            Mean Q-loss.
        """
        q_losses = []
        for transition in batch:
            x = transition["x"]
            edge_idx = transition["edge_index"]
            node_idx = transition["node_idx"]
            discount = transition["discount"]
            reward = transition["reward"]

            q1, q2 = self.q_net(x, edge_idx, node_idx, discount)

            with torch.no_grad():
                # No next state in this sequential setting — use reward as target
                target_q = torch.tensor(reward, dtype=torch.float32, device=self.device)

            q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            q_losses.append(q_loss)

        if q_losses:
            loss = torch.stack(q_losses).mean()
            self.q_optimizer.zero_grad()
            loss.backward()
            self.q_optimizer.step()
            return float(loss.item())
        return 0.0

    def update_policy(self, batch: List[Dict]) -> float:
        """Policy update: maximize Q - alpha * log_pi.

        Args:
            batch: List of transitions.

        Returns:
            Mean policy loss.
        """
        p_losses = []
        for transition in batch:
            x = transition["x"]
            edge_idx = transition["edge_index"]
            mask = transition["mask"]

            node_idx, discount, log_prob = self.sac_policy.sample_action(
                x, edge_idx, mask
            )

            q1, q2 = self.q_net(x, edge_idx, node_idx, discount)
            min_q = torch.min(q1, q2)

            alpha = self.log_alpha.exp().detach()
            policy_loss = (alpha * log_prob - min_q)
            p_losses.append(policy_loss)

        if p_losses:
            loss = torch.stack(p_losses).mean()
            self.policy_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.sac_policy.parameters(), self.cfg.training.grad_clip
            )
            self.policy_optimizer.step()

            # Temperature update
            alpha_loss = -(self.log_alpha * (
                log_prob.detach() + self.target_entropy
            )).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            return float(loss.item())
        return 0.0

    def _soft_update_target(self) -> None:
        """EMA update target networks: theta_target = tau*theta + (1-tau)*theta_target."""
        for param, target_param in zip(
            self.q_net.parameters(), self.q_net_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def train(self, train_graphs: list) -> Dict:
        """SAC training loop.

        Args:
            train_graphs: List of training graphs.

        Returns:
            Dict with revenues, q_losses, policy_losses.
        """
        n_updates = self.cfg.training.sac_n_updates
        log_every = self.cfg.logging.log_every_n_steps
        revenues, q_losses, p_losses = [], [], []

        self.sac_policy.train()
        self.q_net.train()

        for update in range(n_updates):
            graph = train_graphs[update % len(train_graphs)]
            rev = self.collect_episode(graph)
            revenues.append(rev)

            if len(self.replay_buffer) >= self.batch_size:
                batch = self.replay_buffer.sample(self.batch_size)
                ql = self.update_q_networks(batch)
                pl = self.update_policy(batch)
                self._soft_update_target()
                q_losses.append(ql)
                p_losses.append(pl)

            if update % log_every == 0:
                self.logger.log({
                    "sac/update": update,
                    "sac/revenue": rev,
                    "sac/q_loss": q_losses[-1] if q_losses else 0.0,
                    "sac/policy_loss": p_losses[-1] if p_losses else 0.0,
                })

        return {"revenues": revenues, "q_losses": q_losses, "policy_losses": p_losses}
