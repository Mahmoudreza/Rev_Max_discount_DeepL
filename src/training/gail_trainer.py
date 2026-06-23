"""
src/training/gail_trainer.py

GAIL Trainer (Generative Adversarial Imitation Learning).

Generator: JointPolicy (produces rollouts).
Discriminator: MLP that classifies (state, action) as expert vs agent.
Reward: -log(1 - D(s, a)) per step, used in place of environment reward.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict

from src.utils.helpers import graph_to_pyg_data, get_available_mask
from src.utils.features import compute_static_features, compute_node_features
from src.evaluation.baselines import greedy_discount_trajectory, _make_env


class GAILDiscriminator(nn.Module):
    """GAIL discriminator: MLP on (state_embedding, node_idx_emb, discount).

    Input: cat([global_state (64), node_emb (64), discount (1)]) = 129-dim.
    Output: probability that (s, a) is from expert (sigmoid).

    Args:
        hidden_dim: GNN embedding dimension (default 64).
        disc_hidden: Discriminator MLP hidden size (default 64).
    """

    def __init__(self, hidden_dim: int = 64, disc_hidden: int = 64) -> None:
        super().__init__()
        input_dim = hidden_dim + hidden_dim + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, disc_hidden),
            nn.ReLU(),
            nn.Linear(disc_hidden, disc_hidden),
            nn.ReLU(),
            nn.Linear(disc_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        global_state: torch.Tensor,
        node_emb: torch.Tensor,
        discount: torch.Tensor,
    ) -> torch.Tensor:
        """Predict expert probability.

        Args:
            global_state: Mean-pooled graph embedding, shape (64,).
            node_emb: Selected node embedding, shape (64,).
            discount: Discount value, shape (1,).

        Returns:
            Expert probability in [0, 1], scalar tensor.
        """
        x = torch.cat([global_state, node_emb, discount], dim=-1)
        return self.net(x)


class GAILTrainer:
    """GAIL training loop (Generator + Discriminator alternate updates).

    Args:
        policy: JointPolicy (generator).
        cfg: OmegaConf DictConfig.
        logger: ExperimentLogger.
        device: PyTorch device.
    """

    def __init__(self, policy: nn.Module, cfg, logger, device: torch.device) -> None:
        self.policy = policy
        self.cfg = cfg
        self.logger = logger
        self.device = device

        hidden_dim = cfg.encoder.hidden_dim
        self.discriminator = GAILDiscriminator(hidden_dim=hidden_dim).to(device)

        self.gen_optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=cfg.training.gail_lr_gen,
            weight_decay=cfg.training.weight_decay,
        )
        self.disc_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=cfg.training.gail_lr_disc,
        )

    def _encode_action(
        self, h: torch.Tensor, node_idx: int, discount: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode (state, action) into discriminator input components."""
        global_state = h.mean(dim=0)           # (hidden_dim,)
        node_emb = h[node_idx]                 # (hidden_dim,)
        d = torch.tensor([discount], dtype=torch.float32, device=self.device)
        return global_state, node_emb, d

    def _collect_agent_rollout(self, graph) -> List[Dict]:
        """Collect one agent episode trajectory."""
        env = _make_env(graph, self.cfg)
        env.reset()
        static = compute_static_features(graph)
        n = graph.number_of_nodes()
        nodes = list(graph.nodes())
        trajectory = []

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
                node_idx, discount, _ = self.policy.select_and_price(
                    data.x, data.edge_index, mask, greedy=False
                )

            h = self.policy.encoder(data.x, data.edge_index)
            global_state, node_emb, d = self._encode_action(h, node_idx, discount)
            trajectory.append({
                "global_state": global_state.detach(),
                "node_emb": node_emb.detach(),
                "discount": d.detach(),
                "node_idx": node_idx,
                "discount_val": discount,
                "data": (data.x.detach(), data.edge_index, mask),
            })

            if node_idx in available:
                env.step(node_idx, discount)
            else:
                env.step(available[0], discount)

        return trajectory

    def _get_expert_demos(self, graph) -> List[Tuple]:
        """Get expert (state, action) pairs."""
        expert_traj = greedy_discount_trajectory(graph, self.cfg)
        env = _make_env(graph, self.cfg)
        env.reset()
        static = compute_static_features(graph)
        n = graph.number_of_nodes()
        nodes = list(graph.nodes())
        demos = []

        for node_idx, discount, _ in expert_traj:
            features = compute_node_features(
                graph=graph, static_features=static,
                S=frozenset(env.S), offered=frozenset(env.offered),
                t=env.t, n=n, k=n, env=env,
            )
            data = graph_to_pyg_data(graph, features, self.device)

            with torch.no_grad():
                h = self.policy.encoder(data.x, data.edge_index)
            global_s, node_e, d = self._encode_action(h.detach(), node_idx, discount)
            demos.append((global_s, node_e, d))

            # Advance env state
            node = nodes[node_idx] if node_idx < n else nodes[0]
            if node not in env.offered:
                env.S.add(node)
                env.offered.add(node)
                env._influence_cache = {}
            env.t += 1

        return demos

    def train(self, train_graphs: list) -> Dict:
        """GAIL training loop.

        Args:
            train_graphs: List of training graphs.

        Returns:
            Dict with disc_losses, gen_losses.
        """
        n_epochs = self.cfg.training.gail_epochs
        log_every = self.cfg.logging.log_every_n_steps
        disc_losses, gen_losses = [], []

        for epoch in range(n_epochs):
            for graph in train_graphs:
                # 1. Discriminator update
                agent_traj = self._collect_agent_rollout(graph)
                expert_demos = self._get_expert_demos(graph)

                disc_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                d_losses = []

                min_len = min(len(agent_traj), len(expert_demos))
                for agent_step, (e_gs, e_ne, e_d) in zip(
                    agent_traj[:min_len], expert_demos[:min_len]
                ):
                    d_expert = self.discriminator(e_gs, e_ne, e_d)
                    d_agent = self.discriminator(
                        agent_step["global_state"],
                        agent_step["node_emb"],
                        agent_step["discount"],
                    )
                    # Expert → 1, Agent → 0
                    d_loss = F.binary_cross_entropy(
                        d_expert, torch.ones(1, device=self.device)
                    ) + F.binary_cross_entropy(
                        d_agent, torch.zeros(1, device=self.device)
                    )
                    d_losses.append(d_loss)

                if d_losses:
                    disc_loss = torch.stack(d_losses).mean()
                    self.disc_optimizer.zero_grad()
                    disc_loss.backward()
                    self.disc_optimizer.step()
                    disc_losses.append(float(disc_loss.item()))

                # 2. Generator (policy) update: maximize -log(1-D(s,a))
                gen_loss_list = []
                for step_data in agent_traj:
                    x, edge_idx, mask = step_data["data"]
                    node_idx = step_data["node_idx"]
                    discount_val = step_data["discount_val"]

                    _, _, log_prob = self.policy.select_and_price(
                        x, edge_idx, mask, greedy=False
                    )
                    h = self.policy.encoder(x, edge_idx)
                    global_s, node_e, d = self._encode_action(h, node_idx, discount_val)
                    d_pred = self.discriminator(global_s, node_e, d)

                    # GAIL reward: -log(1 - D)
                    reward = -torch.log(1 - d_pred + 1e-8)
                    gen_loss_list.append(-log_prob * reward.squeeze())

                if gen_loss_list:
                    gen_loss = torch.stack(gen_loss_list).mean()
                    self.gen_optimizer.zero_grad()
                    gen_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.cfg.training.grad_clip
                    )
                    self.gen_optimizer.step()
                    gen_losses.append(float(gen_loss.item()))

            if epoch % log_every == 0:
                d_l = disc_losses[-1] if disc_losses else 0.0
                g_l = gen_losses[-1] if gen_losses else 0.0
                self.logger.log({
                    "gail/epoch": epoch,
                    "gail/disc_loss": d_l,
                    "gail/gen_loss": g_l,
                })

        return {"disc_losses": disc_losses, "gen_losses": gen_losses}
