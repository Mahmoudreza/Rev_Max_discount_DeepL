"""
src/training/imitation_trainer.py

Imitation Trainer (Phase 1 for Rev-GNN-IM-RL).

Trains the scoring head via MSE against the greedy-discount expert's
marginal revenue gains.  The pricing head is also trained in Phase 1
via MSE against the expert's discount values.

Expert: Babaei et al. Greedy-Discount algorithm.
Loss:   L_IM = mean((score_v - marginal_gain_v)^2) over available nodes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

from src.utils.helpers import graph_to_pyg_data, get_available_mask, set_seed
from src.utils.features import compute_static_features, compute_node_features
from src.evaluation.baselines import greedy_discount_trajectory, _make_env


class ImitationTrainer:
    """Phase-1 imitation learning from Greedy-Discount expert.

    Args:
        policy: JointPolicy (scoring head + pricing head + encoder).
        cfg: OmegaConf DictConfig.
        logger: ExperimentLogger instance.
        device: PyTorch device.
    """

    def __init__(self, policy: nn.Module, cfg, logger, device: torch.device) -> None:
        self.policy = policy
        self.cfg = cfg
        self.logger = logger
        self.device = device

        self.optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=cfg.training.imitation_lr,
            weight_decay=cfg.training.weight_decay,
        )

    def generate_expert_trajectory(self, graph) -> List[Dict]:
        """Generate greedy-discount expert trajectory for one graph.

        Args:
            graph: NetworkX graph.

        Returns:
            List of dicts with keys: node_idx, discount, marginal_gain, step.
        """
        trajectory_raw = greedy_discount_trajectory(graph, self.cfg)
        trajectory = []
        for step, (node_idx, discount, marginal_gain) in enumerate(trajectory_raw):
            trajectory.append({
                "node_idx": node_idx,
                "discount": discount,
                "marginal_gain": marginal_gain,
                "step": step,
            })
        return trajectory

    def train(self, train_graphs: list) -> List[float]:
        """Phase-1 imitation training loop.

        For each epoch: for each graph, generate expert trajectory,
        compute MSE between policy scores and expert marginal gains.

        Args:
            train_graphs: List of NetworkX graphs.

        Returns:
            List of mean epoch losses.
        """
        n_epochs = self.cfg.training.imitation_epochs
        log_every = self.cfg.logging.log_every_n_steps
        loss_history = []

        self.policy.train()

        for epoch in range(n_epochs):
            epoch_losses = []

            for graph in train_graphs:
                # Precompute static features once per graph
                static = compute_static_features(graph)

                # Get expert trajectory
                trajectory = self.generate_expert_trajectory(graph)
                n = graph.number_of_nodes()
                nodes = list(graph.nodes())

                # Build episode state
                S = frozenset()
                offered = frozenset()
                env = _make_env(graph, self.cfg)
                env.reset()

                for step_data in trajectory:
                    node_idx = step_data["node_idx"]
                    expert_discount = step_data["discount"]
                    expert_gain = step_data["marginal_gain"]

                    # Compute current 20-dim features
                    features = compute_node_features(
                        graph=graph,
                        static_features=static,
                        S=S,
                        offered=offered,
                        t=len(offered),
                        n=n,
                        k=n,
                        env=env,
                    )

                    # Convert to PyG
                    data = graph_to_pyg_data(graph, features, self.device)
                    available_mask = get_available_mask(n, offered, nodes, self.device)

                    # Forward pass
                    scores, masked_scores, h = self.policy.forward(
                        data.x, data.edge_index, available_mask,
                        return_embeddings=True,
                    )

                    # ── Scoring loss: MSE(score_v, marginal_gain) ─────────────
                    target_gains = torch.zeros(n, device=self.device)
                    target_gains[node_idx] = expert_gain
                    score_loss = F.mse_loss(scores[list(available_mask.nonzero().squeeze(-1))],
                                            target_gains[list(available_mask.nonzero().squeeze(-1))])

                    # ── Pricing loss: MSE(predicted_discount, expert_discount) ─
                    if h is not None:
                        predicted_discount = self.policy.pricing_head(
                            h[node_idx].unsqueeze(0)
                        ).squeeze()
                        pricing_loss = F.mse_loss(
                            predicted_discount,
                            torch.tensor(expert_discount, dtype=torch.float32,
                                         device=self.device),
                        )
                    else:
                        pricing_loss = torch.tensor(0.0, device=self.device)

                    loss = score_loss + pricing_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.cfg.training.grad_clip
                    )
                    self.optimizer.step()

                    epoch_losses.append(loss.item())

                    # Update episode state
                    node = nodes[node_idx]
                    if expert_gain > 0:
                        S = frozenset(S | {node})
                    offered = frozenset(offered | {node})

            mean_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            loss_history.append(mean_loss)

            if epoch % log_every == 0:
                self.logger.log({"imitation/epoch": epoch, "imitation/loss": mean_loss})

        self.logger.info(
            f"ImitationTrainer: {n_epochs} epochs done, final loss={loss_history[-1]:.4f}"
        )
        return loss_history
