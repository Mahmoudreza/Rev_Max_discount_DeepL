"""
src/training/imitation_trainer.py

Imitation Trainer (Phase 1 for Rev-GNN-IM-RL).

Trains the SCORING HEAD ONLY via masked cross-entropy against the
greedy-discount expert's node selection.  The pricing head receives NO
gradient here — it learns in Phase 1.5 (pricing-only REINFORCE).

Expert: Babaei et al. Greedy-Discount algorithm.
Loss:   L_IM = CrossEntropy(masked_logits, expert_node_idx)
        size-invariant ranking loss (not MSE on absolute magnitudes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

from src.utils.helpers import graph_to_pyg_data, get_available_mask, set_seed
from src.utils.features import (compute_static_features, compute_node_features,
                                  build_graph_feature_cache, compute_node_features_fast)
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
            List of dicts with keys: node_idx, discount, marginal_gain,
            price, accepted, step.
        """
        trajectory_raw = greedy_discount_trajectory(graph, self.cfg)
        trajectory = []
        for step, traj_item in enumerate(trajectory_raw):
            trajectory.append({**traj_item, "step": step})
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

        # Precompute static features + vectorized graph caches ONCE per graph
        graph_statics = {id(g): compute_static_features(g) for g in train_graphs}
        graph_caches = {id(g): build_graph_feature_cache(g, graph_statics[id(g)])
                        for g in train_graphs}

        self.policy.train()

        for epoch in range(n_epochs):
            epoch_losses = []

            for graph in train_graphs:
                static = graph_statics[id(graph)]   # cached — not recomputed

                # Get expert trajectory
                trajectory = self.generate_expert_trajectory(graph)
                n = graph.number_of_nodes()
                nodes = list(graph.nodes())

                # Build episode state
                S = frozenset()
                offered = frozenset()
                env = _make_env(graph, self.cfg)
                env.reset()
                cache = graph_caches[id(graph)]
                episode_losses: List[torch.Tensor] = []

                for step_data in trajectory:
                    node_idx = step_data["node_idx"]
                    expert_discount = step_data["discount"]
                    expert_gain = step_data["marginal_gain"]

                    features = compute_node_features_fast(
                        cache=cache, S=S, offered=offered,
                        t=len(offered), k=n, env=env,
                    )
                    data = graph_to_pyg_data(graph, features, self.device)
                    available_mask = get_available_mask(n, offered, nodes, self.device)

                    scores, masked_scores, h = self.policy.forward(
                        data.x, data.edge_index, available_mask,
                        return_embeddings=True,
                    )

                    # Masked cross-entropy: size-invariant ranking loss
                    masked_logits = scores.clone()
                    masked_logits[~available_mask] = float('-inf')
                    loss_node = F.cross_entropy(
                        masked_logits.unsqueeze(0),
                        torch.tensor([node_idx], device=self.device),
                    )

                    # Pricing supervision: MSE between Beta-mean and expert discount.
                    # Uses get_discount_distribution() instead of raw pricing_head()
                    # so it stays compatible with both 1-output (old) and 2-output
                    # (Beta distribution) pricing heads.
                    pricing_weight = float(
                        getattr(self.cfg.training, "pricing_loss_weight", 0.0)
                    )
                    if pricing_weight > 0.0:
                        if hasattr(self.policy, 'get_discount_distribution'):
                            # Beta distribution policy: supervise distribution mean
                            pred_disc = self.policy.get_discount_distribution(
                                h[node_idx]
                            ).mean
                        else:
                            # Legacy Sigmoid policy (backward-compat)
                            pred_disc = self.policy.pricing_head(
                                h[node_idx].unsqueeze(0)
                            ).squeeze()
                        expert_disc_t = torch.tensor(
                            expert_discount, dtype=torch.float32,
                            device=self.device,
                        )
                        loss_price = F.mse_loss(pred_disc, expert_disc_t)
                        loss = loss_node + pricing_weight * loss_price
                    else:
                        loss = loss_node

                    episode_losses.append(loss)

                    # Update episode state: add to S if expert accepted the node
                    # (includes FREE seeds with marginal=0 AND priced+accepted)
                    node = nodes[node_idx]
                    if step_data.get("accepted", expert_gain > 0):
                        S = frozenset(S | {node})
                        env.S.add(node)
                        for _nb in graph.neighbors(node):
                            env._influence_cache.pop(_nb, None)
                            env._true_val_cache.pop(_nb, None)
                            env._est_val_cache.pop(_nb, None)
                    offered = frozenset(offered | {node})
                    env.offered.add(node)
                    env.t += 1

                # ONE backward + ONE optimizer step per episode (vs n per step before)
                if episode_losses:
                    ep_loss = torch.stack(episode_losses).mean()
                    self.optimizer.zero_grad()
                    ep_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.cfg.training.grad_clip
                    )
                    self.optimizer.step()
                    epoch_losses.append(ep_loss.item())

            mean_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            loss_history.append(mean_loss)

            if epoch % log_every == 0:
                self.logger.log({"imitation/epoch": epoch, "imitation/loss": mean_loss})

            # Top-k accuracy check every 50 epochs (measures whether model is
            # actually learning to rank expert nodes, not just minimising CE numerically)
            if epoch % 50 == 0:
                acc_graph = train_graphs[0]
                acc_traj  = self.generate_expert_trajectory(acc_graph)
                acc_cache = graph_caches[id(acc_graph)]
                acc_n     = acc_graph.number_of_nodes()
                acc_nodes = list(acc_graph.nodes())
                acc_env   = _make_env(acc_graph, self.cfg)
                acc_env.reset()
                S_acc     = frozenset()
                off_acc   = frozenset()
                top1_hits = top5_hits = n_acc = 0

                self.policy.eval()
                for step_data in acc_traj[:10]:
                    nidx = step_data["node_idx"]
                    feats = compute_node_features_fast(
                        cache=acc_cache, S=S_acc, offered=off_acc,
                        t=len(off_acc), k=acc_n, env=acc_env,
                    )
                    data = graph_to_pyg_data(acc_graph, feats, self.device)
                    amask = get_available_mask(acc_n, off_acc, acc_nodes, self.device)
                    with torch.no_grad():
                        sc, _, _ = self.policy.forward(
                            data.x, data.edge_index, amask, return_embeddings=True
                        )
                    rank = int((sc[amask] > sc[nidx]).sum().item()) + 1
                    top1_hits += rank == 1
                    top5_hits += rank <= 5
                    n_acc     += 1
                    nd = acc_nodes[nidx]
                    if step_data["marginal_gain"] > 0:
                        S_acc = frozenset(S_acc | {nd})
                    off_acc = frozenset(off_acc | {nd})
                    acc_env.offered.add(nd)
                    acc_env.t += 1
                self.policy.train()

                top1 = top1_hits / max(n_acc, 1)
                top5 = top5_hits / max(n_acc, 1)
                self.logger.info(
                    f"Epoch {epoch:4d}: CE={mean_loss:.4f}  "
                    f"top1={top1:.2%}  top5={top5:.2%}"
                )
                self.logger.log({
                    "imitation/epoch": epoch, "imitation/loss": mean_loss,
                    "imitation/top1_acc": top1, "imitation/top5_acc": top5,
                })

        self.logger.info(
            f"ImitationTrainer: {n_epochs} epochs done, final loss={loss_history[-1]:.4f}"
        )
        return loss_history
