# Revenue Maximization via Joint Seed Selection and Discounting

> **Target venue:** AAAI 2027 | **Status:** 🚧 In Progress

---

## Overview

This paper extends Babaei et al. (2013) *"Revenue Maximization in Social Networks through Discounting"*
and our WSDM 2027 GNN-RL framework to **jointly learn seed selection and discount assignment**
end-to-end via deep reinforcement learning.

The original paper decouples seed selection (who to target) from discount design (what price to offer).
We show this decoupling is suboptimal and train a GNN-based RL agent to make both decisions jointly,
achieving higher revenue on both synthetic and real-world networks.

We additionally propose a **time-discounted revenue objective** (Idea 2) where γ < 1 penalises
delayed revenue — directly connecting the RL discount factor to the economic concept of NPV.

---

## Models

| Model | Encoder | Training | Key idea |
|-------|---------|----------|----------|
| Rev-GNN-IM-RL | GraphSAGE | Imitation + REINFORCE | WSDM backbone, joint action |
| Rev-GAIL-RL-Rich | GraphSAGE | GAIL + REINFORCE | Adversarial imitation from greedy-discount |
| Rev-PPO | GraphSAGE | PPO | On-policy actor-critic |
| Rev-SAC | GraphSAGE | SAC | Off-policy, handles continuous discount well |
| Rev-GraphTransformer | Graph Transformer | GAIL + REINFORCE | Richer attention-based encoder |
| Rev-NPV | GraphSAGE | REINFORCE, γ<1 | Time-discounted revenue (Idea 2) |

**Baselines** (from Babaei et al. 2013): IE-Strategy, µ-Discount, Greedy-Discount, σ-Discount

---

## Project Structure

```
revmax-aaai2027/
├── CLAUDE.md          ← instructions for Claude Code (VS Code)
├── README.md          ← this file
├── VSCODE_PROMPTS.md  ← step-by-step VS Code prompts
├── configs/           ← YAML experiment configs
├── data/              ← raw and processed networks
├── src/
│   ├── env/           ← MDP environment + influence models
│   ├── models/        ← GNN encoders + RL policies
│   ├── training/      ← training loops (GAIL, REINFORCE, PPO, SAC)
│   ├── evaluation/    ← metrics + baseline implementations
│   └── utils/         ← features, logging, helpers, visualization
├── experiments/       ← thin entry point scripts
├── results/           ← logs, checkpoints, figures
└── paper/             ← LaTeX source
```

---

## Setup

```bash
git clone <repo>
cd revmax-aaai2027
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.2+, PyTorch Geometric

---

## Reproducing Baselines

```bash
python experiments/run_baselines.py --config configs/base_config.yaml
```

## Training Models

```bash
# Rev-GNN-IM-RL (fastest to run first)
python experiments/run_rev_gnn_im_rl.py --config configs/experiments/rev_gnn_im_rl.yaml

# Rev-GAIL-RL-Rich
python experiments/run_rev_gail_rl_rich.py --config configs/experiments/rev_gail_rl_rich.yaml

# Rev-PPO
python experiments/run_rev_ppo.py --config configs/experiments/rev_ppo.yaml

# Rev-SAC
python experiments/run_rev_sac.py --config configs/experiments/rev_sac.yaml
```

---

## Key Results (to be filled)

| Method | Forest Fire (mono) | Forest Fire (non-mono) | Facebook |
|--------|-------------------|----------------------|----------|
| IE-Strategy | — | — | — |
| Greedy-Discount | — | — | — |
| Rev-GNN-IM-RL (ours) | — | — | — |
| Rev-GAIL-RL-Rich (ours) | — | — | — |
| Rev-PPO (ours) | — | — | — |
| Rev-SAC (ours) | — | — | — |

---

## Related Papers

- Babaei et al. (2013). Revenue Maximization in Social Networks through Discounting. *SNAM*.
- Seyedin et al. (2027). Deep RL for Fast, Fair, Time-Critical Influence Maximization. *WSDM*.
- Hartline et al. (2008). Optimal Marketing Strategies over Social Networks. *WWW*.
- Hamilton et al. (2017). Inductive Representation Learning on Large Graphs. *NeurIPS*.

---

## Citation

```bibtex
@inproceedings{babaei2027revmax,
  title   = {Joint Seed Selection and Discounting for Revenue Maximization via Deep RL},
  author  = {[Authors]},
  booktitle = {Proceedings of the 41st AAAI Conference on Artificial Intelligence},
  year    = {2027}
}
```
