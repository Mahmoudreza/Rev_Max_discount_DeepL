# VS Code Claude Code Prompts — Revenue Maximization (AAAI 2027)
# Use these IN ORDER. Each prompt is self-contained and builds on the previous.
# Before running any prompt, always make sure the venv is active.

---

## SETUP

### Prompt S.1 — Environment setup
```
Read CLAUDE.md completely. Then:
1. Create a Python 3.10+ virtual environment called `venv` and activate it.
2. Install all packages from requirements.txt.
3. Verify these imports work: torch, torch_geometric, networkx, gymnasium, stable_baselines3.
4. Print: torch version, cuda available, torch_geometric version.
```

### Prompt S.2 — Verify existing files
```
Read CLAUDE.md. List all files currently in src/ and confirm which ones exist.
Then run: pytest tests/ -v --tb=short
Report which tests pass and which are missing.
```

---

## PHASE 1 — Graph Infrastructure

### Prompt 1.1 — Graph generators
```
Read CLAUDE.md and src/env/revenue_env.py. Then implement src/env/graph_generators.py with:

1. `generate_forest_fire(n, p, pb, seed)` → nx.Graph
   Forest Fire model: Leskovec et al. (2007). Each new node chooses random ambassador w,
   selects out-links with prob p and in-links with prob pb.

2. `generate_modular_forest_fire(module_sizes, p, pb, inter_prob, seed)` → nx.Graph
   Build n_modules independent forest fire graphs, then rewire inter-modular edges
   with probability inter_prob.

3. `load_real_network(name, data_dir)` → nx.Graph
   Load edge list from data/raw/{name}.txt. Supported names:
   "facebook", "yeast", "wiki", "newman", "hep"

4. `build_graph_from_config(cfg)` → nx.Graph
   Factory function that reads cfg.graph and calls the right generator.

All functions must: handle directed/undirected properly, use seed for reproducibility,
return undirected graphs (convert if needed), have type hints and Google docstrings.
```

### Prompt 1.2 — Download real network data
```
Read CLAUDE.md and src/env/graph_generators.py. Then write a script
experiments/download_networks.py that downloads the 5 real networks from SNAP:

- Facebook UCI: https://snap.stanford.edu/data/CollegeMsg.txt.gz → data/raw/facebook.txt
- Yeast PPI: https://snap.stanford.edu/data/bio-yeast.txt.gz → data/raw/yeast.txt
- Wiki-vote: https://snap.stanford.edu/data/wiki-Vote.txt.gz → data/raw/wiki.txt
- Newman collab: https://snap.stanford.edu/data/ca-CondMat.txt.gz → data/raw/newman.txt
- HEP citation: https://snap.stanford.edu/data/cit-HepTh.txt.gz → data/raw/hep.txt

After downloading, preprocess each: strip comment lines, convert to undirected nx.Graph,
save as pickle to data/processed/{name}.pkl. Print node/edge counts for each.
```

---

## PHASE 2 — Environment + Features

### Prompt 2.1 — Test the revenue environment
```
Read CLAUDE.md and src/env/revenue_env.py. Then implement tests/test_env.py with:

1. test_reset(): env resets correctly (empty S, empty offered, t=0)
2. test_step_accept(): step with discount=1.0 always results in acceptance (free item)
3. test_step_reject(): step with discount=0.0 on node with 0 influence → offered_price=0 → rejected
4. test_valuation_increases(): after adding influential node to S, valuation of its neighbors increases
5. test_revenue_sum(): total_revenue = sum of all revenue_step values
6. test_npv_mode(): with gamma=0.9, rewards are discounted by 0.9^t
7. test_monotone_model(): valuation is non-decreasing as more nodes join S
8. test_non_monotone_model(): valuation eventually decreases as S gets very large

Use a small graph (n=20, BA model) for all tests.
Run: pytest tests/test_env.py -v
```

### Prompt 2.2 — Test and verify node features
```
Read CLAUDE.md, src/utils/features.py, and src/env/revenue_env.py.
Then implement tests/test_features.py with:

1. test_feature_shape(): compute_node_features returns shape (n, 20)
2. test_static_features_cached(): calling compute_static_features twice gives same result
3. test_current_influence_zero_at_start(): dim 16 = 0 for all nodes when S is empty
4. test_influence_updates_after_step(): dim 16 increases for neighbors after a buyer joins S
5. test_was_offered_flag(): dim 18 = 1 for nodes in offered set, 0 otherwise
6. test_all_features_in_range(): all feature values are finite (no NaN, no inf)

Run: pytest tests/test_features.py -v
```

---

## PHASE 3 — Baselines (Babaei et al. 2013)

### Prompt 3.1 — Implement all baselines
```
Read CLAUDE.md, src/env/revenue_env.py, and the description of baselines in CLAUDE.md.
Then implement src/evaluation/baselines.py with:

1. `ie_strategy(graph, cfg)` → float
   Influence-and-Exploit: greedy hill climbing selects S (top-k by influence gain),
   gives item FREE to S, then offers remaining buyers at myopic price.
   Returns total revenue.

2. `mu_discount(graph, cfg)` → float
   Discount based on average degree µ (Section 4.1 in Babaei et al. 2013).
   Step j: offer price f(j/µ) until influence threshold is met.
   Returns total revenue.

3. `greedy_discount(graph, cfg)` → float
   Degree-based greedy discount (Section 4.2 in Babaei et al. 2013).
   k=6 influence regions, maximize buyers in blue interval.
   Returns total revenue.

4. `sigma_discount(graph, cfg)` → float
   Standard deviation based discount (Section 4.2.1 in Babaei et al. 2013).
   Uses µ and σ of degree distribution to identify influential nodes.
   Returns total revenue.

5. `run_all_baselines(graph, cfg, n_trials=10)` → Dict[str, float]
   Runs all 4 baselines, averages over n_trials (different link weight samples).
   Returns dict: {"ie_strategy": X, "mu_discount": Y, "greedy_discount": Z, "sigma_discount": W}

All baselines should use cfg.influence.n_mc_samples MC samples for valuation estimation.
Type hints and docstrings required.
```

### Prompt 3.2 — Run and verify baselines
```
Read CLAUDE.md, src/evaluation/baselines.py, and src/env/graph_generators.py.
Then implement experiments/run_baselines.py that:

1. Generates a Forest Fire graph (n=1000, p=0.37, pb=0.32) from cfg
2. Runs all 4 baselines (both monotone and non-monotone influence models)
3. Prints a rich table showing revenue for each method
4. Saves results to results/logs/baselines_{network}_{model}.csv
5. Verifies the ordering: greedy_discount > sigma_discount > mu_discount > ie_strategy
   (this is the result in Babaei et al. 2013 Figure 3)

Run: python experiments/run_baselines.py --config configs/base_config.yaml
If ordering is wrong, check the influence model and weight sampling.
```

---

## PHASE 4 — GNN Models

### Prompt 4.1 — Graph Transformer encoder
```
Read CLAUDE.md and src/models/encoders/graphsage.py. Then implement
src/models/encoders/graph_transformer.py with a GraphTransformerEncoder class
that has the SAME interface as GraphSAGEEncoder (same __init__ args, same forward signature)
but uses PyTorch Geometric's TransformerConv instead of SAGEConv.

The architecture should be:
- Input projection: Linear(in_dim, hidden_dim) + LayerNorm + ReLU
- 2x TransformerConv(hidden_dim, hidden_dim, heads=4, concat=False) with residual
- Same LayerNorm and residual pattern as GraphSAGEEncoder

Test: both encoders should accept the same (x, edge_index) input and return (n, 64).
```

### Prompt 4.2 — Build graph-to-PyG conversion utility
```
Read CLAUDE.md, src/utils/features.py, and src/models/encoders/graphsage.py.
Then add to src/utils/helpers.py:

1. `graph_to_pyg_data(graph, features, device)` → torch_geometric.data.Data
   Converts networkx graph + numpy feature matrix to a PyG Data object.
   edge_index should be bidirectional for undirected graphs.

2. `get_available_mask(n, offered, device)` → torch.Tensor
   Returns boolean mask of shape (n,): True for nodes not in offered set.

3. `set_seed(seed)`, `get_device()`, `load_config(path, overrides)`,
   `get_project_root()`, `ensure_dir(path)` (same as before).

Type hints and docstrings required.
```

---

## PHASE 5 — Training Pipelines

### Prompt 5.1 — Imitation trainer (Phase 1 for Rev-GNN-IM-RL)
```
Read CLAUDE.md, src/models/policies/joint_policy.py, src/env/revenue_env.py,
and the WSDM paper description (GNN-IM-RL Section 4.3).

Implement src/training/imitation_trainer.py with class ImitationTrainer:

The expert for revenue maximization is the Greedy-Discount algorithm from baselines.py.
At each step, the expert chooses: (node = greedy_discount_choice, discount = computed_discount).

The imitation loss is:
  L_IM = (1/|available|) * sum_{v available} (score_v - delta_v)^2
where delta_v = marginal revenue gain of offering node v next with expert discount.

Training loop:
1. Generate expert trajectories from greedy_discount on training graphs
2. For each step in trajectory: compute node scores from policy
3. Compute MSE loss against expert's marginal gains
4. Backprop through scoring head and encoder ONLY (not pricing head in Phase 1)
5. Log loss every cfg.logging.log_every_n_steps steps

__init__(policy, cfg, logger, device)
train(train_graphs) → returns loss history
generate_expert_trajectory(graph) → list of (node, discount, marginal_gain) tuples
```

### Prompt 5.2 — GAIL trainer (Phase 1 for Rev-GAIL-RL-Rich)
```
Read CLAUDE.md, src/training/imitation_trainer.py, the WSDM paper Section 4.4,
and src/models/policies/joint_policy.py.

Implement src/training/gail_trainer.py with class GAILTrainer:

Discriminator D: shares the GraphSAGE backbone, outputs scalar ∈ [0,1].
  D(G, S_t, v) = "is this choice (v at step t) from the expert or the policy?"

Generator loss (Eq. 17 in WSDM): L_G = -E[log D(v_hat)]
Discriminator loss: L_D = E[log D(v*)] + E[log(1 - D(v_hat))]

Expert trajectories: greedy_discount algorithm choices (same as imitation_trainer.py).

__init__(policy, cfg, logger, device)
train_gail_phase(train_graphs) → initializes backbone with greedy-like behavior
_build_discriminator() → nn.Module sharing encoder backbone
_gail_loss(real_traj, fake_traj) → (L_D, L_G) tensors
```

### Prompt 5.3 — REINFORCE trainer (Phase 2 for all models)
```
Read CLAUDE.md, src/training/imitation_trainer.py, src/env/revenue_env.py,
and src/models/policies/joint_policy.py.

Implement src/training/reinforce_trainer.py with class REINFORCETrainer:

Reward:
  Idea 1 (flat):  r_t = price_paid if accepted else 0
  Idea 2 (NPV):   r_t = gamma^t * price_paid if accepted else 0

REINFORCE update (episode-level):
  G = sum_t r_t  (total episode return)
  loss = -sum_t log_prob(action_t) * G   (with baseline: subtract mean G)

__init__(policy, cfg, logger, device)
train(train_graphs, n_epochs) → trains policy via REINFORCE
run_episode(env, graph, greedy=False) → (actions, log_probs, rewards, total_revenue)
_compute_returns(rewards) → discounted returns tensor

The joint action (node, discount) log_prob:
  log_prob = log_prob_node + log_prob_discount
where log_prob_discount is computed from a Gaussian distribution centered at
the pricing head output (with small fixed std=0.1 for exploration).
```

### Prompt 5.4 — PPO trainer
```
Read CLAUDE.md, src/training/reinforce_trainer.py, and configs/base_config.yaml.

Implement src/training/ppo_trainer.py with class PPOTrainer using the PPO algorithm:

The action space is:
  - Discrete: node selection (n_available choices)
  - Continuous: discount ∈ [0,1] from pricing head

Use a combined actor-critic architecture:
  - Actor: JointPolicy (scoring head for node selection + pricing head for discount)
  - Critic: separate MLP head on top of the GNN encoder → scalar value V(s)

PPO clip loss (standard):
  r_t = pi_new(a_t|s_t) / pi_old(a_t|s_t)
  L_clip = min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)
  L_total = -L_clip + cfg.training.ppo_entropy_coef * entropy

__init__(policy, cfg, logger, device)
train(train_graphs) → full PPO training loop with cfg.training.ppo_epochs inner epochs
collect_rollouts(envs, n_steps) → rollout buffer
update_policy(rollout_buffer) → PPO gradient step
```

### Prompt 5.5 — SAC trainer
```
Read CLAUDE.md, src/training/ppo_trainer.py, and configs/base_config.yaml.

Implement src/training/sac_trainer.py with class SACTrainer.

SAC is off-policy and handles continuous actions well — ideal for the discount dimension.
Use the standard SAC formulation:
  - Actor: outputs (node_logits, discount_mean) — node selection discrete, discount continuous
  - Two Q-networks: Q1, Q2 sharing GNN encoder
  - Target networks with soft update (tau from cfg)
  - Replay buffer of (state, action, reward, next_state, done) tuples
  - Entropy regularization with temperature alpha

Key challenge: mixed discrete+continuous action space.
  Node selection: treat as discrete (use log-sum-exp trick for entropy).
  Discount: continuous Gaussian, reparameterized.

__init__(policy, cfg, logger, device)
train(train_graphs) → full SAC training loop
update(batch) → one SAC gradient update step (actor + critics + targets)
```

---

## PHASE 6 — Experiment Scripts

### Prompt 6.1 — Rev-GNN-IM-RL experiment
```
Read CLAUDE.md, all src/ files, and configs/experiments/rev_gnn_im_rl.yaml.
Create experiments/run_rev_gnn_im_rl.py that:

1. Loads config and sets seed
2. Generates training graphs (forest fire, 20 graphs, n=200-500)
3. Phase 1: ImitationTrainer on training graphs (100 epochs)
4. Phase 2: REINFORCETrainer fine-tuning (100 epochs)
5. Evaluation: run on test graphs + all 5 real networks
6. Compare revenue against all 4 baselines
7. Save checkpoint to results/checkpoints/rev_gnn_im_rl.pt
8. Log all metrics via ExperimentLogger

This script must be thin (< 80 lines) — all logic stays in src/.
Run: python experiments/run_rev_gnn_im_rl.py --config configs/experiments/rev_gnn_im_rl.yaml
```

### Prompt 6.2 — Rev-GAIL-RL-Rich experiment
```
Read CLAUDE.md and experiments/run_rev_gnn_im_rl.py.
Create experiments/run_rev_gail_rl_rich.py following the same structure but:
- Phase 1: GAILTrainer instead of ImitationTrainer
- Uses 20-dim Rich features (all dims active)
- Loads config from configs/experiments/rev_gail_rl_rich.yaml
Run: python experiments/run_rev_gail_rl_rich.py --config configs/experiments/rev_gail_rl_rich.yaml
```

### Prompt 6.3 — Rev-PPO and Rev-SAC experiments
```
Read CLAUDE.md and experiments/run_rev_gnn_im_rl.py.
Create experiments/run_rev_ppo.py and experiments/run_rev_sac.py
following the same structure but using PPOTrainer and SACTrainer respectively.
No imitation pretraining for these — train from scratch.
```

### Prompt 6.4 — Idea 2: NPV / time-discounted revenue
```
Read CLAUDE.md, src/env/revenue_env.py (NPV mode), and experiments/run_rev_gnn_im_rl.py.
Create experiments/run_rev_npv.py that:

1. Uses configs/experiments/rev_npv.yaml (gamma=0.9 as default)
2. Trains Rev-GNN-IM-RL with NPV reward (gamma < 1)
3. Evaluates on BOTH flat revenue AND NPV revenue metrics
4. Sweeps gamma ∈ {0.7, 0.8, 0.9, 0.95, 1.0} and saves all results
5. Generates: results/figures/npv_gamma_sweep.png

The key result should show: lower gamma → revenue arrives earlier → better NPV,
at the cost of slightly lower flat revenue total.
```

---

## PHASE 7 — Ablations

### Prompt 7.1 — Ablation: joint vs. decoupled
```
Read CLAUDE.md and experiments/run_rev_gnn_im_rl.py.
Create experiments/ablation/ablation_discount_head.py that compares:

1. Full model (joint): Rev-GNN-IM-RL with both scoring + pricing heads
2. Decoupled-GNN: GNN selects nodes (scoring head only), then applies µ-discount rule
3. Decoupled-Greedy: GNN selects nodes, then applies greedy-discount rule
4. Oracle: GNN selects nodes, then uses true valuation for pricing (upper bound)

Plot revenue curves for all 4 on forest fire network (monotone + non-monotone).
This is the KEY ablation proving the joint policy helps.
Save to results/figures/ablation_joint_vs_decoupled.png
```

### Prompt 7.2 — Ablation: encoder architecture
```
Read CLAUDE.md, src/models/encoders/, and experiments/run_rev_gnn_im_rl.py.
Create experiments/ablation/ablation_encoder.py that trains and compares:

1. Rev-GNN-IM-RL with GraphSAGE encoder
2. Rev-GNN-IM-RL with Graph Transformer encoder

Same training procedure for both. Compare revenue + inference time.
Save results to results/logs/ablation_encoder.csv and
results/figures/ablation_encoder.png
```

---

## PHASE 8 — Visualization + Paper

### Prompt 8.1 — Generate all paper figures
```
Read CLAUDE.md, results/logs/, and src/utils/visualization.py (implement if missing).
Implement src/utils/visualization.py with:

1. `plot_revenue_curves(results_dict, network_name, save_path)`
   Revenue vs number of buyers accepted (matching Babaei et al. Fig 3 style)
   Lines: IE-Strategy, µ-Discount, Greedy-Discount, Rev-GNN-IM-RL, Rev-GAIL, Rev-PPO, Rev-SAC

2. `plot_model_comparison_table(results_df, save_path)`
   Bar chart comparing all methods on all networks

3. `plot_npv_gamma_sweep(sweep_results, save_path)`
   Revenue vs gamma for Idea 2

4. `plot_discount_distribution(discount_history, save_path)`
   Distribution of discounts assigned by learned vs. hand-crafted methods

All plots: plt.style.use("seaborn-v0_8-paper"), dpi=300, save as PNG.
Then create experiments/generate_figures.py that calls all 4 functions.
```

### Prompt 8.2 — Write methodology section
```
Read CLAUDE.md, paper/main.tex, paper/sections/methodology.tex,
src/models/policies/joint_policy.py, src/models/encoders/graphsage.py,
src/training/reinforce_trainer.py, and src/env/revenue_env.py.

Write a complete methodology section in paper/sections/methodology.tex covering:

1. Problem formulation (MDP: state, action space, reward, transition)
   - Reference the Babaei et al. influence model (cite Babaei2013revmax)
   - Write out the Rayleigh valuation function as a LaTeX equation

2. GNN Encoder (cite our WSDM paper Seyedin2027wsdm):
   - GraphSAGE backbone equations (Eq. 12-13 from WSDM)
   - Extended 20-dim feature vector (explain the 4 new pricing dims)

3. Joint Policy:
   - Scoring head (node selection, same as WSDM)
   - Pricing head (new, discount ∈ [0,1])
   - Joint action selection algorithm

4. Training procedure:
   - Phase 1: GAIL warmstart from greedy-discount expert
   - Phase 2: REINFORCE with revenue reward

5. Idea 2 (NPV objective): one paragraph, one equation R = sum gamma^t * r_t

Use \newcommand macros from main.tex for all numbers.
Use \cref{} for all cross-references.
```

---

## PHASE 5B — Sequence Model Training

### Prompt 5B.1 — REINFORCE trainer for sequential policy
```
Read CLAUDE.md, src/models/policies/sequential_joint_policy.py,
src/models/encoders/sequence_models.py, and src/training/reinforce_trainer.py.

The SequentialJointPolicy has internal episode state (LSTM hidden / token history).
Extend or subclass REINFORCETrainer as SequentialREINFORCETrainer that:

1. Calls policy.reset_episode(device) at the start of each rollout episode
2. After each env.step(), calls policy.update_sequence_state(discount, accepted, revenue)
3. Uses policy.select_and_price() for action selection (same interface as before)
4. Everything else (loss, gradient, logging) is identical to REINFORCETrainer

The LSTM hidden state must NOT be detached between steps within an episode
(so gradients flow through the full sequence).
It MUST be detached between episodes (reset_episode sets a new zero hidden state).

Add to src/training/reinforce_trainer.py as a subclass or standalone class.
```

### Prompt 5B.2 — Rev-GNN-LSTM experiment
```
Read CLAUDE.md, src/models/encoders/sequence_models.py,
src/models/policies/sequential_joint_policy.py,
src/training/reinforce_trainer.py (SequentialREINFORCETrainer),
and experiments/run_rev_gnn_im_rl.py.

Create experiments/run_rev_gnn_lstm.py that:

1. Builds: GraphSAGEEncoder(in_dim=20) + EpisodeLSTM(graph_dim=64, lstm_hidden=64)
           + SequentialJointPolicy(encoder, lstm)
2. Phase 1: ImitationTrainer (pretrain scoring head, same as Rev-GNN-IM-RL)
3. Phase 2: SequentialREINFORCETrainer (fine-tune with LSTM memory active)
4. Evaluates on all test networks and compares against Rev-GNN-IM-RL
5. Key metric to check: does the LSTM help on networks where early rejections
   should signal a need for higher discounts later?

Config: configs/experiments/rev_gnn_lstm.yaml (create it too, extending base_config)
The only new hyperparameters: lstm_hidden=64, lstm_n_layers=1
```

### Prompt 5B.3 — Rev-GNN-Transformer experiment
```
Read CLAUDE.md, src/models/encoders/sequence_models.py,
src/models/policies/sequential_joint_policy.py,
and experiments/run_rev_gnn_lstm.py.

Create experiments/run_rev_gnn_transformer_seq.py that uses
EpisodeTransformer instead of EpisodeLSTM:

1. Builds: GraphSAGEEncoder(in_dim=20) + EpisodeTransformer(graph_dim=64, n_heads=4, n_layers=2)
           + SequentialJointPolicy(encoder, transformer)
2. Phase 1: Imitation pretraining (same as LSTM version)
3. Phase 2: SequentialREINFORCETrainer

Key difference from LSTM:
  - Transformer attends over ALL past steps (not just the last hidden state)
  - May need more memory for long episodes (n > 500 nodes)
  - Use cfg.training.max_seq_len to cap history if needed

Config: configs/experiments/rev_gnn_transformer_seq.yaml
New hyperparameters: n_heads=4, n_layers=2, ff_dim=128, dropout=0.1
```

### Prompt 5B.4 — Rev-GAIL-LSTM (strongest model)
```
Read CLAUDE.md, experiments/run_rev_gnn_lstm.py,
src/training/gail_trainer.py, and src/models/policies/sequential_joint_policy.py.

Create experiments/run_rev_gail_lstm.py:
  Phase 1: GAILTrainer (discriminator distinguishes policy from greedy-discount expert)
  Phase 2: SequentialREINFORCETrainer (LSTM memory active)

This is your STRONGEST model — GAIL initialization + LSTM temporal memory.
Expected to be the top performer in the comparison table.

The GAIL discriminator should receive:
  D(G, S_t, v, context) where context = LSTM hidden state at step t
  This means the discriminator also sees the history, not just the current choice.

Config: configs/experiments/rev_gail_lstm.yaml
```

### Prompt 5B.5 — Ablation: does the sequence model help?
```
Read CLAUDE.md and results from experiments 5B.2-5B.4.
Create experiments/ablation/ablation_sequence_model.py that compares:

1. Rev-GNN-IM-RL (no memory baseline)
2. Rev-GNN-LSTM (LSTM memory)
3. Rev-GNN-Transformer-Seq (Transformer memory)
4. Rev-GAIL-LSTM (best model)

The KEY experiment: on a network where early buyers REJECT offers,
does the model with memory learn to offer higher discounts later?

Plot: discount offered at each step t for a single episode
  x-axis: step t (offer sequence)
  y-axis: discount offered
  lines: memory vs no-memory models

This directly shows the TEMPORAL LEARNING the sequence model provides.
Save to results/figures/ablation_sequence_discount_trajectory.png
```

If a training run crashes:
```
Read CLAUDE.md and the error traceback. Check:
1. Is the feature computation returning NaN? → Check influence=0 edge cases in features.py
2. Is the reward always 0? → Check valuation > 0 before offering (discount too high)
3. Is loss NaN? → Add gradient clipping (already in cfg.training.grad_clip=1.0)
4. CUDA OOM? → Reduce batch_size in config

Paste the full traceback and say: "Fix this error. Read CLAUDE.md first."
```

If baseline results don't match Babaei et al.:
```
Read CLAUDE.md and src/evaluation/baselines.py. The expected ordering is:
greedy_discount > sigma_discount > mu_discount > ie_strategy
(~21% improvement for greedy_discount over ie_strategy in monotone model)
Check: are link weights sampled correctly from Uniform(0, 2)?
Is the Rayleigh function using y=2x (not just x)?
```
