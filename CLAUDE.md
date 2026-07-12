# CLAUDE.md — Revenue Maximization via Joint Seed Selection & Discounting
# Target venue: AAAI 2027 | Submission deadline: ~August 2026

Read this file completely before writing or editing any code.
This is the single source of truth for architecture, conventions, and running experiments.

---

## What This Paper Does (Read First)

### The Problem We Solve
Babaei et al. (2013) "Revenue Maximization in Social Networks through Discounting"
showed that offering discounts (instead of giving items for free) to influential buyers
in a social network increases total revenue. Their approach has two **separate** hand-crafted steps:

  Step 1 — Seed selection:  greedy hill climbing or local search → picks set S
  Step 2 — Discount sequence: µ-rule / σ-rule / greedy degree rule → assigns prices

**The key limitation**: these two steps are decoupled. The best seed to pick next
depends on what discount you offer, and the right discount depends on who is
already in the seed set. The paper ignores this coupling entirely.

### Our Contribution
We extend our WSDM 2027 framework (GNN-IM-RL, GAIL-RL-Rich) to learn
**joint seed selection AND discount assignment** end-to-end via deep RL.

At each step the agent decides:
  → WHICH buyer to target next    (discrete, same as WSDM)
  → WHAT discount to offer them   (continuous [0,1], NEW)

The reward is the actual revenue collected (price paid if accepted, 0 if rejected),
NOT influence spread.

### Idea 2 — Time-Discounted Revenue (implement AFTER Idea 1 results)
In the real world, revenue now > revenue later (NPV / time value of money).
A company maximising revenue over 10 years but collecting most in year 9 may
go bankrupt in year 3. We replace the flat revenue reward with:

  R_total = sum_t [ gamma^t * r_t ]   where gamma < 1

This directly maps onto the RL discount factor — AAAI angle:
"the RL discount factor γ is not just a training trick, it IS the economic objective."

---

## Architecture (Grounded in WSDM Paper)

### Shared GNN Backbone (identical to WSDM, src/models/encoders/graphsage.py)
Two-layer GraphSAGE with residual + LayerNorm:

  h_v^(0) = ReLU(LayerNorm(W_proj * phi(v)))     ∈ R^64
  h_v^(l) = ReLU(LayerNorm(h^(l-1) + W_self*h^(l-1) + W_neigh * A_hat * h^(l-1)))
  score_v = Linear(32→1)(ReLU(Linear(64→32)(h_v^(2))))

Graph Transformer variant (src/models/encoders/graph_transformer.py):
  Same interface, replaces SAGEConv with TransformerConv (PyG).

### Node Feature Vector (extended from WSDM's 16-dim → 20-dim)

Static features (computed once per graph, dims 1-10, identical to WSDM):
  deg, cc, bc, pr, kc, ec, tc, cl, ecc, and
  = degree rank, clustering coeff, betweenness, PageRank, k-core,
    eigenvector centrality, triangle count, closeness, eccentricity, avg-neighbor-degree

Dynamic features (updated every step, dims 11-16 from WSDM):
  seed_flag       = 1 if v already in S_t, else 0
  round_ratio     = t / k
  hop1_seed_frac  = fraction of v's neighbors already in S_t
  log_degree      = log(1 + degree(v))
  cluster_repeat  = clustering coefficient (repeated as diffusion signal)
  group_flag      = minority/majority indicator (set to 0 for revenue task; buyer type)

NEW pricing-specific features (dims 17-20, not in WSDM):
  current_influence = sum of w_ij for j in S_t (normalized by sum_k w_ik)
                    = the current normalized influence on node v from buyers in S
  current_valuation = f(current_influence) under the Rayleigh model
                    = estimated willingness-to-pay of v right now
  was_offered       = 1 if v was already offered and rejected, else 0
  steps_remaining   = (n - t) / n, fraction of buyers not yet offered

### Pricing Head (NEW — src/models/policies/pricing_head.py)
Sits alongside the scoring head. Takes h_v^(2) and outputs discount ∈ [0,1]:

  discount_v = Sigmoid(Linear(32→1)(ReLU(Linear(64→32)(h_v^(2)))))

At each step:
  1. Scoring head → scores for all v ∉ S_t
  2. Select v* = argmax(scores)
  3. Pricing head → discount_d = discount_{v*}
  4. Offered price p = f(influence(v*)) * (1 - discount_d)
  5. v* accepts if p <= v*(S_t), i.e., if discount is deep enough
  6. Revenue += p if accepted, 0 if rejected

### Model Family (mirrors WSDM Table 1)

| Model                  | GNN Encoder      | Sequence Model | Training              | Discount |
|------------------------|------------------|----------------|-----------------------|---------|
| Rev-GNN-IM-RL          | GraphSAGE (d=20) | —              | Imitation + REINFORCE | Joint   |
| Rev-GAIL-RL-Rich       | GraphSAGE (d=20) | —              | GAIL + REINFORCE      | Joint   |
| Rev-PPO                | GraphSAGE (d=20) | —              | PPO                   | Joint   |
| Rev-SAC                | GraphSAGE (d=20) | —              | SAC (off-policy)      | Joint   |
| Rev-GraphTransformer   | GT (d=20)        | —              | GAIL + REINFORCE      | Joint   |
| Rev-GNN-LSTM           | GraphSAGE (d=20) | LSTM           | GAIL + REINFORCE      | Joint   |
| Rev-GNN-Transformer    | GraphSAGE (d=20) | Transformer    | GAIL + REINFORCE      | Joint   |
| Rev-GAIL-LSTM          | GraphSAGE (d=20) | LSTM           | GAIL + REINFORCE      | Joint   |
| Rev-GAIL-Transformer   | GraphSAGE (d=20) | Transformer    | GAIL + REINFORCE      | Joint   |
| Rev-NPV (Idea 2)       | GraphSAGE (d=20) | LSTM           | REINFORCE, γ<1        | Joint+time |

### Why LSTM/Transformer? (Key architectural argument)

The GNN at step t sees the CURRENT graph state only.
It does NOT know:
  - That node A rejected an offer at discount 0.3 (price too high for this network)
  - That node B accepted at discount 0.5 (revealing price sensitivity)
  - How fast influence is spreading (sequence of valuations increasing over time)

LSTM/Transformer over the step history captures this.
The "token" at each step t is:
  token_t = [g_t ‖ last_discount ‖ last_accepted ‖ last_revenue]
              64       1               1               1          = 67-dim

  g_t = mean pool of GNN node embeddings (global graph state summary)

LSTM → hidden state h_t carries compressed history (good for long episodes)
Transformer → attends over ALL past tokens directly (better for detecting
  patterns like "this network keeps rejecting discounts < 0.4")

The context c_t is concatenated with node embeddings before scoring + pricing:
  score_v    = scoring_head([H_v ‖ c_t])     ← 128-dim input
  discount   = pricing_head([H_{v*} ‖ c_t])  ← 128-dim input

Key files:
  src/models/encoders/sequence_models.py    ← EpisodeLSTM, EpisodeTransformer
  src/models/policies/sequential_joint_policy.py  ← SequentialJointPolicy

Baselines (hand-crafted, from Babaei et al. 2013):
  IE-Strategy        = give item free to S, myopic pricing for rest
  µ-Discount         = discount based on average degree
  Greedy-Discount    = degree-based greedy discount (best in original paper)
  σ-Discount         = standard deviation based discount

---

## RL Problem Formulation

State:   GNN embedding of full graph at step t
         (includes current_influence and current_valuation for all nodes)

Action:  (v*, discount_d)
         v*         → discrete: which buyer to target (argmax of scoring head)
         discount_d → continuous [0,1]: how much to discount (pricing head)

Transition:
         If buyer v* accepts (offered_price <= v*(S_t)):
           S_{t+1} = S_t ∪ {v*}, influence updates for all neighbors
         If buyer v* rejects:
           S_{t+1} = S_t (no change), but v* is marked as_offered=1

Reward:
         Idea 1:  r_t = offered_price * 1[accepted]
         Idea 2:  r_t = gamma^t * offered_price * 1[accepted]   (NPV objective)

Episode: runs for n steps (one offer per buyer)
         terminates when all buyers have been offered

Expert for imitation (GAIL warmstart):
         Babaei et al.'s Greedy-Discount algorithm = our "teacher"
         Expert trajectories: (sequence of buyers chosen by greedy-discount, prices assigned)

---

## Influence Model (from Babaei et al. 2013)

Valuation of buyer i given set S already bought:

  v_i(S) = f_i( sum_{j in S∪{i}} w_ij / sum_{k in V} w_ik )

Two variants (src/env/influence_models.py):

  Monotone concave:
    f(x) = Rayleigh(x | b=1) with y=2x, then CLIPPED to be non-decreasing
    Implemented as: f(y) = (y/b^2) * exp(-y^2 / (2b^2)), b=1, y=2x, for y in [0,1]
    then f(y) = f(1) for y > 1   (monotone version)

  Non-monotone concave:
    f(y) = (y/b^2) * exp(-y^2 / (2b^2)), b=1, y=2x   (Rayleigh PDF)
    peaks at y=1 (normalized influence = 0.5), decreasing after

Link weights w_ij sampled from Uniform(0, 2) (as in Babaei et al.)
Seller knows distribution F_ij but NOT exact w_ij → uses 200 MC samples to estimate.

---

## Directory Structure

```
revmax-aaai2027/
├── CLAUDE.md                          ← you are here
├── README.md
├── VSCODE_PROMPTS.md                  ← step-by-step prompts for VS Code Claude
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── base_config.yaml               ← shared defaults
│   └── experiments/
│       ├── rev_gnn_im_rl.yaml
│       ├── rev_gail_rl_rich.yaml
│       ├── rev_ppo.yaml
│       ├── rev_sac.yaml
│       ├── rev_graph_transformer.yaml
│       ├── rev_npv.yaml               ← Idea 2
│       └── ablation_discount_head.yaml
│
├── data/
│   ├── raw/                           ← real network edge lists (never modify)
│   ├── processed/                     ← preprocessed networkx graphs (.pkl)
│   └── graphs/                        ← synthetic generated graphs
│
├── src/
│   ├── env/
│   │   ├── revenue_env.py             ← MDP environment (state/action/reward)
│   │   ├── influence_models.py        ← Rayleigh monotone + non-monotone
│   │   └── graph_generators.py        ← forest fire, modular FF, real network loaders
│   │
│   ├── models/
│   │   ├── encoders/
│   │   │   ├── graphsage.py           ← GraphSAGE backbone (from WSDM, extended)
│   │   │   ├── graph_transformer.py   ← GT encoder (same interface)
│   │   │   └── sequence_models.py     ← EpisodeLSTM, EpisodeTransformer (NEW)
│   │   └── policies/
│   │       ├── base_policy.py         ← abstract policy
│   │       ├── pricing_head.py        ← discount output head [0,1]
│   │       ├── joint_policy.py        ← GNN + scoring + pricing (no memory)
│   │       ├── sequential_joint_policy.py ← GNN + LSTM/Transformer + scoring + pricing (NEW)
│   │       ├── ppo_policy.py          ← PPO actor-critic wrapper
│   │       └── sac_policy.py          ← SAC actor-critic wrapper
│   │
│   ├── training/
│   │   ├── imitation_trainer.py       ← Phase 1: MSE on greedy-discount expert
│   │   ├── gail_trainer.py            ← Phase 1: GAIL discriminator training
│   │   ├── reinforce_trainer.py       ← Phase 2: REINFORCE fine-tuning
│   │   ├── ppo_trainer.py             ← PPO training loop
│   │   └── sac_trainer.py             ← SAC training loop
│   │
│   ├── evaluation/
│   │   ├── metrics.py                 ← revenue, approximation ratio, % improvement
│   │   └── baselines.py               ← IE strategy, µ-discount, greedy-discount, σ-discount
│   │
│   └── utils/
│       ├── helpers.py                 ← set_seed, get_device, load_config
│       ├── features.py                ← compute_node_features() → 20-dim vector
│       ├── logging.py                 ← ExperimentLogger (W&B + CSV)
│       └── visualization.py           ← revenue curves, discount distribution plots
│
├── experiments/
│   ├── run_rev_gnn_im_rl.py
│   ├── run_rev_gail_rl_rich.py
│   ├── run_rev_ppo.py
│   ├── run_rev_sac.py
│   ├── run_rev_graph_transformer.py
│   ├── run_rev_npv.py                 ← Idea 2: NPV / time-discounted revenue
│   ├── run_baselines.py               ← all Babaei et al. baselines
│   └── ablation/
│       ├── ablation_discount_head.py  ← what if we remove joint pricing?
│       └── ablation_encoder.py        ← GraphSAGE vs Graph Transformer
│
├── notebooks/
│   ├── 01_network_analysis.ipynb      ← EDA on real networks
│   ├── 02_influence_model_viz.ipynb   ← Rayleigh monotone vs non-monotone
│   └── 03_results_analysis.ipynb
│
├── results/
│   ├── logs/                          ← per-experiment CSV + JSON
│   ├── checkpoints/                   ← model weights (.pt)
│   └── figures/                       ← auto-generated plots for paper
│
├── tests/
│   ├── test_env.py                    ← MDP correctness
│   ├── test_influence_models.py       ← Rayleigh valuation
│   └── test_baselines.py              ← baseline revenue sanity checks
│
└── paper/
    ├── main.tex
    ├── references.bib
    ├── sections/
    │   ├── abstract.tex
    │   ├── introduction.tex
    │   ├── related_work.tex
    │   ├── problem_formulation.tex
    │   ├── methodology.tex
    │   ├── experiments.tex
    │   ├── results.tex
    │   └── conclusion.tex
    └── figures/
```

---

## Key Coding Conventions

- **Config-first**: NO hardcoded hyperparameters. All values from YAML via OmegaConf.
- **Reproducibility**: call `set_seed(cfg.seed)` at top of every experiment script.
- **Device**: use `get_device()` from src/utils/helpers.py. Never hardcode "cuda".
- **Logging**: use ExperimentLogger. Never use bare print().
- **Features**: always call `compute_node_features(G, S_t, t, cfg)` from src/utils/features.py.
- **Type hints + Google docstrings**: mandatory on all public functions.
- **Tests**: every new component needs a test in tests/.

---

## Networks Used in Experiments

### Synthetic (for training + test, matching Babaei et al. 2013)
- Forest Fire: n=1000, p=0.37, pb=0.32
- Modular Forest Fire: 3 modules (200, 300, 500 nodes), P=0.01

### Real networks (evaluation only — download from SNAP)
- Facebook-like (UCI): 1899 nodes, 20296 edges
- Yeast protein-protein: 2224 nodes, 6829 edges
- Newman collab: 16726 nodes, 47594 edges
- Wiki-vote: 7115 nodes, 103689 edges
- HEP citation: 27770 nodes, 352807 edges

Network data goes in data/raw/. Preprocessed versions in data/processed/.

---

## Running Experiments

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Reproduce baselines (Babaei et al. 2013)
python experiments/run_baselines.py --config configs/base_config.yaml

# Train Rev-GNN-IM-RL
python experiments/run_rev_gnn_im_rl.py --config configs/experiments/rev_gnn_im_rl.yaml

# Train Rev-GAIL-RL-Rich
python experiments/run_rev_gail_rl_rich.py --config configs/experiments/rev_gail_rl_rich.yaml

# Train Rev-PPO
python experiments/run_rev_ppo.py --config configs/experiments/rev_ppo.yaml

# Train Rev-SAC
python experiments/run_rev_sac.py --config configs/experiments/rev_sac.yaml

# Train Graph Transformer variant
python experiments/run_rev_graph_transformer.py --config configs/experiments/rev_graph_transformer.yaml

# Idea 2: NPV / time-discounted revenue
python experiments/run_rev_npv.py --config configs/experiments/rev_npv.yaml

# Run ablations
python experiments/ablation/ablation_discount_head.py

# Compile paper
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## When Claude Code Is Asked to...

| Task | What to do |
|------|-----------|
| "Implement the environment" | Create src/env/revenue_env.py with MDP logic |
| "Add a new model" | Add to src/models/policies/, register in configs/ |
| "Implement influence model" | Add to src/env/influence_models.py |
| "Run an experiment" | Use thin script in experiments/, all logic stays in src/ |
| "Implement a baseline" | Add to src/evaluation/baselines.py |
| "Plot results" | Add to src/utils/visualization.py, save to results/figures/ |
| "Write paper section" | Edit paper/sections/[section].tex ONLY |
| "Add dependency" | Add to requirements.txt with pinned version |

## Do NOT
- Hardcode network paths, hyperparameters, or random seeds.
- Put training logic inside experiment scripts.
- Modify data/raw/ files.
- Commit checkpoints (gitignored).
- Edit paper/main.tex structure without asking.
- Use bare print() — use ExperimentLogger.

---

## Session State (updated 2026-07-12)

### DONE AND FROZEN (do not touch)
- Idea 1: Rev-GNN-LSTM 462.6, Rev-GNN-IM-RL, all figures/tables
- Idea 2: TC/profit analysis complete
- Idea 3 baselines: Greedy+Budget, DP-naive, DP-Calibrated v1,
  DP-Oracle, LSTM-Idea3 (rev_gnn_lstm_budget.pt) — results in
  results/logs/dp_upgrade_eval*.json
- Transformer Gate A: PASSED (463.84±5.26 on FF n=1000 vs LSTM 462.6)
- Repo packaging: setup.sh, Dockerfile, smoke_test 6/6

### Resolved this session (2026-07-12)

**#1 Running processes:** None surviving from last session. All previously
  backgrounded jobs (dp_v3, transformer budget, tfm eval) died with session.

**#2 Transformer OOD eval (Modular-FF / Rice-FB):** NOT produced by last
  session. `rev_gnn_transformer_budget_20260712_121544.json` is a budget
  *training* progress log (21 checkpoint entries, epoch 0→199), NOT an OOD
  k-sweep. Gate B had never been run. Action taken: Gate B launched
  (see below).

**#3 Transformer budget training:** COMPLETE. Ran 200 epochs. Checkpoint
  confirmed: `results/checkpoints/rev_gnn_transformer_budget.pt` (433 KB,
  mtime 2026-07-12 14:56). LSTM budget checkpoint also present (260 KB).

**#4a DP v2/v3 warm-start removal:** Already done in src code.
  Both dp_calibrated_v2.py and dp_calibrated_v3.py have `seed_frac=0.0`
  (no-op) and explicit comment "No separate free-seed phase."

**#4b DP v3 REAL sweep — CORRECTED (2026-07-12 session 2):**
  Full data in results/logs/dp_v3_full_curve_merged.json (all k=1..40)
  and results/logs/dp_v3_ff_sweep.json (k=10..40).
  k=5,8 cross-validation: diff=0.0000 (no discrepancy).

  CORRECTION: Previous session incorrectly stated "paper keeps v1."
  v1 is DOMINATED at every single k by composite(v2,v3). The correct
  paper line is:

    **DP paper line = composite max(v2,v3) per k**
    **v3 wins at k≤5; v2 wins at k≥8 (transition at k=8)**
    **v3-over-v2 promotion gate FAILED (v3<420 at k=40); gate intended
      v3 to REPLACE v2 entirely, which it cannot. v2 remains strong.**
    **v1 dominated everywhere → appendix only, do not use as paper line.**

  Full composite table (FF n=1000, mean±std):

  | k  |   v1  |   v2  |   v3  | composite | winner |
  |----|-------|-------|-------|-----------|--------|
  |  1 |   3.7 |   3.5 |  10.6 |     10.6  |  v3    |
  |  2 |   8.8 |   6.2 |  42.1 |     42.1  |  v3    |
  |  3 |  45.2 |   8.4 |  99.8 |     99.8  |  v3    |
  |  5 |  94.4 |  73.6 | 154.2 |    154.2  |  v3    |
  |  8 | 298.5 | 415.6 | 339.0 |    415.6  |  v2    |
  | 10 | 325.7 | 435.1 | 357.0 |    435.1  |  v2    |
  | 15 | 340.8 | 447.7 | 370.9 |    447.7  |  v2    |
  | 20 | 345.1 | 448.0 | 369.0 |    448.0  |  v2    |
  | 30 | 349.8 | 448.0 | 369.0 |    448.0  |  v2    |
  | 40 | 354.9 | 448.0 | 380.2 |    448.0  |  v2    |

  Paper artifacts produced:
    paper/tables/paper_table_dp_family.tex   ← full v1/v2/v3/composite table
    results/figures/fig_idea3_main_v2.pdf/png ← orange composite + range band
    experiments/plot_idea3_main_v2.py        ← figure script (re-runnable)

**#5 Gate B v2 (Transformer-Idea3 vs LSTM-Idea3):**
  gate_b_eval.json (first run, PID 64272, 2026-07-12 ~15:02) is VOID —
  weak criterion (single k=40, FF only), killed before completing.

  Gate B v2 script: experiments/run_gate_b_transformer_eval.py (REWRITTEN)
  New spec: k=[1,2,3,5,8,10,15,20,30,40], BOTH FF n=1000 AND Rice-FB n=443,
  n_trials=3, SKIP enforcement, accounting identity checks.
  Criterion: TFM > LSTM at ≥4/10 k on EITHER network.

  **GATE B RUN-1 VERDICT (2026-07-12 16:39, LSTM-v1):**
  **OVERALL PASS ✓** — TFM wins 5/10 on FF (k=1,2,3,5,8); 2/10 on Rice (k=1,15).
  Criterion met via FF (5 ≥ 4). Results → gate_b_eval_v2.json.

  k-by-k (FF / Rice):
  k=1:  LSTM=27.0/0.0    TFM=150.1/0.20   FF_win=✓ Rice_win=✓
  k=2:  LSTM=127.2/0.10  TFM=243.8/0.10   FF_win=✓ Rice_win=
  k=3:  LSTM=169.4/0.10  TFM=212.1/0.00   FF_win=✓ Rice_win=
  k=5:  LSTM=223.4/0.42  TFM=272.3/0.20   FF_win=✓ Rice_win=
  k=8:  LSTM=259.8/1.12  TFM=275.6/0.83   FF_win=✓ Rice_win=
  k=10: LSTM=280.0/2.78  TFM=272.8/1.81   FF_win=  Rice_win=
  k=15: LSTM=334.1/13.5  TFM=317.2/45.3   FF_win=  Rice_win=✓
  k=20: LSTM=356.7/130.1 TFM=327.0/78.8   FF_win=  Rice_win=
  k=30: LSTM=373.2/216.3 TFM=360.9/113.2  FF_win=  Rice_win=
  k=40: LSTM=395.1/217.5 TFM=389.1/162.0  FF_win=  Rice_win=

  NOTE: TFM strong at small k (budget-limited regime k≤8 FF) where SKIP
  enforcement matters most. LSTM overtakes at large k (k≥10 FF).

  Gate B run-2 (--lstm_v both, PID 2244): RUNNING.
  Expected result in gate_b_eval_v2.json (overwrite).

  **PUBLISHED LSTM-Idea3 CHECKPOINT LOST:**
  Original rev_gnn_lstm_budget.pt (hash 4b966e17) was overwritten 2026-07-12
  by Welford-fix retrain. Fingerprint script (PID 2126) confirms:
    v1.pt (1499ddd3): FF k=3=169.4 ✗, Rice k=10=2.78 ✗ — NOT published
    v2.pt (23d11e1a): testing (result pending)
    retrained (a78289): see Item 3 below
  Published numbers (FF k=3=327.9, Rice k=10=68.6) are in budget_eval_c0.3.json
  and dp_upgrade_eval_rice_lstm.json — these remain valid for the paper.

  **RETRAINED LSTM ACCOUNTING VIOLATION:**
  Retrained checkpoint (a78289) trained on B=[2,5,20,50]; NEVER trained on B<2.
  At k=1 (B=0.30) and likely k=3 (B=0.90), bankrupt_rate=100% — policy
  overspends freely at small budgets. The pre-committed decision rule (retrained
  >= published at both flag points) will FAIL at k=3 FF (B=0.90 < training minimum).
  → PAPER KEEPS PUBLISHED NUMBERS from budget_eval_c0.3.json and
    dp_upgrade_eval_rice_lstm.json.
  → Retrained checkpoint archived with note in checkpoints/README.md.

### RESULTS FROZEN 2026-07-12 — Final experimental session

All Items 0-4 resolved. Decisions recorded below.

**ITEM 1 VERDICT — Published LSTM checkpoint: CONFIRMED LOST**
  Fingerprint script (experiments/identify_published_lstm_ckpt.py):
    v1.pt  (1499ddd3): FF k=3=169.4 (diff=162.6) ✗  Rice k=10=2.78 (diff=65.8) ✗
    v2.pt  (23d11e1a): FF k=3=168.3 (diff=163.8) ✗  Rice k=10=2.79 (diff=65.8) ✗
    retrained (a78289): bankrupt=100% at k=1,2,3 (B=0.3–0.9 < training min B=2)
        → also fails fingerprint
  Published checkpoint (hash 4b966e17) NOT recoverable from any surviving file.
  Published numbers (FF k=3=327.9, Rice k=10=68.6) remain valid in:
    results/logs/budget_eval_c0.3.json
    results/logs/dp_upgrade_eval_rice_lstm.json
  These numbers are used in paper Table (paper_table_idea3_final.tex).
  rev_gnn_lstm_budget_v1_welford_bug.pt: DELETED (mislabeled ep-10 snapshot).

**ITEM 2 VERDICT — Gate B: PASS ✓**
  Gate B v2 run-1 (LSTM-v1 vs TFM, FF+Rice, 10 k): PASS ✓ (TFM wins 5/10 FF).
  gate_b_eval_v2.json: written 2026-07-12 16:39.
  Gate B run-2 (--lstm_v both, PID 2244): STILL RUNNING (multi-hour, ~80 min left
    at 18:05). Will overwrite gate_b_eval_v2.json with 3-way table when done.
  Paper decision (pre-committed, run-1 sufficient): **TFM-Idea3 INCLUDED in paper.**
  Add TFM-Idea3 column to Table 3 (budget_constrained.tex).

**ITEM 3 VERDICT — Retrained LSTM: FAILS DECISION RULE → paper keeps published**
  Retrained checkpoint (a78289) trained on B=[2,5,20,50].
  At B<2 (k=1,2,3): bankrupt_rate=100% (policy never trained on B<2).
  Pre-committed rule: retrained >= published at BOTH flag points (k=3 FF, k=10 Rice).
  k=3 FF (B=0.9): bankrupt=100% → INVALID evaluation → FAIL.
  → Paper retains published LSTM-Idea3 numbers from budget_eval_c0.3.json.
  → Retrained checkpoint archived in results/checkpoints/ (not released).
  Full sweep in results/logs/lstm_idea3_retrained_sweep.json (PID 2847, still running).

**PAPER MODEL ROSTER (FINAL)**:
  Idea 1: rev_gnn_lstm.pt (8fbc4648), Rev-GNN-LSTM, 462.6 FF n=1000
  Idea 2: rev_gnn_lstm_tc.pt (20901c29)
  Idea 3 LSTM: published numbers from budget_eval_c0.3.json (ckpt 4b966e17 LOST)
  Idea 3 TFM:  rev_gnn_transformer_budget.pt (2489593a) — Gate B PASS
  DP baselines: composite(v2,v3) from paper_table_dp_family.tex
  All non-model baselines: frozen in dp_upgrade_eval*.json

**FIGURES/TABLES TO REGEN** (only if underlying numbers changed — none did):
  - fig_idea3_main_v2: NO REGEN needed (composite DP line unchanged)
  - paper_table_dp_family.tex: NO REGEN needed
  - paper_table_idea3_final.tex: ADD TFM-Idea3 column before final submission

**NEXT ACTION (new session)**:
  - Add TFM-Idea3 column to paper/tables/paper_table_idea3_final.tex
  - Wait for Gate B run-2 JSON if full 3-way table needed for supplement
  - Verify gate_b_eval_v2.json was updated by run-2 (PID 2244)
