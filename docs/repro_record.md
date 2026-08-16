# Reproducibility Record — revmax-aaai2027
Generated: 2026-08-16

## 1. Hyperparameters

### Phase 1 — Imitation Learning
| Param | Value |
|---|---|
| Optimizer | Adam |
| LR | 1e-3 |
| Batch size | 32 |
| Epochs | 200 |
| Episodes per epoch | 50 |
| Entropy coeff | 0.01 |
| GNN hidden dim | 64 |
| GNN layers | 2 |
| GNN in_dim | 21 |
| LSTM hidden | 64 |
| LSTM input | 3 (disc, acc, rev) |
| n_tiers | 5 (1.0, 0.8, 0.5, 0.2, 0.0) |
| Expert: Greedy-Discount | – |

### Phase 2 — REINFORCE
| Param | Value |
|---|---|
| Optimizer | Adam |
| LR | 5e-4 |
| Clip gradient | 1.0 |
| Entropy coeff | 0.005 |
| Episodes | 100 |
| Baseline | exponential moving average |
| Reward | total_revenue per episode |

### Cal-DP Calibration
| Param | Value |
|---|---|
| n_sims (per cell) | 5 |
| Total observed offers | 5×5 = 25k |
| Degree classes | 5 |
| Influence buckets | 5 |
| Budget buckets (n_buckets) | 20 |
| Tiers | (1.0, 0.8, 0.5, 0.2, 0.0) |
| Delta (budget step) | B_max / n_buckets |
| Interpolation | nearest-neighbor for empty cells |

### BudgetRevenueEnv
| Param | Value |
|---|---|
| production_cost c | 0.3 |
| B_0 | k × 0.3 |
| weight_high | 1.0 (Uniform[0,1]) |
| Acceptance | deterministic (v ≥ p) |
| Valuation | Rayleigh (sigma = weight × b_RAY, b_RAY = 1.0) |

## 2. Wall-Clock Estimates (GPU box, A100)

| Task | Time |
|---|---|
| Phase-1 training (FF_1000, 200 ep) | ~45 min |
| Phase-2 REINFORCE (100 ep) | ~20 min |
| Cal-DP calibration (FF_1000, 25k) | ~8 min |
| Full 10-seed budget sweep (1 network) | ~60–120 min |
| Misspecification sweep (FF-1000, k=3) | ~90 min |

## 3. Checkpoint SHAs

| Checkpoint | Path | SHA8 | Role |
|---|---|---|---|
| arm_b (FF+BA) | results/checkpoints/rev_gnn_lstm_densemix.pt | 0b549f93 | Main policy |
| arm_b ep80 | results/checkpoints/rev_gnn_lstm_densemix.pt@ep80 | 00368482 | Intermediate |
| unified | results/checkpoints/rev_gnn_lstm_unified.pt | 57c23076 | Budget sweep OURS k<20 |
| largek | results/checkpoints/rev_gnn_lstm_largek.pt | 3033620a | Budget sweep OURS k≥20 |
| lstm_v1 | results/checkpoints/rev_gnn_lstm_budget.pt | a7828957 | Budget-trained v1 |
| C1 (unconstrained) | results/checkpoints/rev_gnn_lstm.pt | 8fbc4648 | C1 paper eval |
| c1_ffba_2to1 | results/checkpoints/c1_ffba_2to1_final.pt | fbea89ca | FF:BA=2:1 |
| c1_ffba_50_50 | results/checkpoints/c1_ffba_50_50_final.pt | a190f4e3 | FF:BA=1:1 |

**Training seeds**: n=1 for every listed checkpoint.

## 4. Seed Lists

| Block | Seeds |
|---|---|
| A (budget sweep 10-seed) | 0,1,2,3,4,5,6,7,8,9 |
| B (unconstrained ablation) | 0,1,2,3,4,5,6,7,8,9 |
| C (controls) | 0,1,2,3,4,5,6,7,8,9 |
| D (off-graph Cal-DP) | 0,1,2,3,4,5,6,7,8,9 |
| E (misspecification) | 0,1,2,3,4,5,6,7,8,9 |
| Cal-DP calibration | 0 (single calibration pass per graph) |

## 5. Section Headings (script → output mapping)

- §A Budget Sweep → `experiments/budget_sweep_10seed.py` → `results/logs/budget_10s_{NET}.json`
- §A Merge+Tests → `experiments/merge_budget_10seed.py` → `results/logs/budget_sweep_10seed.json`
- §B Ordering+P1 → `experiments/ablation_unc_10seed.py` → `results/logs/ablation_unc_10seed.json`
- §C Controls → `experiments/controls_10seed.py` → `results/logs/ctrl_{NET}.json`
- §D Off-graph → `experiments/caldp_offgraph.py` → `results/logs/caldp_offgraph_{NET}.json`
- §D Adaptation → `experiments/adapt_policy.py` → `results/checkpoints/adapted_{NET}_{sha8}.pt`
- §E Misspec → `experiments/misspec_eval.py` → `results/logs/misspec_eval.json`

## 6. D1: Policy Adaptation (fine-tuning per network)

Script: `experiments/adapt_policy.py`
Recipe: resume Phase-2 REINFORCE from arm_b (sha 0b549f93), 25k on-graph episodes
        (500 episodes × 50 per epoch), same LR/clip as Phase 2.
        One adapted checkpoint saved per network; sha recorded in output JSON.
