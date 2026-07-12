# results/checkpoints/README.md — Checkpoint catalog

`.pt` files are **not** stored in git (gitignored).
Download released checkpoints from GitHub Release **v1.0-aaai2027**:

```bash
bash scripts/download_checkpoints.sh
```

> **⚠ IMPORTANT — 2026-07-12 retrain incident:**
> The original published Idea-3 LSTM budget checkpoint (`rev_gnn_lstm_budget.pt`,
> SHA256=`4b966e17...`) was **overwritten** on 2026-07-12 by a Welford-fix retrain
> (PID 61149) before a backup was made. That hash matches no surviving file.
> It produced: FF k=3=327.9, Rice k=10=68.6 (published in paper).
> The retrained replacement (SHA256=`a78289…`) is evaluated in
> `results/logs/lstm_idea3_retrained_sweep.json` (pending Item 3 decision).
> See CLAUDE.md Session State for full provenance.

---

## Checkpoint file → Experiment → Expected metric

### Idea 1 — Rev-GNN-LSTM (FROZEN, DO NOT RETRAIN)

| File | Experiment | Expected metric | SHA256 (prefix) |
|------|-----------|-----------------|-----------------|
| `rev_gnn_lstm.pt` | Idea 1 main — Rev-GNN-LSTM | Rev = 462.6 ± ? (FF n=1000) | `8fbc4648` |
| `rev_gnn_im_rl.pt` | Ablation: IM+RL without LSTM | Rev ≈ 445 | `a8232ce2` |
| `rev_gail_lstm.pt` | Ablation: GAIL + LSTM | GAIL comparison | `f77393ab` |
| `rev_gail_rl_rich.pt` | Ablation: GAIL + RL rich features | GAIL comparison | `8b64e55b` |

### Idea 2 — Time-critical (FROZEN)

| File | Experiment | Expected metric | SHA256 (prefix) |
|------|-----------|-----------------|-----------------|
| `rev_gnn_lstm_tc.pt` | Idea 2 — Time-critical LSTM | TC-Rev > no-TC baseline | `20901c29` |

### Idea 3 — Budget-constrained LSTM (Idea-3)

| File | Status | Notes | SHA256 (prefix) |
|------|--------|-------|-----------------|
| `rev_gnn_lstm_budget.pt` | **RETRAINED** (Welford-fixed) | 2026-07-12, 200 eps from `rev_gnn_lstm.pt` base, best at ep 10. Decision rule result: see `lstm_idea3_retrained_sweep.json`. | `a7828957` |
| `rev_gnn_lstm_budget_v1.pt` | Archive — Jul-4 clean original | First budget training attempt. NOT the published model (fails fingerprint). | `1499ddd3` |
| `rev_gnn_lstm_budget_v2.pt` | Archive — Jul-6 | Second budget training attempt. | `23d11e1a` |
| `rev_gnn_lstm_budget_v1_welford_bug.pt` | **DELETE CANDIDATE** — mislabeled | Created 2026-07-12 14:58, mid-retrain (ep 10 snapshot). Same weights as current `rev_gnn_lstm_budget.pt`. Filename misleading. | `a7828957` |

> **PUBLISHED CHECKPOINT LOST:** Original `rev_gnn_lstm_budget.pt` that
> produced paper Table (FF k=3=327.9, Rice k=10=68.6) had
> SHA256=`4b966e17b435fcd6de4fe60909393c64215c499d62b8ec3dde23a52331241e8e`
> This file was overwritten and is NOT recoverable.
> Fingerprint script confirmed: no surviving checkpoint matches `4b966e17`.

### Idea 3 — Transformer (Gate A PASSED)

| File | Status | Notes | SHA256 (prefix) |
|------|--------|-------|-----------------|
| `rev_gnn_transformer.pt` | Gate A PASSED | 463.84±5.26 on FF n=1000 (vs LSTM 462.6). Idea-1 baseline transformer. | `c24215b8` |
| `rev_gnn_transformer_budget.pt` | **Budget-aware Transformer** | 200 epochs budget training. Gate B v2 run-2 in progress. | `2489593a` |
| `rev_gnn_transformer_best.pt` | Archive — training best snapshot | Best epoch during transformer training. | `330792e1` |
| `rev_gnn_transformer_p1.pt` | Archive — Phase 1 snapshot | Pre-budget-training phase 1 checkpoint. | `8e5fc475` |

---

## SHA256 checksums (full, for verification)

Generated 2026-07-12 after retrain completion:

```
8fbc4648ea4eda4e9a6a604041ee5aa780a63ce912f9eda171626cb122e1a7b6  rev_gnn_lstm.pt
a7828957060233a298bdcfce3ea93341ba65b47898f7244e6ec2975fb4bda270  rev_gnn_lstm_budget.pt           ← RETRAINED (Welford-fixed, 2026-07-12)
1499ddd3694af7a19966eb3b628293c4900a1f0dc746b4f299723f3e29f02d6e  rev_gnn_lstm_budget_v1.pt        ← Jul-4 original (archived)
23d11e1a0faf48e53420e77e07311e4388c8696c0429b15f152d4c5db3800b4a  rev_gnn_lstm_budget_v2.pt        ← Jul-6 retrain (archived)
a7828957060233a298bdcfce3ea93341ba65b47898f7244e6ec2975fb4bda270  rev_gnn_lstm_budget_v1_welford_bug.pt  ← DELETE (mislabeled ep-10 snapshot)
20901c29a714628c4da0ffbf507e88641e7958134c21fdc920aa82af5f13ea0c  rev_gnn_lstm_tc.pt
a8232ce2998e9aed9a27b04feddf6d2104c01ca6ef8285a374bf3ed77cd600df  rev_gnn_im_rl.pt
f77393ab7e1ac097fc1dfb539a6f5fb4e967650cdee26627310f89c8511f9343  rev_gail_lstm.pt
8b64e55b62a0bd0a4b45147726e104269abe7aef946854980492efe35df659b5  rev_gail_rl_rich.pt
c24215b838ee3877b6ad1908d845c803b502930e749add70bf4e50143b09e284  rev_gnn_transformer.pt
2489593a2ae09eddb3bd53e46bc9b5a3af6bcd4184c2e3ee2b47b3394dfd3735  rev_gnn_transformer_budget.pt    ← Budget-aware TFM, Gate B v2
330792e1ba7d1e2a25933de9a7cb9745a639ef529b43b5b8415f0a3e5048e3bc  rev_gnn_transformer_best.pt
8e5fc47541f8659af810746dc39db4f9005eb2fbe502d60b852d458560a872ec  rev_gnn_transformer_p1.pt
```

> PREVIOUSLY PUBLISHED (lost): `4b966e17b435fcd6de4fe60909393c64215c499d62b8ec3dde23a52331241e8e  rev_gnn_lstm_budget.pt`

Verify surviving files: `shasum -a 256 results/checkpoints/*.pt`

---

## Architecture

### LSTM models (Idea 1, 2, 3)
- GraphSAGEEncoder: `input_dim=20, hidden=64, n_layers=2`
- EpisodeLSTM: `hidden=64, n_layers=1`
- SequentialJointPolicy: `gnn_dim=64, context_dim=64`
- Budget-aware: `input_dim=21` (extra budget_remaining feature)

### Budget-aware input extension
All Idea-3 budget checkpoints use 21-dim input (vs 20-dim for Idea-1):
- Dim 21 = normalized budget remaining = `budget_remaining / B`

### Transformer models (Idea 3, Gate A)
- GraphSAGEEncoder: `input_dim=21, hidden=64, n_layers=2`
- EpisodeTransformerSliding: `window=256`
- TransformerJointPolicy: `gnn_dim=64, context_dim=transformer_dim`
- Total params: 106,627 (vs LSTM 64,291)
