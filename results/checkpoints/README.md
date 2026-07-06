# results/checkpoints/README.md — Checkpoint catalog

`.pt` files are **not** stored in git (gitignored).  
Download them from the GitHub Release **v1.0-aaai2027**:

```bash
bash scripts/download_checkpoints.sh
```

---

## Checkpoint file → Experiment → Expected metric

| File | Experiment | Producing command | Expected metric |
|---|---|---|---|
| `rev_gnn_lstm.pt` | **Idea 1 main** — Rev-GNN-LSTM (FF n=1000) | `experiments/run_rev_gnn_lstm.py` | Rev ≈ 462 (FF n=1000, Greedy-D baseline 460) |
| `rev_gnn_lstm_budget.pt` | **Idea 3** — Budget-aware LSTM (B=3.0, k=10) | `experiments/run_budget_training.py` | Rev ≈ 347 (k=10, B=3.0) |
| `rev_gnn_lstm_tc.pt` | **Idea 2** — Time-critical LSTM | `experiments/run_tc_lstm_training.py` | TC-Rev > no-TC baseline |
| `rev_gnn_im_rl.pt` | Ablation: IM+RL without LSTM | `experiments/run_rev_gnn_im_rl.py` | Rev ≈ 445 (below LSTM) |
| `rev_gail_lstm.pt` | Ablation: GAIL + LSTM | `experiments/run_rev_gail_lstm.py` | GAIL comparison |
| `rev_gail_rl_rich.pt` | Ablation: GAIL + RL rich features | `experiments/run_rev_gail_rl_rich.py` | GAIL comparison |

---

## SHA256 checksums (full, for verification)

```
8fbc4648ea4eda4e9a6a604041ee5aa780a63ce912f9eda171626cb122e1a7b6  rev_gnn_lstm.pt
4b966e17b435fcd6de4fe60909393c64215c499d62b8ec3dde23a52331241e8e  rev_gnn_lstm_budget.pt
20901c29a714628c4da0ffbf507e88641e7958134c21fdc920aa82af5f13ea0c  rev_gnn_lstm_tc.pt
a8232ce2998e9aed9a27b04feddf6d2104c01ca6ef8285a374bf3ed77cd600df  rev_gnn_im_rl.pt
f77393ab7e1ac097fc1dfb539a6f5fb4e967650cdee26627310f89c8511f9343  rev_gail_lstm.pt
8b64e55b62a0bd0a4b45147726e104269abe7aef946854980492efe35df659b5  rev_gail_rl_rich.pt
```

Verify with: `shasum -a 256 results/checkpoints/*.pt`

---

## Note on in-progress checkpoints

These files exist only on the dev machine and are **not** released:

- `rev_gnn_lstm_budget_v2.pt` — Budget retrain v2 (in progress, PID 89062)
- `rev_gnn_transformer.pt` — Rev-GNN-Transformer (not yet trained; see `experiments/run_rev_gnn_transformer.py`)

---

## Architecture (all 528K `.pt` files)

- GraphSAGEEncoder: `input_dim=20, hidden=64, n_layers=2`
- EpisodeLSTM: `hidden=64, n_layers=1`
- SequentialJointPolicy: `gnn_dim=64, context_dim=64`
- Scorer: `Linear(128,128,ReLU,Linear(128,1))`
- Pricing: `Linear(128,2) → Beta(α,β)`

200K files (rev_gnn_im_rl, rev_gail_rl_rich) omit the LSTM.
