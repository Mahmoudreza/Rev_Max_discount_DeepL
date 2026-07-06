# RevMax — Revenue-Maximising Offer Sequencing on Social Networks

**AAAI 2027 submission.** This repository contains the full implementation of
Rev-GNN-LSTM, a graph neural network + episode-LSTM policy that learns to
sequence personalised discount offers across a social influence network to
maximise total revenue. We study three extensions: unconstrained (Idea 1),
time-critical (Idea 2), and budget-constrained (Idea 3) settings.

---

## Quick start

### 1. Clone and set up (Apple Silicon Mac, Python 3.9)

```bash
git clone https://github.com/Mahmoudreza/Rev_Max_discount_DeepL.git
cd Rev_Max_discount_DeepL
bash setup.sh                         # creates venv, installs deps, runs pytest
```

### 2. Smoke test (~5 min, no checkpoint needed)

```bash
source venv/bin/activate
bash scripts/smoke_test.sh
```

Expected output:
```
=== RevMax smoke test ===
[1] Imports...         PASS all imports
[2] Graph + features   PASS graph + features
[3] Greedy baseline    PASS greedy baseline (n=100)
[4] 2-epoch imitation  PASS 2-epoch imitation (loss stable)
[5] pytest             PASS pytest
[6] Figure render      PASS figure render
Smoke test: 6 passed, 0 failed
STATUS: OK — machine is ready for full reproduction.
```

### 3. Download released checkpoints _(coming soon)_

> **Checkpoints are not yet published.**  
> They will be attached to GitHub Release **v0.1-checkpoints** once training
> finishes. Until then the download script prints instructions and exits.

```bash
bash scripts/download_checkpoints.sh  # probes GitHub; falls back with instructions
# see results/checkpoints/README.md for the full catalog + SHA256s
```

### 4. Reproduce all paper results (~4–5 h with released checkpoints)

```bash
SKIP_TRAINING=1 bash scripts/reproduce_all.sh   # uses downloaded checkpoints
#           ^^^^ remove to retrain from scratch (adds ~6–8 h)
```

---

## Docker (Linux/CPU fallback)

Suitable for evaluation + figures. Training is ~3× slower without MPS/CUDA.

```bash
docker build -t revmax .
docker run --rm -v $(pwd)/results:/app/results revmax bash scripts/smoke_test.sh
# With released checkpoints:
docker run --rm -v $(pwd)/results:/app/results revmax \
  SKIP_TRAINING=1 bash scripts/reproduce_all.sh
```

---

## Repository layout

```
revmax-aaai2027/
├── configs/                        # OmegaConf YAML configs (all hyper-params)
│   ├── base_config.yaml            # shared defaults
│   └── experiments/                # per-experiment overrides
├── src/
│   ├── env/                        # Revenue / BudgetRevenue / TC environments
│   ├── models/
│   │   ├── encoders/               # GraphSAGEEncoder, EpisodeLSTM, EpisodeTransformerSliding
│   │   └── policies/               # SequentialJointPolicy, TransformerJointPolicy
│   ├── training/                   # REINFORCETrainer, WelfordNormalizer
│   ├── evaluation/                 # Baselines (Greedy-D, Degree, Babaei 2013)
│   └── utils/                      # ExperimentLogger, features, helpers
├── experiments/                    # Thin scripts (<80 lines) — logic stays in src/
│   ├── run_rev_gnn_lstm.py         # Idea 1 main training (Phase 1 + Phase 2)
│   ├── run_budget_training.py      # Idea 3 budget training
│   ├── run_tc_lstm_training.py     # Idea 2 time-critical training
│   ├── run_rev_gnn_transformer.py  # Rev-GNN-Transformer (comparator)
│   ├── generate_paper_figures.py   # All paper figures
│   └── run_baselines.py            # Table 1 baselines
├── tests/                          # pytest unit + integration tests
├── data/
│   ├── raw/                        # Rice-FB edge list (in git, small)
│   └── graphs/                     # Preprocessed .pkl (gitignored, generated locally)
├── results/
│   ├── checkpoints/                # *.pt (gitignored, download via scripts/)
│   └── figures/                    # Generated PDFs (in git, small)
├── scripts/
│   ├── setup.sh → setup.sh         # (root-level alias)
│   ├── reproduce_all.sh            # Ordered pipeline
│   ├── smoke_test.sh               # 5-min sanity check
│   └── download_checkpoints.sh     # Checkpoint downloader with SHA256 verify
├── setup.sh                        # Environment bootstrap
├── requirements.txt                # Pinned direct dependencies
├── requirements-freeze-raw.txt     # Full transitive freeze (forensic)
└── Dockerfile                      # Linux/CPU evaluation image
```

---

## Reproduce paper results (per-figure mapping)

| Figure / Table | Script | Needs checkpoint |
|---|---|---|
| **Fig 1** Revenue vs k (Forest Fire) | `generate_paper_figures.py` | `rev_gnn_lstm.pt` |
| **Fig 2** Learning curve | `generate_paper_figures.py` | training logs in `results/logs/` |
| **Table 1** Baselines comparison | `run_baselines.py` | none |
| **Table 2** Budget eval (Idea 3) | `run_budget_eval.py` | `rev_gnn_lstm_budget.pt` |
| **Table 3** Time-critical (Idea 2) | `run_tc_eval.py` | `rev_gnn_lstm_tc.pt` |
| **Ablation** GNN type | `ablation_encoder_type.py` | retrains |
| **Ablation** Graph type | `ablation_graph_type.py` | retrains |

All figures output to `results/figures/`.

For the full ordered pipeline with runtime estimates, see `scripts/reproduce_all.sh`.

---

## Hardware notes — Apple Silicon MPS

We developed and ran all experiments on an Apple M-series Mac using PyTorch MPS.

**MPS quirks we encountered:**

1. **Phase 2 REINFORCE must run on CPU.** `torch.no_grad()` inside
   `collect_rollout` permanently corrupts the MPS autograd state, causing
   `loss.backward()` to fail with a silent zeroed gradient. All experiment
   scripts automatically move the policy to CPU before any REINFORCE phase.

2. **MPS fallback env var.** Some PyTorch ops (e.g. `torch.kthvalue`) are not
   yet implemented in MPS. Set:
   ```bash
   export PYTORCH_ENABLE_MPS_FALLBACK=1
   ```
   before running any Phase 1 training script.

3. **MPS vs CPU nondeterminism.** We observed ±1–2 revenue points between
   identical seeds run on MPS vs CPU, even with `set_seed()`. This is expected
   due to floating-point ordering differences in MPS kernels. All reported
   numbers are from the MPS Phase 1 + CPU Phase 2 configuration.

**Approximate runtimes (Apple M-series, n=1000 graph):**

| Step | Time |
|---|---|
| Phase 1 imitation (50 epochs, MPS) | ~45 min |
| Phase 2 REINFORCE (200 epochs, CPU) | ~2 h |
| Budget training (150 epochs, CPU) | ~3.5 h |
| Eval on FF n=1000 (3 trials) | ~3–4 min |
| Full reproduce_all.sh (with checkpoints) | ~4–5 h |

---

## Checkpoints download

See `results/checkpoints/README.md` for the full catalog with SHA256s.

```bash
bash scripts/download_checkpoints.sh
```

---

## License

[To be determined before submission.]

## Citation

```bibtex
@inproceedings{revmax2027,
  title     = {Revenue-Maximising Offer Sequencing on Social Networks},
  author    = {[Authors]},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2027},
}
```
