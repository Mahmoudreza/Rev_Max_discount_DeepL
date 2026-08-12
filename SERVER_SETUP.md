# Server Setup Guide — revmax-aaai2027

## What's in git (clone directly)
All Python source code + `data/raw/rice-facebook-undergrads-*.txt`

## What is NOT in git (must copy manually)
| File | Size | Why needed |
|------|------|-----------|
| `results/checkpoints/rev_gnn_lstm.pt` | 261KB | **Base model** (SHA 8fbc4648) — required for warm-start |
| `results/checkpoints/rev_gnn_lstm_densemix.pt` | 261KB | ep80 checkpoint — only if resuming from ep80 |
| `data/raw/polblogs_konect.tar.bz2` | ~500KB | Only for Gate B eval (NOT needed for training) |

Trajectory cache (`results/logs/ba_traj_cache/*.pkl`) is auto-rebuilt on first run.

---

## Step-by-step server setup

### 1. Clone the repository
```bash
git clone https://github.com/Mahmoudreza/Rev_Max_discount_DeepL.git revmax-aaai2027
cd revmax-aaai2027
```

### 2. Create Python 3.9+ virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### 3a. Install PyTorch — CUDA 12.8 Linux (Blackwell / H100 / A100 server — RECOMMENDED)
```bash
pip install torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

### 3b. Install PyTorch — CPU-only Linux (no GPU)
```bash
pip install torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cpu
```
> Note: Training script auto-detects GPU (`torch.cuda.is_available()`).
> It will use GPU automatically if cu128 PyTorch is installed.

### 4. Install PyTorch Geometric
```bash
pip install torch-geometric==2.6.1
```

### 5. Install remaining dependencies
```bash
pip install \
    networkx==3.2.1 \
    scipy==1.13.1 \
    numpy==2.0.2 \
    pandas==2.3.3 \
    scikit-learn==1.6.1 \
    matplotlib==3.9.4 \
    seaborn==0.13.2 \
    omegaconf==2.3.1 \
    PyYAML==6.0.3 \
    stable-baselines3==2.7.1 \
    gymnasium==1.1.1 \
    tqdm==4.68.3
```

### 6. Copy required checkpoints from local Mac to server
```bash
# Run these on your LOCAL Mac:
scp results/checkpoints/rev_gnn_lstm.pt         USER@SERVER:/path/to/revmax-aaai2027/results/checkpoints/
scp results/checkpoints/rev_gnn_lstm_densemix.pt USER@SERVER:/path/to/revmax-aaai2027/results/checkpoints/
# If doing eval (Gate B), also copy polblogs:
scp data/raw/polblogs_konect.tar.bz2             USER@SERVER:/path/to/revmax-aaai2027/data/raw/
```

### 7. Verify setup
```bash
cd revmax-aaai2027
source venv/bin/activate
python -c "import torch; import torch_geometric; print('torch', torch.__version__, 'pyg OK')"
python -c "
import hashlib
sha = hashlib.sha256(open('results/checkpoints/rev_gnn_lstm.pt','rb').read()).hexdigest()[:8]
assert sha == '8fbc4648', f'SHA mismatch: {sha}'
print('Base checkpoint OK sha8=', sha)
"
```

---

## Run training (ARM B, resume from ep80)
```bash
cd revmax-aaai2027
source venv/bin/activate
nohup python -u experiments/run_topology_arms.py --arm-b-only --resume-ep 80 \
    > /tmp/topology_arms.log 2>&1 &
echo "Training PID: $!"
```

**Monitor:**
```bash
tail -f /tmp/topology_arms.log
```

Expected output at each 20-epoch checkpoint:
```
  [B] P1 ep 80: loss=2.2447
  [B] periodic save → results/checkpoints/rev_gnn_lstm_densemix.pt (ep80)
  [B] P1 ep100: loss=2.xx
  ...
```

Training will save `densemix.pt` every 20 epochs during Phase 1 (ep80→199) and at the end of Phase 2.

**Total runtime estimate:** ~22h on CPU; significantly faster on GPU (GNN ops benefit from CUDA).

---

## Run Gate B eval (after checkpoint is ready)
```bash
nohup python -u experiments/run_topology_arms_eval.py --arm-b-only \
    > /tmp/gateB_eval.log 2>&1 &
# Wait ~55 min, then:
grep "GATE B:" /tmp/gateB_eval.log
```

Gate B: STRONG requires polblogs≥530.4 AND FF_1000≥440.0 AND Rice≥190.0

---

## Bring checkpoint back from server to Mac
```bash
# On your LOCAL Mac:
scp USER@SERVER:/path/to/revmax-aaai2027/results/checkpoints/rev_gnn_lstm_densemix.pt \
    /Users/reza/Desktop/revmax-aaai2027/results/checkpoints/rev_gnn_lstm_densemix.pt
```
