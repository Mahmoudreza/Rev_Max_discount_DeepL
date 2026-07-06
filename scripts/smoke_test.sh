#!/usr/bin/env bash
# scripts/smoke_test.sh — 5-minute end-to-end sanity check on a fresh machine.
# Tests the full stack (graph generation → imitation → eval → figure) on tiny
# graphs (n=100) WITHOUT requiring any pre-trained checkpoints.
#
# Usage: bash scripts/smoke_test.sh
# Expected runtime: 3–5 min on Apple Silicon, 8–12 min on Linux CPU.
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate

echo "=== RevMax smoke test ==="
PASS=0; FAIL=0

fail() { echo "  FAIL: $1"; FAIL=$((FAIL+1)); }
pass() { echo "  PASS: $1"; PASS=$((PASS+1)); }

# ── 1. Core imports ─────────────────────────────────────────────────────────
echo ""
echo "[1] Imports..."
python - <<'EOF'
import torch, torch_geometric, networkx, omegaconf, numpy, scipy
import matplotlib, tqdm, gymnasium, stable_baselines3
print(f"  torch {torch.__version__} | pyg {torch_geometric.__version__} | "
      f"networkx {networkx.__version__}")
EOF
pass "all imports"

# ── 2. Graph generation + static features ─────────────────────────────────
echo ""
echo "[2] Graph + features (n=100)..."
python - <<'EOF'
import sys; sys.path.insert(0,'.')
from src.env.graph_generators import generate_forest_fire
from src.utils.features import compute_static_features, build_graph_feature_cache, compute_node_features_fast
from src.utils.helpers import load_config_with_base
from src.evaluation.baselines import _make_env
g = generate_forest_fire(100, 0.37, 0.32, seed=42)
statics = compute_static_features(g)   # returns dict
cache   = build_graph_feature_cache(g, statics)
cfg = load_config_with_base('configs/base_config.yaml')
env = _make_env(g, cfg); env.reset()
feats = compute_node_features_fast(cache=cache, S=frozenset(), offered=frozenset(),
                                   t=0, k=g.number_of_nodes(), env=env)
assert feats.shape == (100, 20), f"Expected (100,20), got {feats.shape}"
print(f"  graph n={g.number_of_nodes()} m={g.number_of_edges()} | feats {feats.shape}")
EOF
pass "graph + features"

# ── 3. Tiny baseline run ────────────────────────────────────────────────────
echo ""
echo "[3] Greedy baseline on n=100..."
python - <<'EOF'
import sys; sys.path.insert(0,'.')
from omegaconf import OmegaConf
from src.utils.helpers import load_config_with_base
from src.env.graph_generators import generate_forest_fire
from src.evaluation.baselines import greedy_discount
g = generate_forest_fire(100, 0.37, 0.32, seed=42)
cfg = load_config_with_base('configs/base_config.yaml')
rev = greedy_discount(g, cfg)
print(f"  Greedy revenue on n=100: {rev:.1f}")
assert rev > 0, "Greedy returned zero revenue"
EOF
pass "greedy baseline"

# ── 4. 2-epoch imitation training (LSTM, tiny graph) ───────────────────────
echo ""
echo "[4] 2-epoch LSTM imitation on n=100 (sanity, not a real checkpoint)..."
python - <<'EOF'
import sys, torch, copy; sys.path.insert(0,'.')
import torch.nn.functional as F
from src.utils.helpers import load_config_with_base, set_seed, graph_to_pyg_data, get_available_mask
from src.utils.features import compute_static_features, build_graph_feature_cache, compute_node_features_fast
from src.env.graph_generators import generate_forest_fire
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.evaluation.baselines import greedy_discount_trajectory, _make_env

cfg = load_config_with_base('configs/experiments/rev_gnn_lstm.yaml')
set_seed(42)
device = torch.device('cpu')
g = generate_forest_fire(100, cfg.graph.p, cfg.graph.pb, seed=42)
n = g.number_of_nodes(); nodes = list(g.nodes())
enc  = GraphSAGEEncoder(20, 64, 2, 0.0)
lstm = EpisodeLSTM(graph_dim=64, lstm_hidden=64, n_layers=1)
pol  = SequentialJointPolicy(enc, lstm, gnn_dim=64, context_dim=64).to(device)
opt  = torch.optim.Adam(pol.parameters(), lr=1e-3)
statics = compute_static_features(g)
cache   = build_graph_feature_cache(g, statics)
traj    = greedy_discount_trajectory(g, cfg)
losses = []
for epoch in range(2):
    env = _make_env(g, cfg); env.reset()
    pol.reset_episode(device); pol.train()
    S, off = frozenset(), frozenset()
    for td in traj[:20]:  # first 20 steps only for speed
        nidx, ed, acc = td['node_idx'], td['discount'], td.get('accepted', True)
        feats = compute_node_features_fast(cache=cache, S=S, offered=off, t=len(off), k=n, env=env)
        data  = graph_to_pyg_data(g, feats, device)
        mask  = get_available_mask(n, off, nodes, device)
        ms, h, ctx, _ = pol.forward(data.x, data.edge_index, mask)
        loss = F.cross_entropy(ms.unsqueeze(0), torch.tensor([nidx], device=device))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        node = nodes[nidx]
        if acc: S = frozenset(S | {node})
        off = frozenset(off | {node}); env.t += 1
        pol.update_sequence_state(ed, acc, 0.0)
print(f"  Loss after 2 epochs × 20 steps: {losses[-1]:.4f}")
assert losses[-1] < losses[0] * 2, "Loss appears diverging"
EOF
pass "2-epoch imitation (loss stable)"

# ── 5. pytest ─────────────────────────────────────────────────────────────
echo ""
echo "[5] pytest..."
pytest tests/ -q --tb=short 2>&1 | tail -6
pass "pytest"

# ── 6. Figure render (tiny) ────────────────────────────────────────────────
echo ""
echo "[6] Minimal figure render (matplotlib smoke test)..."
python - <<'EOF'
import matplotlib.pyplot as plt, numpy as np
fig, ax = plt.subplots()
ax.plot([1,2,3], [10,20,15], label='test')
ax.set_xlabel('k'); ax.set_ylabel('revenue'); ax.legend()
fig.savefig('/tmp/revmax_smoke_fig.png', dpi=72)
plt.close()
print("  Figure saved to /tmp/revmax_smoke_fig.png")
EOF
pass "figure render"

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "================================"
echo "Smoke test: $PASS passed, $FAIL failed"
if [ $FAIL -eq 0 ]; then
    echo "STATUS: OK — machine is ready for full reproduction."
    echo "Next: bash scripts/reproduce_all.sh  (3-4 h total)"
    exit 0
else
    echo "STATUS: FAIL — see errors above before proceeding."
    exit 1
fi
