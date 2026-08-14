#!/usr/bin/env bash
# setup_c1_core.sh — Task 1: Isolate C1 material
# Uses git mv to preserve history. Nothing deleted.
# Run from repo root: bash setup_c1_core.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "=== Creating directory structure ==="
mkdir -p c1_core/{checkpoints,experiments}
mkdir -p archive/{src/{env,evaluation,models/{encoders,policies,sequence},training,utils},experiments}

# ── c1_core/__init__.py ──────────────────────────────────────────────────
cat > c1_core/__init__.py << 'PYEOF'
"""
C1 core — Rev-GNN-LSTM (Contribution 1, unconstrained revenue maximisation)

Checkpoint: results/checkpoints/rev_gnn_lstm.pt  sha8=8fbc4648
            c1_core/checkpoints/rev_gnn_lstm.pt   (copy)

C1 source modules (remain in src/ — imported normally):
  src/env/revenue_env.py           — RevenueEnv (unconstrained)
  src/env/graph_generators.py      — ForestFire / Modular generators
  src/env/polblogs_loader.py       — polblogs graph loader
  src/env/influence_models.py      — IC / LT influence models
  src/env/sbm_generators.py        — SBM generator
  src/models/encoders/graphsage.py — GraphSAGE encoder
  src/models/encoders/sequence_models.py — EpisodeLSTM
  src/models/policies/base_policy.py           — BasePolicy
  src/models/policies/sequential_joint_policy.py — SequentialJointPolicy (LSTM)
  src/training/imitation_trainer.py   — Phase-1 imitation (IL)
  src/training/reinforce_trainer.py   — Phase-2 REINFORCE (RL)
  src/training/mixed_expert_trajectories.py — expert trajectory generation
  src/evaluation/baselines.py     — IE-Strategy, mu-Discount, Greedy-Discount
  src/evaluation/idea1_eval.py    — C1 evaluation harness
  src/evaluation/paper_eval.py    — paper protocol helpers
  src/evaluation/evaluate.py      — single-episode eval
  src/utils/features.py           — node feature computation
  src/utils/logging.py            — ExperimentLogger
  src/utils/helpers.py            — misc helpers

C1 experiments: c1_core/experiments/
  run_rev_gnn_lstm.py   — C1 training (Phase-1 IL + Phase-2 RL)
  run_rev_gnn_im_rl.py  — stateless ablation training (IM-RL)
  run_polblogs_eval.py  — polblogs eval harness
  eval_idea1.py         — generalisation eval (5 networks)
  eval_on_rice_facebook.py — Rice-FB eval
  run_baselines.py      — baseline runner
  eval_c1_final.py      — NEW: re-run with shas attribution
  run_c1_final_parallel.sh — NEW: GPU-parallel launcher
"""
PYEOF

echo "=== Copying checkpoint (not moving) ==="
cp results/checkpoints/rev_gnn_lstm.pt c1_core/checkpoints/rev_gnn_lstm.pt
# Allow .pt files in c1_core/checkpoints/ (override .gitignore *.pt rules)
if ! grep -q 'c1_core/checkpoints' .gitignore 2>/dev/null; then
  printf '\n# Allow C1 checkpoint copy\n!c1_core/checkpoints/*.pt\n' >> .gitignore
fi

# Helper: git mv if tracked, plain mv otherwise (with git add for dest).
# Idempotent: skips if source missing or destination already exists.
safe_mv() {
  local src="$1" dst="$2"
  [ -f "$src" ] || return 0   # source missing — skip
  [ -f "$dst" ] && return 0   # dest already exists — skip (previous partial run)
  if git ls-files --error-unmatch "$src" &>/dev/null 2>&1; then
    git mv "$src" "$dst"
  else
    mv "$src" "$dst"
    git add -f "$dst"   # -f: override any .gitignore rules on archive/ paths
  fi
}

echo "=== git mv: C1 experiments → c1_core/experiments/ ==="
for f in \
  run_rev_gnn_lstm.py \
  run_rev_gnn_im_rl.py \
  run_polblogs_eval.py \
  eval_idea1.py \
  eval_on_rice_facebook.py \
  run_baselines.py; do
  safe_mv "experiments/$f" "c1_core/experiments/$f" || echo "  SKIP (missing): $f"
done

echo "=== git mv: non-C1 experiments → archive/experiments/ ==="
NON_C1_EXPS=(
  run_budget_eval.py run_budget_retrain.py run_budget_sweep.py
  run_budget_training.py run_budget_unified_training.py
  run_c1_ffba_training.py eval_c1_ffba.py
  run_dp_upgrade_eval.py run_dp_v3_eval.py run_dp_v3_ff_sweep.py run_dp_v3_full_curve.py
  run_fair_greedy_gate_f1.py run_fair_rl_training.py run_fairness_audit.py run_gnn_fairness_audit.py
  run_gate_b_transformer_eval.py eval_transformer_gate_a.py
  run_hybrid_sweep.py run_largek_eval.py run_largek_specialist.py
  run_lstm_idea3_retrained_sweep.py
  run_phase2_only.py
  run_polblogs_budget_sweep.py run_polblogs_caldp_only.py
  run_rev_gail.py run_rev_gail_lstm.py run_rev_gail_rl_rich.py
  run_rev_gnn_transformer.py run_rev_ppo.py
  run_stage_a_rollout_gate.py run_stage_b_student_training.py
  run_tc_eval.py run_tc_lstm_training.py run_tc_training.py run_time_critical.py
  run_topology_arms.py run_topology_arms_eval.py
  run_transformer_budget_training.py run_unified_sweep.py
  run_ba_probes.py run_collapse_probe1.py run_proxy_dashboard.py
  run_all_experiments.py run_benchmark.py run_rev_ppo.py
  eval_all_methods_k50.py eval_all_methods_ksweep.py
  eval_arm_b_ep160.py eval_baselines_budget_k50.py
  eval_ie_budget_ksweep.py eval_protocol_unification.py
  eval_unified_k16_25.py eval_unified_k16_25_fast.py
  gen_largek_trajectories.py
  merge_all_methods_shards.py merge_c1_ffba_shards.py merge_ie_shards.py merge_ksweep_shards.py
  run_budget_retrain.py run_ie_budget_ksweep.py
  continue_rice_sweep.py download_networks.py env_validate_frozen.py
  identify_published_lstm_ckpt.py audit_r0_5.py
  plot_fairness_figures.py plot_figures.py plot_idea3_main_v2.py plot_idea3_main_v3.py
  paper_figures.py generate_paper_figures.py make_idea3_figures.py
  ablation_encoder_type.py ablation_graph_type.py ablation_influence_model.py ablation_reward_function.py
  quick_modff_eval.py spotcheck_gatefail.py verify_f1_decomposition.py
  sanity_largek_ff1000_k40.py sanity_unified_ff1000_k10.py
  run_budget_unified_training.py run_ie_shards.py run_ksweep_parallel.sh
  run_ie_budget_ksweep_parallel.sh
)
for f in "${NON_C1_EXPS[@]}"; do
  safe_mv "experiments/$f" "archive/experiments/$f"
done
# Move figures subdir
if [ -d "experiments/figures" ]; then
  if git ls-files --error-unmatch "experiments/figures" &>/dev/null 2>&1; then
    git mv experiments/figures archive/experiments/figures
  else
    mv experiments/figures archive/experiments/figures
    git add -f "archive/experiments/figures"
  fi
fi

echo "=== git mv: non-C1 src → archive/src/ ==="

# env
for f in budget_revenue_env.py time_critical_revenue_env.py ba_generators.py; do
  safe_mv "src/env/$f" "archive/src/env/$f"
done

# evaluation
for f in budget_baselines.py bmin_feasibility.py dp_calibrated.py dp_calibrated_v2.py \
          dp_calibrated_v3.py fair_baselines.py fairness_audit.py \
          hybrid_lookahead_policy.py ie_budget.py rollout_expert.py \
          tc_baselines.py tc_evaluation.py; do
  safe_mv "src/evaluation/$f" "archive/src/evaluation/$f"
done

# models
for f in encoders/episode_transformer.py encoders/graph_transformer.py \
          policies/joint_policy.py policies/ppo_policy.py policies/sac_policy.py \
          policies/transformer_joint_policy.py \
          sequence/lstm_policy.py sequence/transformer_policy.py; do
  safe_mv "src/models/$f" "archive/src/models/$f"
done

# training
for f in gail_trainer.py ppo_trainer.py sac_trainer.py tc_reinforce_trainer.py; do
  safe_mv "src/training/$f" "archive/src/training/$f"
done

# utils
for f in budget_features.py budget_visualization.py dp_upgrade_visualization.py \
          idea3_figures.py tc_visualization.py; do
  safe_mv "src/utils/$f" "archive/src/utils/$f"
done

echo "=== git add new files ==="
git add c1_core/__init__.py
git add -f c1_core/checkpoints/rev_gnn_lstm.pt   # -f: override *.pt in .gitignore
git add .gitignore                                # record the !c1_core/checkpoints exception

echo "=== Done. Committing... ==="
git commit -m "feat: isolate C1 material into c1_core/, archive non-C1 code

- git mv C1 experiments → c1_core/experiments/
- git mv non-C1 experiments → archive/experiments/
- git mv non-C1 src/ files → archive/src/
- copy checkpoint → c1_core/checkpoints/rev_gnn_lstm.pt (orig preserved)
- add c1_core/__init__.py manifest

Nothing deleted. All history preserved."

echo "=== Smoke test ==="
python3 -c "
import sys; sys.path.insert(0,'.')
from src.env.revenue_env import RevenueEnv
from src.env.graph_generators import make_forest_fire
from src.evaluation.baselines import ie_strategy, greedy_discount
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.training.imitation_trainer import ImitationTrainer
from src.training.reinforce_trainer import ReinforceTrainer
from src.utils.features import compute_node_features
print('C1 smoke test: ALL IMPORTS OK')
"
