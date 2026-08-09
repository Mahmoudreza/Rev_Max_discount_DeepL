#!/usr/bin/env python3
"""eval_protocol_unification.py — Rerun Greedy+Budget, Cal-DP composite, TFM-budget
under the specialist-eval harness (FF n=1000 graph seed=0, n_trials=3).

k sweep: [1, 3, 5, 10, 40]
Compare vs frozen paper_table_idea3_final.tex values.
Check: no best-per-row winner flips.

Output: results/logs/protocol_unification.json
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.graph_generators import generate_forest_fire
from src.env.budget_revenue_env import BudgetEnvConfig
from src.evaluation.budget_baselines import greedy_discount_budget
from src.evaluation.dp_calibrated_v2 import dp_calibrated_v2_budget
from src.evaluation.dp_calibrated_v3 import dp_calibrated_v3_budget

C          = 0.3
WEIGHT_HIGH = 2.0
N_TRIALS   = 3
K_EVAL     = [1, 3, 5, 10, 40]

# Frozen paper values (from paper_table_idea3_final.tex)
FROZEN = {
    1:  {"greedy": 7.4,   "dp":  10.6, "lstm": 298.0, "tfm": 150.1},
    3:  {"greedy": 23.6,  "dp":  99.8, "lstm": 327.9, "tfm": 211.9},
    5:  {"greedy": 48.9,  "dp": 154.2, "lstm": 332.0, "tfm": 278.4},
    10: {"greedy": 118.2, "dp": 435.1, "lstm": 346.9, "tfm": 273.8},
    40: {"greedy": 448.7, "dp": 448.0, "lstm": 473.1, "tfm": 389.1},
}
# Frozen winner (best non-oracle method per row in the paper table)
FROZEN_WINNER = {1: "lstm", 3: "lstm", 5: "lstm", 10: "dp", 40: "lstm"}


def eval_greedy(graph, k):
    r = greedy_discount_budget(graph, B=k*C, c=C, n_trials=N_TRIALS,
                                weight_high=WEIGHT_HIGH)
    return r["revenue"]["mean"], r["revenue"]["std"]


def eval_cal_dp(graph, k):
    """Cal-DP composite = max(v2, v3) per k."""
    B   = k * C
    cfg = BudgetEnvConfig(budget_B=B, production_cost=C, seed=0, weight_high=WEIGHT_HIGH)
    r2  = dp_calibrated_v2_budget(graph, cfg, B=B, c=C, n_trials=N_TRIALS)
    r3  = dp_calibrated_v3_budget(graph, cfg, B=B, c=C, n_trials=N_TRIALS)
    m2  = r2["revenue"]["mean"]
    m3  = r3["revenue"]["mean"]
    if m3 >= m2:
        return m3, r3["revenue"]["std"], "v3"
    return m2, r2["revenue"]["std"], "v2"


def eval_tfm(graph, k, device):
    """TFM-budget eval via evaluate_budget_aware_policy from budget_baselines."""
    from pathlib import Path
    import torch
    from src.evaluation.budget_baselines import evaluate_budget_aware_policy
    from src.models.encoders.graphsage import GraphSAGEEncoder
    from src.models.encoders.episode_transformer import EpisodeTransformerSliding
    from src.models.policies.transformer_joint_policy import TransformerJointPolicy
    from src.utils.helpers import load_config_with_base

    ROOT = Path(__file__).resolve().parents[1]
    TFM_CKPT  = ROOT / "results/checkpoints/rev_gnn_transformer_budget.pt"
    CFG_BUDGET = ROOT / "configs/experiments/budget_constrained.yaml"
    CFG_TFM   = ROOT / "configs/experiments/rev_gnn_transformer_300ep.yaml"

    from omegaconf import OmegaConf
    cfg_b = load_config_with_base(str(CFG_BUDGET))
    cfg_t = load_config_with_base(str(CFG_TFM))
    cfg   = OmegaConf.merge(cfg_b,
                OmegaConf.create({"transformer": OmegaConf.to_container(cfg_t.transformer)}))

    enc = GraphSAGEEncoder(21, int(cfg.encoder.hidden_dim),
                           int(cfg.encoder.n_layers), float(cfg.encoder.dropout))
    tfm = EpisodeTransformerSliding.from_config(cfg.transformer)
    pol = TransformerJointPolicy(enc, tfm,
                                 gnn_dim=int(cfg.encoder.hidden_dim),
                                 context_dim=cfg.transformer.d_model)
    sd  = torch.load(str(TFM_CKPT), map_location=device, weights_only=True)
    pol.load_state_dict(sd, strict=False)
    pol.to(device).eval()

    r = evaluate_budget_aware_policy(pol, graph, B=k*C, c=C,
                                     device=device, n_trials=N_TRIALS,
                                     weight_high=WEIGHT_HIGH)
    return r["revenue"]["mean"], r["revenue"]["std"]


def main():
    t0 = time.time()
    import torch
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Same graph as specialist eval
    graph = generate_forest_fire(1000, 0.37, 0.32, seed=0)
    n_edges = graph.number_of_edges()
    print(f"Graph: FF n=1000 edges={n_edges}")

    # Load TFM once (expensive)
    print("Loading TFM-budget...")
    t_tfm_load = time.time()
    try:
        _tfm_memo = {}
        def _get_tfm():
            if "pol" not in _tfm_memo:
                from pathlib import Path
                import torch
                from src.evaluation.budget_baselines import evaluate_budget_aware_policy
                from src.models.encoders.graphsage import GraphSAGEEncoder
                from src.models.encoders.episode_transformer import EpisodeTransformerSliding
                from src.models.policies.transformer_joint_policy import TransformerJointPolicy
                from src.utils.helpers import load_config_with_base
                from omegaconf import OmegaConf
                ROOT = Path(__file__).resolve().parents[1]
                TFM_CKPT  = ROOT / "results/checkpoints/rev_gnn_transformer_budget.pt"
                CFG_BUDGET = ROOT / "configs/experiments/budget_constrained.yaml"
                CFG_TFM   = ROOT / "configs/experiments/rev_gnn_transformer_300ep.yaml"
                cfg_b = load_config_with_base(str(CFG_BUDGET))
                cfg_t = load_config_with_base(str(CFG_TFM))
                cfg   = OmegaConf.merge(cfg_b,
                            OmegaConf.create({"transformer": OmegaConf.to_container(cfg_t.transformer)}))
                enc = GraphSAGEEncoder(21, int(cfg.encoder.hidden_dim),
                                       int(cfg.encoder.n_layers), float(cfg.encoder.dropout))
                tfm_mod = EpisodeTransformerSliding.from_config(cfg.transformer)
                pol = TransformerJointPolicy(enc, tfm_mod,
                                             gnn_dim=int(cfg.encoder.hidden_dim),
                                             context_dim=cfg.transformer.d_model)
                sd  = torch.load(str(TFM_CKPT), map_location=device, weights_only=True)
                pol.load_state_dict(sd, strict=False)
                pol.to(device).eval()
                _tfm_memo["pol"] = pol
                _tfm_memo["eval_fn"] = evaluate_budget_aware_policy
            return _tfm_memo["pol"], _tfm_memo["eval_fn"]

        tfm_pol, eval_fn_tfm = _get_tfm()
        tfm_ok = True
        print(f"  TFM loaded in {time.time()-t_tfm_load:.1f}s")
    except Exception as e:
        print(f"  TFM load failed: {e}")
        tfm_ok = False

    results = {}
    winner_flips = []
    print(f"\n{'k':>3} | {'Greedy new':>10} {'frz':>7} | {'DP new':>8} {'frz':>7} | {'TFM new':>8} {'frz':>7} | winner_chg")
    print("-" * 85)

    for k in K_EVAL:
        row = {}

        # Greedy
        gm, gs = eval_greedy(graph, k)
        row["greedy"] = {"mean": gm, "std": gs, "frozen": FROZEN[k]["greedy"],
                         "drift": round(gm - FROZEN[k]["greedy"], 1)}

        # Cal-DP composite
        dm, ds, winner_v = eval_cal_dp(graph, k)
        row["dp"] = {"mean": dm, "std": ds, "frozen": FROZEN[k]["dp"],
                     "drift": round(dm - FROZEN[k]["dp"], 1), "dp_winner": winner_v}

        # TFM
        if tfm_ok:
            try:
                r_tfm = eval_fn_tfm(tfm_pol, graph, B=k*C, c=C,
                                    device=device, n_trials=N_TRIALS,
                                    weight_high=WEIGHT_HIGH)
                tm = r_tfm["revenue"]["mean"]
                ts = r_tfm["revenue"]["std"]
            except Exception as e:
                print(f"  TFM k={k} error: {e}")
                tm, ts = float("nan"), 0.0
        else:
            tm, ts = float("nan"), 0.0
        row["tfm"] = {"mean": tm, "std": ts, "frozen": FROZEN[k]["tfm"],
                      "drift": round(tm - FROZEN[k]["tfm"], 1) if not np.isnan(tm) else "N/A"}

        # LSTM is frozen (not re-run; specialist at k=40 used)
        lm = FROZEN[k]["lstm"]
        row["lstm"] = {"mean": lm, "frozen": lm, "drift": 0.0}

        # Determine winner in new run (among greedy, dp, tfm, lstm)
        contenders = {"greedy": gm, "dp": dm, "lstm": lm}
        if not np.isnan(tm):
            contenders["tfm"] = tm
        new_winner = max(contenders, key=lambda x: contenders[x])
        old_winner = FROZEN_WINNER[k]
        flip = (new_winner != old_winner)
        if flip:
            winner_flips.append(k)

        results[k] = row
        tfm_str = f"{tm:8.1f}" if not np.isnan(tm) else "     N/A"
        print(f"{k:>3} | {gm:10.1f} {FROZEN[k]['greedy']:7.1f} | {dm:8.1f} {FROZEN[k]['dp']:7.1f} | "
              f"{tfm_str} {FROZEN[k]['tfm']:7.1f} | "
              f"{'FLIP '+old_winner+'->'+new_winner if flip else 'OK'}")

    print()
    if winner_flips:
        print(f"⚠ WINNER FLIPS at k={winner_flips} — STOP, do not edit paper")
    else:
        print("✓ No winner flips — table is stable")

    # Check max drift
    drifts = []
    for k, row in results.items():
        for method in ["greedy", "dp"]:
            d = row[method]["drift"]
            if isinstance(d, float):
                drifts.append(abs(d))
    max_drift = max(drifts) if drifts else 0.0
    print(f"Max drift (Greedy/DP): {max_drift:.1f}")
    print(f"Wall time: {time.time()-t0:.1f}s")

    out = {
        "graph_seed": 0,
        "n_trials": N_TRIALS,
        "k_eval": K_EVAL,
        "winner_flips": winner_flips,
        "max_drift_greedy_dp": max_drift,
        "results": {str(k): v for k, v in results.items()}
    }
    json.dump(out, open("results/logs/protocol_unification.json", "w"), indent=2)
    print("Saved → results/logs/protocol_unification.json")


if __name__ == "__main__":
    main()
