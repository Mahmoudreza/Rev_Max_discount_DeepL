"""
experiments/run_baselines.py

Run the full 10-method comparison table and print paper-ready results.
Saves CSV + LaTeX to results/logs/.

10 methods in 3 groups:
  Group 1 — Babaei et al. 2013:
    ie_strategy, mu_discount, sigma_discount, greedy_discount
  Group 2 — Deep IM (decoupled seeds + greedy pricing):
    s2v_dqn, touple_gdd
  Group 3 — Ours (joint seed + pricing):
    rev_gnn_im_rl, rev_gail_rl, rev_gnn_lstm, rev_gail_lstm

Usage:
  cd revmax-aaai2027
  python experiments/run_baselines.py
  python experiments/run_baselines.py --graph forest_fire --n 100 --n_trials 5
"""

import sys
import csv
import time
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.helpers import load_config_with_base, set_seed
from src.utils.logging import ExperimentLogger
from src.env.graph_generators import generate_forest_fire
from src.evaluation.baselines import run_full_comparison


# ── Group metadata ─────────────────────────────────────────────────────────────

GROUPS = [
    ("Babaei et al. 2013",
     ["ie_strategy", "mu_discount", "sigma_discount", "greedy_discount"]),
    ("Deep IM — Decoupled Seeds",
     ["s2v_dqn", "touple_gdd"]),
    ("Ours — Joint Selection + Pricing",
     ["rev_gnn_im_rl", "rev_gail_rl", "rev_gnn_lstm", "rev_gail_lstm"]),
]

METHOD_LABELS = {
    "ie_strategy":    "IE-Strategy",
    "mu_discount":    "µ-Discount",
    "sigma_discount": "σ-Discount",
    "greedy_discount":"Greedy-Discount",
    "s2v_dqn":        "S2V-DQN (decoupled)",
    "touple_gdd":     "ToupleGDD (decoupled)",
    "rev_gnn_im_rl":  "Rev-GNN-IM-RL  ▲",
    "rev_gail_rl":    "Rev-GAIL-RL    ▲",
    "rev_gnn_lstm":   "Rev-GNN-LSTM   ▲",
    "rev_gail_lstm":  "Rev-GAIL-LSTM  ▲",
}


def print_table(results, graph_name: str, logger):
    ie_rev = results.get("ie_strategy") or 1.0
    valid  = {k: v for k, v in results.items() if v is not None}
    best   = max(valid.values()) if valid else 0.0

    logger.info(f"\n{'='*72}")
    logger.info(f"  RevMax — Baseline Comparison  |  {graph_name}")
    logger.info(f"{'='*72}")
    logger.info(f"  {'Method':<36} {'Revenue':>10}  {'vs IE%':>7}")
    logger.info(f"  {'─'*56}")

    for grp_label, keys in GROUPS:
        logger.info(f"  [{grp_label}]")
        for k in keys:
            v = results.get(k)
            label = METHOD_LABELS.get(k, k)
            if v is None:
                logger.info(f"    {label:<34} {'N/A':>10}  {'—':>7}")
            else:
                pct  = 100.0 * (v - ie_rev) / (ie_rev + 1e-9)
                star = "  ◀ BEST" if abs(v - best) < 1e-9 else ""
                logger.info(f"    {label:<34} {v:>10.3f}  {pct:>+6.1f}%{star}")

    logger.info(f"{'='*72}")


def save_csv(results, graph_name: str, logger) -> Path:
    ie_rev = results.get("ie_strategy") or 1.0
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = Path(f"results/logs/comparison_{graph_name}_{ts}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "group", "revenue", "pct_over_ie"])
        for gi, (_, keys) in enumerate(GROUPS):
            for k in keys:
                v = results.get(k)
                pct = 100.0 * ((v or 0) - ie_rev) / (ie_rev + 1e-9) if v is not None else None
                w.writerow([k, gi + 1, v, pct])
    logger.info(f"CSV saved → {path}")
    return path


def save_latex(results, graph_name: str, logger) -> Path:
    ie_rev = results.get("ie_strategy") or 1.0
    valid  = {k: v for k, v in results.items() if v is not None}
    best   = max(valid.values()) if valid else 0.0
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = Path(f"results/logs/comparison_{graph_name}_{ts}.tex")
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Revenue comparison on " + graph_name.replace("_", r"\_") + r"}",
        r"\label{tab:comparison}",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Method & Group & Revenue & \% vs IE \\",
        r"\midrule",
    ]

    grp_names = ["Babaei 2013", "Deep IM (decoupled)", "Ours (joint)"]
    for gi, (grp_label, keys) in enumerate(GROUPS):
        grp_tex = grp_names[gi]
        lines.append(r"\multicolumn{4}{l}{\textit{" + grp_tex + r"}} \\")
        for k in keys:
            v = results.get(k)
            label = METHOD_LABELS.get(k, k).replace("▲", "").strip()
            label = label.replace("_", r"\_")
            if v is None:
                lines.append(f"  {label} & {gi+1} & N/A & --- \\\\")
            else:
                pct  = 100.0 * (v - ie_rev) / (ie_rev + 1e-9)
                sign = "+" if pct >= 0 else ""
                bold = r"\textbf{" if abs(v - best) < 1e-9 else ""
                endb = r"}" if bold else ""
                lines.append(
                    f"  {bold}{label}{endb} & {gi+1} & "
                    f"{bold}{v:.3f}{endb} & {sign}{pct:.1f}\\% \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines))
    logger.info(f"LaTeX saved → {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="RevMax 10-method comparison")
    parser.add_argument("--graph", default="forest_fire",
                        help="Graph type: forest_fire")
    parser.add_argument("--n",    type=int, default=100,
                        help="Nodes (synthetic graphs)")
    parser.add_argument("--n_trials", type=int, default=5,
                        help="MC trials for Babaei baselines")
    args = parser.parse_args()

    cfg = load_config_with_base(
        "configs/base_config.yaml",
        overrides=["influence.n_mc_samples=5", "training.reinforce_epochs=30"],
    )
    set_seed(cfg.project.seed)
    logger = ExperimentLogger(cfg, run_name=f"comparison_{args.graph}")

    graph = generate_forest_fire(
        args.n, p=cfg.graph.p, pb=cfg.graph.pb, seed=cfg.project.seed)

    logger.info(f"Graph: {args.graph}  n={graph.number_of_nodes()}  "
                f"m={graph.number_of_edges()}")

    t0 = time.time()
    results = run_full_comparison(
        graph, cfg,
        n_trials_babaei=args.n_trials,
        n_trials_deep_im=max(1, args.n_trials // 2),
    )
    logger.info(f"Total time: {time.time()-t0:.1f}s")

    print_table(results, args.graph, logger)
    save_csv(results, args.graph, logger)
    save_latex(results, args.graph, logger)

    logger.log({f"comparison/{k}": v for k, v in results.items() if v is not None})
    logger.finish()


if __name__ == "__main__":
    main()
