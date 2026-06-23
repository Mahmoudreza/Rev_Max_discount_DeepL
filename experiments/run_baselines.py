"""
experiments/run_baselines.py

Run ALL baseline groups and print a paper-ready comparison table.
Saves CSV + LaTeX table to results/logs/.

Usage:
  cd revmax-aaai2027
  python experiments/run_baselines.py
  python experiments/run_baselines.py --graph forest_fire --n_trials 10
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
from src.evaluation.baselines import (
    run_group_a_baselines, run_group_b_baselines, run_group_c_baselines,
)


GROUP_LABELS = {
    "A": "Hand-crafted",
    "B": "Decoupled GNN",
    "C": "Joint Approx",
}

METHOD_GROUP = {
    "random": "A", "myopic_full": "A", "ie_strategy": "A",
    "mu_discount": "A", "sigma_discount": "A", "greedy_discount": "A",
    "hill_climbing": "A", "local_search": "A",
    "s2v_dqn_decoupled": "B", "touple_gdd_decoupled": "B",
    "dgn_decoupled": "B", "wsdm_gnn_im_rl_decoupled": "B",
    "wsdm_gail_rl_decoupled": "B",
    "prisca": "C",
}


def run_baselines(cfg, graph, graph_name: str, n_trials: int, logger):
    logger.info(f"\n{'='*65}")
    logger.info(f"  Baselines on {graph_name}  "
                f"(n={graph.number_of_nodes()}, m={graph.number_of_edges()})")
    logger.info(f"{'='*65}")

    results = {}

    logger.info("\n[Group A] Hand-crafted baselines …")
    t0 = time.time()
    results.update(run_group_a_baselines(graph, cfg, n_trials=n_trials))
    logger.info(f"  Group A done in {time.time()-t0:.1f}s")

    logger.info("[Group B] Decoupled GNN baselines …")
    t0 = time.time()
    results.update(run_group_b_baselines(graph, cfg, n_trials=max(1, n_trials // 2)))
    logger.info(f"  Group B done in {time.time()-t0:.1f}s")

    logger.info("[Group C] PriSCa joint approx …")
    t0 = time.time()
    results.update(run_group_c_baselines(graph, cfg, n_trials=n_trials))
    logger.info(f"  Group C done in {time.time()-t0:.1f}s")

    ie_rev = results.get("ie_strategy", 1.0) or 1.0

    logger.info(f"\n{'─'*65}")
    logger.info(f"  {'Method':<30} {'Group':>5} {'Revenue':>10} {'vs IE%':>8}")
    logger.info(f"{'─'*65}")
    for grp in ("A", "B", "C"):
        for method, rev in results.items():
            if METHOD_GROUP.get(method) != grp:
                continue
            if rev is None:
                logger.info(f"  {method:<30} {grp:>5} {'N/A':>10} {'—':>8}")
            else:
                pct = 100.0 * (rev - ie_rev) / (ie_rev + 1e-9)
                marker = " ◀" if rev == max(v for v in results.values() if v is not None) else ""
                logger.info(f"  {method:<30} {grp:>5} {rev:>10.4f} {pct:>+7.1f}%{marker}")
    logger.info(f"{'='*65}")

    # Save CSV
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = Path(f"results/logs/baselines_{graph_name}_{ts}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "group", "revenue", "pct_over_ie"])
        for method, rev in results.items():
            grp = METHOD_GROUP.get(method, "?")
            pct = 100.0 * ((rev or 0) - ie_rev) / (ie_rev + 1e-9) if rev is not None else None
            w.writerow([method, grp, rev, pct])
    logger.info(f"CSV saved → {csv_path}")

    # Save LaTeX
    tex_path = Path(f"results/logs/baselines_{graph_name}_{ts}.tex")
    _write_latex_table(results, ie_rev, graph_name, tex_path)
    logger.info(f"LaTeX saved → {tex_path}")

    logger.log({f"baselines/{k}": v for k, v in results.items() if v is not None})
    return results


def _write_latex_table(results, ie_rev, graph_name, path):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Baseline revenue comparison on " + graph_name.replace("_", r"\_") + r"}",
        r"\label{tab:baselines}",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Method & Group & Revenue & \% vs IE \\",
        r"\midrule",
    ]
    for grp, grp_label in [("A", "Hand-crafted"), ("B", "Decoupled GNN"), ("C", "Joint Approx")]:
        lines.append(r"\multicolumn{4}{l}{\textit{" + grp_label + r"}} \\")
        for method, rev in results.items():
            if METHOD_GROUP.get(method) != grp:
                continue
            m = method.replace("_", r"\_")   # pre-escape for LaTeX (no backslash in f-str)
            if rev is None:
                lines.append(f"  {m} & {grp} & N/A & --- \\\\")
            else:
                pct = 100.0 * (rev - ie_rev) / (ie_rev + 1e-9)
                sign = "+" if pct >= 0 else ""
                pct_str = f"{sign}{pct:.1f}\\%"
                lines.append(f"  {m} & {grp} & {rev:.4f} & {pct_str} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Run all RevMax baselines")
    parser.add_argument("--graph", default="forest_fire")
    parser.add_argument("--n", type=int, default=100, help="Graph size (synthetic)")
    parser.add_argument("--n_trials", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config_with_base("configs/base_config.yaml",
                                overrides=["influence.n_mc_samples=5",
                                           "training.reinforce_epochs=30"])
    set_seed(cfg.project.seed)
    logger = ExperimentLogger(cfg, run_name=f"baselines_{args.graph}")

    graph = generate_forest_fire(args.n, p=cfg.graph.p, pb=cfg.graph.pb,
                                 seed=cfg.project.seed)
    run_baselines(cfg, graph, args.graph, args.n_trials, logger)
    logger.finish()


if __name__ == "__main__":
    main()
