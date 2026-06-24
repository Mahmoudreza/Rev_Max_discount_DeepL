"""
experiments/run_budget_sweep.py

Run all 10 methods on SBM and rice_facebook across varying seed-budget k.
Produces a 2-D table: rows = methods, columns = (graph, k) pairs.

Saved outputs:
  results/logs/budget_sweep_<timestamp>.csv
  results/logs/budget_sweep_<timestamp>.tex
  results/logs/budget_sweep_<timestamp>.md   ← console-friendly Markdown

Usage:
  cd revmax-aaai2027
  python experiments/run_budget_sweep.py
  python experiments/run_budget_sweep.py --budgets 5,10,20,30,50 --n_trials 3
  python experiments/run_budget_sweep.py --budgets 10,30 --n_trials 1   # quick test
"""

import sys
import csv
import time
import pickle
import argparse
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from src.utils.helpers import load_config_with_base, set_seed
from src.utils.logging import ExperimentLogger
from src.env.graph_generators import generate_sbm
from src.evaluation.baselines import run_full_comparison


# ── Method display order and labels ──────────────────────────────────────────

GROUPS = [
    ("Babaei 2013",
     ["ie_strategy", "mu_discount", "sigma_discount", "greedy_discount"]),
    ("Deep IM — Decoupled",
     ["s2v_dqn", "touple_gdd"]),
    ("Ours — Joint (GNN+RL)",
     ["rev_gnn_im_rl", "rev_gail_rl"]),
    ("Ours — Joint (GNN+LSTM)",
     ["rev_gnn_lstm", "rev_gail_lstm"]),
]

ALL_METHODS = [m for _, ms in GROUPS for m in ms]

METHOD_LABELS = {
    "ie_strategy":    "IE-Strategy",
    "mu_discount":    "µ-Discount",
    "sigma_discount": "σ-Discount",
    "greedy_discount":"Greedy-Discount",
    "s2v_dqn":        "S2V-DQN (dec.)",
    "touple_gdd":     "ToupleGDD (dec.)",
    "rev_gnn_im_rl":  "Rev-GNN-IM-RL ▲",
    "rev_gail_rl":    "Rev-GAIL-RL   ▲",
    "rev_gnn_lstm":   "Rev-GNN-LSTM  ▲",
    "rev_gail_lstm":  "Rev-GAIL-LSTM ▲",
}


def _override_budget(cfg, k: int):
    """Return config with budget.k overridden to k."""
    return OmegaConf.merge(cfg, OmegaConf.create({"budget": {"k": k}}))


def run_sweep(graphs: dict, budgets: list, cfg, n_trials: int, logger) -> dict:
    """Run all methods on all (graph, k) combinations.

    Args:
        graphs:   {name: nx.Graph}
        budgets:  List of k values to sweep
        cfg:      Base OmegaConf config
        n_trials: MC trials per Babaei method
        logger:   ExperimentLogger

    Returns:
        results[(graph_name, k)] = {method: revenue_or_None}
    """
    results = {}
    combos = [(name, k) for name in graphs for k in budgets]
    total  = len(combos)

    for i, (name, k) in enumerate(combos):
        graph = graphs[name]
        cfg_k = _override_budget(cfg, k)
        n, m  = graph.number_of_nodes(), graph.number_of_edges()

        logger.info(f"\n[{i+1}/{total}]  {name}  n={n}  m={m}  k={k}")
        t0 = time.time()

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            r = run_full_comparison(
                graph, cfg_k,
                n_trials_babaei=n_trials,
                n_trials_deep_im=max(1, n_trials // 2),
            )

        elapsed = time.time() - t0
        results[(name, k)] = r

        # Quick per-row summary
        ie_rev = r.get("ie_strategy") or 1.0
        valid  = {k_: v for k_, v in r.items() if v is not None}
        best_k, best_v = (max(valid, key=valid.get), max(valid.values())) \
            if valid else ("—", 0.0)
        logger.info(
            f"  done in {elapsed:.1f}s  |  best={best_k}  {best_v:.2f}  "
            f"(IE={r.get('ie_strategy', 0):.2f}  "
            f"µ={r.get('mu_discount', 0):.2f}  "
            f"gr={r.get('greedy_discount', 0):.2f})"
        )
        logger.log({f"sweep/{name}/k{k}/{mk}": mv
                    for mk, mv in r.items() if mv is not None})

    return results


def print_table(sweep_results: dict, budgets: list, graph_names: list, logger):
    """Print a grouped table: rows = methods, columns = (graph, k)."""
    cols = [(g, k) for g in graph_names for k in budgets]
    col_width = 9

    # Header
    header = f"  {'Method':<22}"
    for g, k in cols:
        gshort = g.replace("rice_facebook", "rice_fb")
        header += f"  {gshort[:5]+'/k'+str(k):>{col_width}}"
    logger.info(f"\n{'='*90}")
    logger.info(f"  Budget Sweep: Revenue by Method, Graph, and k")
    logger.info(f"{'='*90}")
    logger.info(header)
    logger.info(f"  {'─'*88}")

    for grp_label, methods in GROUPS:
        logger.info(f"  [{grp_label}]")
        for m in methods:
            label = METHOD_LABELS.get(m, m)
            row = f"    {label:<20}"
            for g, k in cols:
                v = sweep_results.get((g, k), {}).get(m)
                row += f"  {v:>{col_width}.2f}" if v is not None else f"  {'—':>{col_width}}"
            logger.info(row)

    logger.info(f"{'='*90}")


def save_csv(sweep_results: dict, budgets: list, graph_names: list,
             ts: str, logger) -> Path:
    path = Path(f"results/logs/budget_sweep_{ts}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["method", "group"] + [
            f"{g}_k{k}" for g in graph_names for k in budgets]
        w.writerow(header)

        for gi, (grp_label, methods) in enumerate(GROUPS):
            for m in methods:
                row = [m, gi + 1]
                for g in graph_names:
                    for k in budgets:
                        v = sweep_results.get((g, k), {}).get(m)
                        row.append(f"{v:.4f}" if v is not None else "")
                w.writerow(row)

    logger.info(f"CSV  saved → {path}")
    return path


def save_latex(sweep_results: dict, budgets: list, graph_names: list,
               ts: str, logger) -> Path:
    path = Path(f"results/logs/budget_sweep_{ts}.tex")
    cols = [(g, k) for g in graph_names for k in budgets]

    # Find best per column (for bolding)
    col_best: dict = {}
    for g, k in cols:
        valid = {m: sweep_results.get((g, k), {}).get(m)
                 for m in ALL_METHODS
                 if sweep_results.get((g, k), {}).get(m) is not None}
        col_best[(g, k)] = max(valid.values()) if valid else 0.0

    n_cols = len(cols)
    col_spec = "l" + "r" * n_cols

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Revenue comparison across graphs and seed budget $k$. "
        r"Our joint Rev-GNN/GAIL models are marked with $\blacktriangle$.}",
        r"\label{tab:budget_sweep}",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
    ]

    # Column headers: graph name row
    gnames_row = "Method"
    for g in graph_names:
        gshort = g.replace("rice_facebook", "rice\_fb")
        n_k = len(budgets)
        gnames_row += f" & \\multicolumn{{{n_k}}}{{c}}{{\\textit{{{gshort}}}}}"
    lines.append(gnames_row + r" \\")

    # k values row
    k_row = ""
    for g in graph_names:
        for k in budgets:
            k_row += f" & $k={k}$"
    lines.append(k_row.lstrip(" & ").replace("& ", "Method & ", 1).rstrip()
                 + r" \\")
    # Actually let's do it properly:
    lines[-1] = "Method" + k_row + r" \\"
    lines.append(r"\midrule")

    grp_names_tex = [
        r"\textit{Babaei et al. 2013}",
        r"\textit{Deep IM (decoupled)}",
        r"\textit{Ours (joint GNN+RL)}",
        r"\textit{Ours (joint GNN+LSTM)}",
    ]

    for gi, (_, methods) in enumerate(GROUPS):
        lines.append(r"\multicolumn{" + str(n_cols + 1) + r"}{l}{" +
                     grp_names_tex[gi] + r"} \\")
        for m in methods:
            label = METHOD_LABELS.get(m, m).replace("▲", r"$\blacktriangle$") \
                                          .replace("µ", r"$\mu$") \
                                          .replace("σ", r"$\sigma$")
            row = f"  {label}"
            for g, k in cols:
                v = sweep_results.get((g, k), {}).get(m)
                if v is None:
                    row += " & ---"
                else:
                    best = col_best.get((g, k), 0.0)
                    if abs(v - best) < 1e-6:
                        row += f" & \\textbf{{{v:.2f}}}"
                    else:
                        row += f" & {v:.2f}"
            lines.append(row + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    path.write_text("\n".join(lines))
    logger.info(f"LaTeX saved → {path}")
    return path


def save_markdown(sweep_results: dict, budgets: list, graph_names: list,
                  ts: str, logger) -> Path:
    path = Path(f"results/logs/budget_sweep_{ts}.md")
    cols = [(g, k) for g in graph_names for k in budgets]

    lines = [
        "# Budget Sweep — Revenue Comparison",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    # Header row
    h = "| Method |"
    for g, k in cols:
        gs = g.replace("rice_facebook", "rice_fb")
        h += f" {gs}/k{k} |"
    lines += [h, "|" + "--------|" * (len(cols) + 1)]

    for grp_label, methods in GROUPS:
        lines.append(f"| **{grp_label}** |" + " |" * len(cols))
        for m in methods:
            row = f"| {METHOD_LABELS.get(m, m)} |"
            for g, k in cols:
                v = sweep_results.get((g, k), {}).get(m)
                row += f" {v:.2f} |" if v is not None else " — |"
            lines.append(row)

    path.write_text("\n".join(lines))
    logger.info(f"Markdown saved → {path}")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="RevMax budget sweep on SBM + rice_facebook")
    parser.add_argument("--budgets",   default="5,10,20,30",
                        help="Comma-separated k values (default: 5,10,20,30)")
    parser.add_argument("--n_trials",  type=int, default=3,
                        help="MC trials per Babaei method (default: 3)")
    parser.add_argument("--n_sbm",     type=int, default=120,
                        help="Number of nodes for SBM graph (default: 120)")
    parser.add_argument("--sbm_blocks",type=int, default=3)
    parser.add_argument("--sbm_pin",   type=float, default=0.30)
    parser.add_argument("--sbm_pout",  type=float, default=0.01)
    args = parser.parse_args()

    budgets = [int(k) for k in args.budgets.split(",")]

    cfg = load_config_with_base(
        "configs/base_config.yaml",
        overrides=["influence.n_mc_samples=5"],   # fast for sweep; use 200 for paper
    )
    set_seed(cfg.project.seed)
    ts = time.strftime("%Y%m%d_%H%M%S")
    logger = ExperimentLogger(cfg, run_name=f"budget_sweep_{ts}")

    # ── Build graphs ──────────────────────────────────────────────────────────
    graphs = {}

    # SBM
    sbm = generate_sbm(
        args.n_sbm,
        n_blocks=args.sbm_blocks,
        p_in=args.sbm_pin,
        p_out=args.sbm_pout,
        seed=cfg.project.seed,
    )
    graphs["sbm"] = sbm
    logger.info(f"SBM:           n={sbm.number_of_nodes()}  m={sbm.number_of_edges()}  "
                f"blocks={args.sbm_blocks}  p_in={args.sbm_pin}  p_out={args.sbm_pout}")

    # Rice facebook
    rice_path = "data/processed/rice_facebook.pkl"
    try:
        with open(rice_path, "rb") as f:
            rice = pickle.load(f)
        graphs["rice_facebook"] = rice
        logger.info(f"rice_facebook:  n={rice.number_of_nodes()}  m={rice.number_of_edges()}")
    except FileNotFoundError:
        logger.info(f"WARNING: {rice_path} not found — skipping rice_facebook")

    graph_names = list(graphs.keys())
    logger.info(f"\nBudgets k = {budgets}")
    logger.info(f"Graphs      = {graph_names}")
    logger.info(f"MC trials   = {args.n_trials}")
    logger.info(f"Total runs  = {len(graph_names) * len(budgets)}\n")

    # ── Run sweep ─────────────────────────────────────────────────────────────
    t_total = time.time()
    sweep_results = run_sweep(graphs, budgets, cfg, args.n_trials, logger)
    logger.info(f"\nTotal wall time: {time.time()-t_total:.1f}s")

    # ── Print + save ──────────────────────────────────────────────────────────
    print_table(sweep_results, budgets, graph_names, logger)
    save_csv(     sweep_results, budgets, graph_names, ts, logger)
    save_latex(   sweep_results, budgets, graph_names, ts, logger)
    save_markdown(sweep_results, budgets, graph_names, ts, logger)

    logger.finish()
    return sweep_results


if __name__ == "__main__":
    main()
