"""
experiments/continue_rice_sweep.py

Continuation: runs rice_facebook with k=[5,10,20,30] (all budgets) and
combines with hardcoded SBM results from the previous partial run.

Saves final outputs:
  results/logs/budget_sweep_final.csv
  results/logs/budget_sweep_final.tex
  results/logs/budget_sweep_final.md

Usage:
  cd revmax-aaai2027
  nohup venv/bin/python experiments/continue_rice_sweep.py > results/logs/continue_rice.log 2>&1 &
"""

import sys
import csv
import time
import pickle
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from src.utils.helpers import load_config_with_base, set_seed
from src.evaluation.baselines import run_full_comparison

# ── Hardcoded SBM results from previous run ────────────────────────────────────
# (from budget_sweep_20260624_112057... rows 0-3 / steps 0-3)
# Each row: methods in order [ie_strategy, mu_discount, sigma_discount,
#                              greedy_discount, s2v_dqn, touple_gdd]

SBM_RESULTS = {
    ("sbm", 5): {
        "ie_strategy":    15.595131010645822,
        "mu_discount":    60.97288601431193,
        "sigma_discount": 48.12783808137946,
        "greedy_discount":42.05925362405735,
        "s2v_dqn":        41.28020601990295,
        "touple_gdd":     41.20123565564141,
    },
    ("sbm", 10): {
        "ie_strategy":    28.168799955192412,
        "mu_discount":    60.97288601431193,
        "sigma_discount": 48.12783808137946,
        "greedy_discount":42.05925362405735,
        "s2v_dqn":        41.28020601990295,
        "touple_gdd":     41.201045038556835,
    },
    ("sbm", 20): {
        "ie_strategy":    41.639198297771074,
        "mu_discount":    60.97288601431193,
        "sigma_discount": 48.12783808137946,
        "greedy_discount":42.05925362405735,
        "s2v_dqn":        41.083645701557444,
        "touple_gdd":     41.07978631139072,
    },
    ("sbm", 30): {
        "ie_strategy":    47.92373848293633,
        "mu_discount":    60.97288601431193,
        "sigma_discount": 48.12783808137946,
        "greedy_discount":42.05925362405735,
        "s2v_dqn":        40.93252031374705,
        "touple_gdd":     40.430084307236825,
    },
    ("rice_facebook", 5): {
        "ie_strategy":    26.543425342117803,
        "mu_discount":    241.2822353352729,
        "sigma_discount": 190.27508735708358,
        "greedy_discount":158.19117604692917,
        "s2v_dqn":        166.08745835066202,
        "touple_gdd":     None,   # timed out at 120s; now retrying with 300s
    },
}

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
    return OmegaConf.merge(cfg, OmegaConf.create({"budget": {"k": k}}))


def main():
    budgets = [5, 10, 20, 30]

    print(f"[continue_rice_sweep] Loading config...", flush=True)
    cfg = load_config_with_base(
        "configs/base_config.yaml",
        overrides=["influence.n_mc_samples=5"],
    )
    set_seed(cfg.project.seed)

    # Load rice_facebook graph
    rice_path = "data/processed/rice_facebook.pkl"
    try:
        with open(rice_path, "rb") as f:
            rice = pickle.load(f)
        print(f"[continue_rice_sweep] rice_facebook: n={rice.number_of_nodes()} "
              f"m={rice.number_of_edges()}", flush=True)
    except FileNotFoundError:
        print(f"[continue_rice_sweep] ERROR: {rice_path} not found", flush=True)
        sys.exit(1)

    # Start with pre-computed SBM data
    sweep_results = dict(SBM_RESULTS)

    # Run ALL rice_facebook budgets (re-runs k=5 to also get ToupleGDD with 300s timeout)
    for k in budgets:
        cfg_k = _override_budget(cfg, k)
        print(f"\n[continue_rice_sweep] rice_facebook  k={k}  "
              f"(ToupleGDD timeout=300s)", flush=True)
        t0 = time.time()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = run_full_comparison(
                rice, cfg_k,
                n_trials_babaei=1,
                n_trials_deep_im=1,
            )
        elapsed = time.time() - t0

        if caught:
            for w in caught:
                print(f"  WARN: {w.message}", flush=True)

        sweep_results[("rice_facebook", k)] = r
        valid = {m: v for m, v in r.items() if v is not None}
        best_m = max(valid, key=valid.get) if valid else "—"
        best_v = valid[best_m] if valid else 0.0
        print(f"  done in {elapsed:.1f}s  best={best_m}  {best_v:.2f}  "
              f"ie={r.get('ie_strategy',0):.2f}  "
              f"mu={r.get('mu_discount',0):.2f}  "
              f"s2v={r.get('s2v_dqn') or '—'}  "
              f"touple={r.get('touple_gdd') or '—'}", flush=True)

    # ── Save results ──────────────────────────────────────────────────────────
    graph_names = ["sbm", "rice_facebook"]
    ts = "final"

    # --- CSV ---
    csv_path = Path("results/logs/budget_sweep_final.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["method", "group"] + [f"{g}_k{k}"
                                         for g in graph_names for k in budgets]
        w.writerow(header)
        for gi, (grp_label, methods) in enumerate(GROUPS):
            for m in methods:
                row = [m, gi + 1]
                for g in graph_names:
                    for k in budgets:
                        v = sweep_results.get((g, k), {}).get(m)
                        row.append(f"{v:.4f}" if v is not None else "")
                w.writerow(row)
    print(f"\nCSV  saved → {csv_path}", flush=True)

    # --- Markdown ---
    cols = [(g, k) for g in graph_names for k in budgets]
    md_path = Path("results/logs/budget_sweep_final.md")
    lines = [
        "# Budget Sweep — Revenue Comparison",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "*(SBM results from earlier run; ToupleGDD timeout increased to 300s)*",
        "",
    ]
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
    md_path.write_text("\n".join(lines))
    print(f"MD   saved → {md_path}", flush=True)

    # --- LaTeX ---
    tex_path = Path("results/logs/budget_sweep_final.tex")
    col_best = {}
    for g, k in cols:
        valid = {m: sweep_results.get((g, k), {}).get(m)
                 for m in ALL_METHODS
                 if sweep_results.get((g, k), {}).get(m) is not None}
        col_best[(g, k)] = max(valid.values()) if valid else 0.0

    n_cols = len(cols)
    tex_lines = [
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\caption{Revenue comparison across graphs and seed budget $k$.}",
        r"\label{tab:budget_sweep}",
        r"\begin{tabular}{l" + "r" * n_cols + r"}",
        r"\toprule",
    ]
    gnames_row = "Method"
    for g in graph_names:
        gs = g.replace("rice_facebook", r"rice\_fb")
        gnames_row += f" & \\multicolumn{{{len(budgets)}}}{{c}}{{\\textit{{{gs}}}}}"
    tex_lines.append(gnames_row + r" \\")
    k_row = "Method"
    for g in graph_names:
        for k in budgets:
            k_row += f" & $k={k}$"
    tex_lines.append(k_row + r" \\")
    tex_lines.append(r"\midrule")

    grp_tex = [r"\textit{Babaei et al. 2013}",
               r"\textit{Deep IM (decoupled)}",
               r"\textit{Ours (joint GNN+RL)}",
               r"\textit{Ours (joint GNN+LSTM)}"]
    for gi, (_, methods) in enumerate(GROUPS):
        tex_lines.append(r"\multicolumn{" + str(n_cols+1) + r"}{l}{" +
                         grp_tex[gi] + r"} \\")
        for m in methods:
            label = (METHOD_LABELS.get(m, m)
                     .replace("▲", r"$\blacktriangle$")
                     .replace("µ", r"$\mu$")
                     .replace("σ", r"$\sigma$"))
            row = f"  {label}"
            for g, k in cols:
                v = sweep_results.get((g, k), {}).get(m)
                if v is None:
                    row += " & ---"
                else:
                    best = col_best.get((g, k), 0.0)
                    row += (f" & \\textbf{{{v:.2f}}}" if abs(v - best) < 1e-6
                            else f" & {v:.2f}")
            tex_lines.append(row + r" \\")
    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    tex_path.write_text("\n".join(tex_lines))
    print(f"LaTeX saved → {tex_path}", flush=True)

    # --- Console table ---
    print(f"\n{'='*90}", flush=True)
    print(f"  Budget Sweep: Revenue by Method, Graph, and k", flush=True)
    print(f"{'='*90}", flush=True)
    hdr = f"  {'Method':<22}"
    for g, k in cols:
        gs = g.replace("rice_facebook","rice_fb")
        hdr += f"  {(gs[:5]+'/k'+str(k)):>9}"
    print(hdr, flush=True)
    print(f"  {'─'*88}", flush=True)
    for grp_label, methods in GROUPS:
        print(f"  [{grp_label}]", flush=True)
        for m in methods:
            lbl = METHOD_LABELS.get(m, m)
            row = f"    {lbl:<20}"
            for g, k in cols:
                v = sweep_results.get((g, k), {}).get(m)
                row += f"  {v:>9.2f}" if v is not None else f"  {'—':>9}"
            print(row, flush=True)
    print(f"{'='*90}", flush=True)
    print("\n[continue_rice_sweep] DONE.", flush=True)

    return sweep_results


if __name__ == "__main__":
    main()
