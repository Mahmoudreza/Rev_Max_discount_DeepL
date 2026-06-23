"""
experiments/figures/paper_tables.py

Generate LaTeX tables for the paper from logged CSV results.

Tables:
  Table 1: Main results (baselines vs GNN models, BA/WS/ER graphs)
  Table 2: Ablation: influence model
  Table 3: Ablation: encoder type
  Table 4: Ablation: reward function
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import csv
import json
from pathlib import Path
from typing import Dict, List


def load_results_csv(csv_path: Path) -> List[Dict]:
    """Load result rows from a CSV log file.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        List of row dicts.
    """
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def format_mean_std(mean: float, std: float, decimals: int = 3) -> str:
    """Format as 'mean ± std' for LaTeX tables.

    Args:
        mean: Mean value.
        std: Standard deviation.
        decimals: Decimal places.

    Returns:
        Formatted string.
    """
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(mean)} \\pm {fmt.format(std)}"


def generate_baseline_table(results: Dict[str, float]) -> str:
    """Generate LaTeX table for baseline results.

    Args:
        results: Dict strategy → mean revenue.

    Returns:
        LaTeX table string.
    """
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Revenue of Babaei et al. (2013) baselines on BA graph (n=500, 10 trials)}",
        r"\label{tab:baselines}",
        r"\begin{tabular}{lc}",
        r"\hline",
        r"Strategy & Revenue \\",
        r"\hline",
    ]
    for strategy, rev in results.items():
        name_map = {
            "ie_strategy": "IE-Strategy",
            "mu_discount": r"$\mu$-Discount",
            "greedy_discount": "Greedy-Discount",
            "sigma_discount": r"$\sigma$-Discount",
        }
        lines.append(f"{name_map.get(strategy, strategy)} & {rev:.4f} \\\\")
    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def main() -> None:
    """Generate all paper tables from available result logs."""
    log_dir = Path("results/logs")
    output_dir = Path("results/tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find baseline results
    baseline_csvs = list(log_dir.glob("baselines_*.csv")) if log_dir.exists() else []

    if not baseline_csvs:
        print("No baseline results found in results/logs/.")
        print("Run experiments/run_baselines.py first.")
        # Generate a stub table for demonstration
        stub_results = {
            "ie_strategy": 0.0,
            "mu_discount": 0.0,
            "greedy_discount": 0.0,
            "sigma_discount": 0.0,
        }
        table = generate_baseline_table(stub_results)
        table_path = output_dir / "tab1_baselines.tex"
        table_path.write_text(table)
        print(f"Stub table saved to {table_path}")
    else:
        # Load most recent results
        latest_csv = sorted(baseline_csvs)[-1]
        rows = load_results_csv(latest_csv)

        # Aggregate by strategy
        strategy_revenues = {}
        for row in rows:
            if "strategy" in row and "revenue" in row:
                strat = row["strategy"]
                try:
                    rev = float(row["revenue"])
                    if strat not in strategy_revenues:
                        strategy_revenues[strat] = []
                    strategy_revenues[strat].append(rev)
                except (ValueError, TypeError):
                    pass

        mean_results = {k: sum(v) / len(v) for k, v in strategy_revenues.items()}
        table = generate_baseline_table(mean_results)
        table_path = output_dir / "tab1_baselines.tex"
        table_path.write_text(table)
        print(f"Table saved to {table_path}")
        print(table)


if __name__ == "__main__":
    main()
