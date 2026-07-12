"""experiments/make_idea3_figures.py — Generate Idea 3 publication figures.

Thin script (<80 lines). All logic in src/utils/idea3_figures.py.

Usage:
  cd revmax-aaai2027 && source venv/bin/activate
  python experiments/make_idea3_figures.py \\
      --ff_json   results/logs/dp_upgrade_eval_ff.json \\
      --rice_json results/logs/dp_upgrade_eval_rice_lstm.json \\
      [--fig_dir  results/figures] \\
      [--log_dir  results/logs]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.helpers import set_seed
from src.utils.idea3_figures import make_idea3_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Idea 3 figures + table")
    parser.add_argument("--ff_json",   default="results/logs/dp_upgrade_eval_ff.json",
                        help="Path to FF n=1000 sweep results JSON")
    parser.add_argument("--rice_json", default="results/logs/dp_upgrade_eval_rice_lstm.json",
                        help="Path to Rice-FB sweep results JSON")
    parser.add_argument("--fig_dir",   default="results/figures",
                        help="Root figure directory (budget/ sub-directory will be created)")
    parser.add_argument("--log_dir",   default="results/logs",
                        help="Directory for LaTeX table output")
    args = parser.parse_args()

    set_seed(42)

    make_idea3_figures(
        ff_json=args.ff_json,
        rice_json=args.rice_json,
        fig_dir=args.fig_dir,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
