"""Budget-constrained revenue vs. capital, five networks.

Values are the monotone envelope R(k) = max_{k' <= k} R(k'), applied
uniformly to every method (a seller with capital B0 may run any
sub-budget deployment and withhold the surplus).

Protocol: BudgetRevenueEnv, c=0.3, B0=k*c, seeds [42,123,7],
SKIP-never-reprice. Source: results/logs/budget_sweep_all_networks.json.

"Rev-GNN-LSTM" here is the FF+BA-trained checkpoint
(rev_gnn_lstm_densemix.pt, sha 0b549f93) — NOT the FF-trained released
model, which has different numbers.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = "results/figures/budget_comparison"
K = [5, 10, 15, 20, 30, 40]

DATA = {
    "polblogs (n=1222, avg deg 27.4)": {
        "IE-Strategy":     [178.2, 316.0, 386.3, 437.9, 498.0, 499.6],
        "Greedy-Discount": [ 50.4,  98.5, 139.4, 173.4, 254.9, 392.0],
        "Cal-DP":          [ 96.1, 187.6, 347.1, 553.1, 648.7, 648.7],
        "Rev-GNN-LSTM":    [ 30.3,  79.1, 173.5, 488.7, 620.6, 620.6],
    },
    "FF-1000 (n=1000, avg deg 4.7)": {
        "IE-Strategy":     [ 84.1, 140.2, 170.4, 195.1, 245.6, 257.0],
        "Greedy-Discount": [ 52.5,  96.7, 345.7, 422.6, 448.3, 451.8],
        "Cal-DP":          [317.9, 434.5, 443.7, 443.7, 443.7, 443.7],
        "Rev-GNN-LSTM":    [296.0, 385.9, 397.4, 399.7, 399.7, 399.7],
    },
    "Rice-Facebook (n=443, avg deg 44.3)": {
        "IE-Strategy":     [  0.9,   6.1,  63.5, 103.9, 141.9, 141.9],
        "Greedy-Discount": [  0.2,   0.2,   0.4,   1.1,   1.8,  10.7],
        "Cal-DP":          [  0.8,   6.1, 156.2, 225.6, 225.7, 225.7],
        "Rev-GNN-LSTM":    [  0.8,   2.9, 141.2, 214.1, 214.1, 214.1],
    },
    "Modular-FF (n=500, avg deg 16.6)": {
        "IE-Strategy":     [  0.9,   8.7,  42.9,  64.0, 105.8, 145.7],
        "Greedy-Discount": [  0.0,   0.4,   0.5,   3.2,   6.8,  28.1],
        "Cal-DP":          [  1.3,  31.0,  80.7, 196.7, 227.4, 227.4],
        "Rev-GNN-LSTM":    [  1.7,  35.3,  84.4, 169.5, 212.4, 212.4],
    },
    "FF-2000 (n=2000, avg deg 5.1)": {
        "IE-Strategy":     [141.9, 221.9, 283.0, 323.9, 412.1, 458.8],
        "Greedy-Discount": [ 35.1, 105.0, 178.4, 286.0, 675.3, 866.5],
        "Cal-DP":          [710.7, 874.5, 905.6, 911.6, 911.6, 911.6],
        "Rev-GNN-LSTM":    [180.7, 691.3, 736.9, 830.1, 838.1, 838.1],
    },
}

# One fixed colour/marker/linestyle per method, identical in every panel.
STYLE = {
    "IE-Strategy":     dict(color="#1f77b4", marker="o", ls="--"),
    "Greedy-Discount": dict(color="#7f7f7f", marker="s", ls="--"),
    "Cal-DP":          dict(color="#2ca02c", marker="^", ls="-"),
    "Rev-GNN-LSTM":    dict(color="#ff7f0e", marker="v", ls="-"),
}
ORDER = list(STYLE.keys())
XLABEL = "budget index $k$  ($B_0 = k\\cdot c$, $c=0.3$)"

plt.rcParams.update({
    "font.size": 14, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _draw(ax, series, title, title_fs):
    for m in ORDER:
        s = STYLE[m]
        ax.plot(K, series[m], color=s["color"], marker=s["marker"],
                ls=s["ls"], lw=2.4, ms=7, label=m)
    ax.set_title(title, fontsize=title_fs)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel("revenue")
    ax.set_xticks(K)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    written = []

    # combined 2x3 panel, shared legend in the empty cell
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    for ax, (title, series) in zip(axes.flat, DATA.items()):
        _draw(ax, series, title, 15)
    axes.flat[-1].axis("off")
    handles = [plt.Line2D([0], [0], color=STYLE[m]["color"],
                          marker=STYLE[m]["marker"], ls=STYLE[m]["ls"],
                          lw=2.6, ms=8, label=m) for m in ORDER]
    axes.flat[-1].legend(handles=handles, loc="center", frameon=False,
                         fontsize=15)
    fig.suptitle("Budget-constrained revenue vs. capital "
                 "(monotone deployment)", fontsize=17, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("png", "pdf"):
        p = f"{OUTDIR}/budget_all_networks.{ext}"
        fig.savefig(p, dpi=170)
        written.append(p)
    plt.close(fig)

    # one figure per network
    for title, series in DATA.items():
        f, ax = plt.subplots(figsize=(7.2, 5.2))
        _draw(ax, series, title, 13)
        ax.legend(fontsize=12, frameon=False)
        f.tight_layout()
        name = title.split(" (")[0].replace("-", "_").replace(" ", "_")
        for ext in ("png", "pdf"):
            p = f"{OUTDIR}/budget_{name}.{ext}"
            f.savefig(p, dpi=170)
            written.append(p)
        plt.close(f)

    for p in written:
        print(p)


if __name__ == "__main__":
    main()
