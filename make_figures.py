"""Generate publication figures from results/summary.csv."""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib.pyplot as plt

def main() -> None:
    results = Path("results")
    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    with (results / "summary.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    labels = [row["condition"].replace("_", "\n") for row in rows]
    means = [100 * float(row["mean_win_rate"]) for row in rows]
    low = [100 * float(row["ci95_low_win_rate"]) for row in rows]
    high = [100 * float(row["ci95_high_win_rate"]) for row in rows]
    errors = [[m-l for m,l in zip(means,low)], [h-m for m,h in zip(means,high)]]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = ["#276FBF" if row["condition"] == "baseline" else "#6C9BD2" for row in rows]
    ax.bar(range(len(rows)), means, yerr=errors, capsize=3, color=colors)
    ax.set_ylabel("Final win rate against random (%)")
    ax.set_xticks(range(len(rows)), labels, fontsize=8)
    ax.set_title("MENACE experimental conditions (mean and 95% CI, n=100)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "condition_win_rates.png", dpi=220)
    plt.close(fig)
    baseline = next(r for r in rows if r["condition"] == "baseline")
    off = next(r for r in rows if r["condition"] == "symmetry_off")
    metrics = ["Win rate (%)", "States", "Convergence games", "Policy stability (%)"]
    base_values = [100*float(baseline["mean_win_rate"]), float(baseline["mean_state_space_size"]), float(baseline["mean_games_to_convergence"]), 100*float(baseline["mean_policy_stability"])]
    off_values = [100*float(off["mean_win_rate"]), float(off["mean_state_space_size"]), float(off["mean_games_to_convergence"]), 100*float(off["mean_policy_stability"])]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2))
    for ax, metric, base, no_sym in zip(axes, metrics, base_values, off_values):
        ax.bar(["Symmetry", "No symmetry"], [base, no_sym], color=["#276FBF", "#E07A5F"])
        ax.set_title(metric, fontsize=10)
        ax.tick_params(axis="x", labelrotation=25, labelsize=8)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Symmetry reduction ablation (n=100 per condition)")
    fig.tight_layout()
    fig.savefig(figures / "symmetry_ablation.png", dpi=220)
    plt.close(fig)

if __name__ == "__main__":
    main()
