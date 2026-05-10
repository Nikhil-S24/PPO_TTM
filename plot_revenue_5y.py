"""Plot 5-Year Cumulative Revenue Comparison across 3 strategies."""

import csv
import numpy as np
import matplotlib.pyplot as plt
import os


def load_cumulative_revenue(filename):
    """Load revenue from CSV. Handles both 'profit' and 'total_revenue' headers."""
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        fields = reader.fieldnames

        if "profit" in fields:
            revenue = []
            r = 0
            for idx, datum in enumerate(reader):
                r += float(datum["profit"])
                if idx % (24 * 7) == 0:  # sample weekly
                    revenue.append(r)
        elif "total_revenue" in fields:
            revenue = []
            for idx, datum in enumerate(reader):
                if idx % (24 * 7) == 0:  # sample weekly
                    revenue.append(float(datum["total_revenue"]))
        else:
            raise ValueError(f"CSV {filename} has no 'profit' or 'total_revenue' column")

    t = np.linspace(0, len(revenue) / 52.0, len(revenue))  # weeks → years
    return t, np.array(revenue)


def main():
    files = {
        "Baseline (80-20)": "baseline_5y.csv",
        "PPO-RL": "ppo_5y.csv",
        "PPO+TTM (Our Work)": "ppottm_5y.csv",
    }

    colors = {
        "Baseline (80-20)": "#1f77b4",
        "PPO-RL": "#ff7f0e",
        "PPO+TTM (Our Work)": "#2ca02c",
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for label, filename in files.items():
        if os.path.exists(filename):
            print(f"Plotting {label} from {filename}...")
            t, revenue = load_cumulative_revenue(filename)
            ax.plot(t, revenue, label=label, linewidth=2, color=colors[label])
            print(f"  -> Final Revenue: ${revenue[-1]:,.2f}")
        else:
            print(f"  Skipping {label} ({filename} not found)")

    ax.set_xlabel("Years")
    ax.set_ylabel("Cumulative Revenue ($)")
    ax.set_title("5-Year Revenue Comparison")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig("revenue_5y_plot.png", dpi=200)
    print("\nSaved plot to revenue_5y_plot.png")
    plt.show()


if __name__ == "__main__":
    main()
