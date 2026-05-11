"""Plot cumulative revenue comparison — handles both old and new CSV formats."""

import csv
import numpy as np
import matplotlib.pyplot as plt
import os


def load_cumulative_revenue(filename):
    """Load revenue from CSV. Handles both 'profit' (new) and 'total_revenue' (old) headers."""
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        fields = reader.fieldnames

        # Detect format
        if "profit" in fields:
            # New format: per-step profit, needs cumulative sum
            revenue = []
            r = 0
            for idx, datum in enumerate(reader):
                r += float(datum["profit"])
                if idx % 24 == 0:
                    revenue.append(r)
        elif "total_revenue" in fields:
            # Old format: already cumulative
            revenue = []
            for idx, datum in enumerate(reader):
                if idx % 24 == 0:
                    revenue.append(float(datum["total_revenue"]))
        else:
            raise ValueError(f"CSV {filename} has no 'profit' or 'total_revenue' column")

    t = np.arange(len(revenue))  # days
    return t, np.array(revenue)


def main():
    files = {
        "Baseline (80-20)": ["test_baseline.csv"],
        "PPO-RL": ["test_ppo.csv"],
        "PPO+TTM": ["test_ppo_ttm.csv"],
    }

    colors = {
        "Baseline (80-20)": "#2ca02c" ,
        "PPO-RL": "#1f77b4",
        "PPO+TTM": "#ff7f0e" ,
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))
    found_any = False

    for label, candidates in files.items():
        filename = None
        for f in candidates:
            if os.path.exists(f):
                filename = f
                break

        if filename:
            found_any = True
            print(f"Plotting {label} from {filename}...")
            t, revenue = load_cumulative_revenue(filename)
            ax.plot(t, revenue, label=label, linewidth=2, color=colors[label])
            print(f"  -> Data points: {len(t)}, Final Revenue: ${revenue[-1]:,.2f}")
        else:
            print(f"  Skipping {label} (no file found)")

    if not found_any:
        print("No CSV files found! Run the simulations first.")
        return

    ax.set_xlabel("Days")
    ax.set_ylabel("Cumulative Revenue ($)")
    ax.set_title("Revenue Comparison")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig("revenue_1m_comparison.png", dpi=200)
    print("\nSaved plot to revenue_1m_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()