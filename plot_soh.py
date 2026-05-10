"""Plot State of Health (SoH) degradation across 3 strategies."""

import csv
import numpy as np
import matplotlib.pyplot as plt
import os


def load_soh(filename, fleet_size=50):
    """Load SoH data from CSV. Returns median, 25th, and 75th percentile over time."""
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        fields = reader.fieldnames

        # Detect column naming: 'soh0' (new) or 'soh_0' (old)
        if "soh0" in fields:
            prefix = "soh"
        elif "soh_0" in fields:
            prefix = "soh_"
        else:
            raise ValueError(f"CSV {filename} has no SoH columns")

        medians = []
        p25 = []
        p75 = []

        for idx, datum in enumerate(reader):
            if idx % 24 == 0:  # sample daily
                soh_values = []
                for v in range(fleet_size):
                    soh_values.append(float(datum[f"{prefix}{v}"]))
                soh = np.array(soh_values)
                medians.append(np.percentile(soh, 50))
                p25.append(np.percentile(soh, 25))
                p75.append(np.percentile(soh, 75))

    t = np.arange(len(medians)) / 365.0  # days → years
    return t, np.array(medians), np.array(p25), np.array(p75)


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
    found_any = False

    for label, filename in files.items():
        if not os.path.exists(filename):
            print(f"  Skipping {label} ({filename} not found)")
            continue

        found_any = True
        print(f"Plotting {label} from {filename}...")
        t, median, p25, p75 = load_soh(filename)

        ax.fill_between(t, p25, p75, alpha=0.2, color=colors[label])
        ax.plot(t, median, label=label, linewidth=2, color=colors[label])

        print(f"  -> Final median SoH: {median[-1]:.4f}")

    if not found_any:
        print("No CSV files found! Run the simulations first.")
        return

    ax.set_xlabel("Years")
    ax.set_ylabel(r"State of Health $\bar{Q}_v(t)/\bar{Q}_v(0)$")
    ax.set_title("Battery Degradation Comparison")
    ax.set_ylim(0.70, 1.0)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("soh_comparison_plot.png", dpi=200)
    print("\nSaved plot to soh_comparison_plot.png")
    plt.show()


if __name__ == "__main__":
    main()