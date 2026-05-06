import pandas as pd
import matplotlib.pyplot as plt

def main():
    baseline = pd.read_csv("baseline_5y.csv")
    ppo = pd.read_csv("ppo_5y.csv")
    ttm = pd.read_csv("ttm_5y.csv")

    print("\n=== baseline total_revenue ===")
    print("last 5:", baseline["total_revenue"].tail(5).tolist())

    print("\n=== ppo total_revenue ===")
    print("last 5:", ppo["total_revenue"].tail(5).tolist())

    print("\n=== ttm total_revenue ===")
    print("last 5:", ttm["total_revenue"].tail(5).tolist())

    # Assuming 1 step = 1 hour, wait, let's see how many steps are there.
    # We will just scale x to 5 years
    steps = len(baseline)
    import numpy as np
    x_years = np.linspace(0, 5, steps)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(x_years, baseline["total_revenue"], label="Baseline", linewidth=2, color="#1f77b4")
    ax.plot(x_years, ppo["total_revenue"], label="PPO-RL", linewidth=2, color="#ff7f0e")
    
    # We label ttm_5y as PPO+TTM (our work) assuming that's what it is, or we'll check the graph
    ax.plot(x_years, ttm["total_revenue"], label="PPO+TTM (our work)", linewidth=2, color="#2ca02c")

    ax.set_xlabel("Years")
    ax.set_ylabel("Cumulative Revenue ($)")
    ax.set_title("5-Year Revenue Comparison")

    ax.legend(loc="upper left")
    
    # Format y-axis
    ax.ticklabel_format(style="sci", axis="y", scilimits=(7, 7))

    plt.tight_layout()
    plt.savefig("5y_comparison_plot.png", dpi=200)
    print("Saved to 5y_comparison_plot.png")

if __name__ == "__main__":
    main()
