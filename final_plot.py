import pandas as pd
import matplotlib.pyplot as plt

# Load data
baseline = pd.read_csv("kde_baseline.csv")
ppo = pd.read_csv("kde_ppo.csv")
ttm = pd.read_csv("kde_ttm.csv")

# Cumulative profit
baseline_profit = baseline["total_revenue"].cumsum()
ppo_profit = ppo["total_revenue"].cumsum()
ttm_profit = ttm["total_revenue"].cumsum()

# Plot
plt.figure()

plt.plot(baseline_profit, label="Baseline (80-20)")
plt.plot(ppo_profit, label="PPO")
plt.plot(ttm_profit, label="TTM")

plt.xlabel("Time Steps")
plt.ylabel("Cumulative Profit")
plt.title("Profit Comparison: Baseline vs PPO vs TTM (KDE Demand)")
plt.legend()

plt.show()