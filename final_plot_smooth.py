import pandas as pd
import matplotlib.pyplot as plt

# Load data
baseline = pd.read_csv("kde_baseline.csv")
ppo = pd.read_csv("kde_ppo.csv")
ttm = pd.read_csv("kde_ttm.csv")

# Rolling smoothing (VERY IMPORTANT)
window = 50

baseline_profit = baseline["total_revenue"].rolling(window).mean()
ppo_profit = ppo["total_revenue"].rolling(window).mean()
ttm_profit = ttm["total_revenue"].rolling(window).mean()

# Plot
plt.figure(figsize=(10, 6))

plt.plot(baseline_profit, label="Baseline (80-20)", linewidth=2)
plt.plot(ppo_profit, label="PPO", linewidth=2)
plt.plot(ttm_profit, label="TTM", linewidth=2)

plt.xlabel("Time Steps")
plt.ylabel("Profit")
plt.title("Profit Comparison (Smoothed - KDE Demand)")
plt.legend()

plt.grid()

plt.show()