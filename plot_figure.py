import pandas as pd
import matplotlib.pyplot as plt

# Load data
baseline = pd.read_csv("output_baseline.csv")
ttm = pd.read_csv("output_ttm.csv")
ppo = pd.read_csv("output_ppo.csv")

# Extract revenue column
baseline_profit = baseline["total_revenue"]
ttm_profit = ttm["total_revenue"]
ppo_profit = ppo["total_revenue"]

# Plot
plt.figure(figsize=(10,6))

plt.plot(baseline_profit, label="Baseline (80-20)")
plt.plot(ttm_profit, label="TTM")
plt.plot(ppo_profit, label="PPO")

plt.xlabel("Time Steps")
plt.ylabel("Total Revenue")
plt.title("Profit Comparison of Charging Strategies")

plt.legend()
plt.grid()

plt.show()