import pandas as pd
import matplotlib.pyplot as plt

baseline = pd.read_csv("baseline_5y.csv")
ppo = pd.read_csv("ppo_5y.csv")
ttm = pd.read_csv("ttm_5y.csv")

print("\n=== baseline total_revenue ===")
print("rows:", len(baseline))
print("last 5:", baseline["total_revenue"].tail(5).tolist())

print("\n=== ppo total_revenue ===")
print("rows:", len(ppo))
print("last 5:", ppo["total_revenue"].tail(5).tolist())

print("\n=== ttm total_revenue ===")
print("rows:", len(ttm))
print("last 5:", ttm["total_revenue"].tail(5).tolist())

window = 500
baseline_s = baseline["total_revenue"].rolling(window).mean()
ppo_s = ppo["total_revenue"].rolling(window).mean()
ttm_s = ttm["total_revenue"].rolling(window).mean()

# Convert x-axis to years (delta t = 3600 sec => 8760 steps/year)
x_years = baseline.index / 8760.0

plt.figure(figsize=(10, 6))

scale = 1000.0
plt.plot(x_years, baseline_s * scale, label="Baseline (80-20)", linewidth=2)
plt.plot(x_years, ppo_s * scale, label="PPO", linewidth=2)
plt.plot(x_years, ttm_s * scale, label="TTM", linewidth=2)

plt.xlabel("Years")
plt.ylabel("Profit")
plt.title("5-Year Profit Comparison (Smoothed)")
plt.legend()
plt.grid()
plt.show()