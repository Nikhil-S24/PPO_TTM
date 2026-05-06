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

# Convert steps to years
x_years = baseline.index / 8760.0   # 1 step = 1 hour

plt.figure(figsize=(10, 6))

plt.plot(x_years, baseline["total_revenue"], label="Baseline (80-20)", linewidth=2)
plt.plot(x_years, ppo["total_revenue"], label="PPO", linewidth=2)
plt.plot(x_years, ttm["total_revenue"], label="TTM", linewidth=2)

plt.xlabel("Years")
plt.ylabel("Cumulative Revenue ($)")
plt.title("5-Year Revenue Comparison")

plt.legend()
plt.grid()
plt.tight_layout()
plt.show()