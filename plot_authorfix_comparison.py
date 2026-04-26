import pandas as pd
import matplotlib.pyplot as plt


# Load data
baseline = pd.read_csv("kde_baseline_1day_LOCKED.csv")
ppo = pd.read_csv("kde_ppo_authorfix_1day_LOCKED.csv")
ttm = pd.read_csv("kde_ttm_1day_LOCKED.csv")

print("\n=== baseline total_revenue ===")
print("first 5:", baseline["total_revenue"].head(5).tolist())
print("last 5 :", baseline["total_revenue"].tail(5).tolist())

print("\n=== ppo total_revenue ===")
print("first 5:", ppo["total_revenue"].head(5).tolist())
print("last 5 :", ppo["total_revenue"].tail(5).tolist())

print("\n=== ttm total_revenue ===")
print("first 5:", ttm["total_revenue"].head(5).tolist())
print("last 5 :", ttm["total_revenue"].tail(5).tolist())

# Rolling smoothing
window = 50
baseline_profit = baseline["total_revenue"].rolling(window).mean()
ppo_profit = ppo["total_revenue"].rolling(window).mean()
ttm_profit = ttm["total_revenue"].rolling(window).mean()

# Plot
plt.figure(figsize=(10, 6))
plt.plot(baseline_profit, label="Baseline (80-20)", linewidth=2)
plt.plot(ppo_profit, label="PPO (Author Fix)", linewidth=2)
plt.plot(ttm_profit, label="TTM", linewidth=2)
plt.xlabel("Time Steps")
plt.ylabel("Profit")
plt.title("Profit Comparison")
plt.legend()
plt.grid()
plt.show()
