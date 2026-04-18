import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Load data
# -------------------------------
baseline = pd.read_csv("kde_baseline.csv")
ppo = pd.read_csv("kde_ppo.csv")
ttm = pd.read_csv("kde_ttm.csv")

# -------------------------------
# Debug check (VERY IMPORTANT)
# -------------------------------
print("\n=== FINAL VALUES ===")
print("Baseline:", baseline["total_revenue"].iloc[-1])
print("PPO     :", ppo["total_revenue"].iloc[-1])
print("TTM     :", ttm["total_revenue"].iloc[-1])

# -------------------------------
# Convert steps → years (5-year simulation)
# -------------------------------
steps = len(baseline)
years = np.linspace(0, 5, steps)

# -------------------------------
# Plot cumulative revenue
# -------------------------------
plt.figure(figsize=(10, 6))

plt.plot(years, baseline["total_revenue"], label="Baseline (80-20)", linewidth=2)
plt.plot(years, ppo["total_revenue"], label="PPO-RL", linewidth=2)
plt.plot(years, ttm["total_revenue"], label="TTM", linewidth=2)

# -------------------------------
# Labels & styling
# -------------------------------
plt.xlabel("Years")
plt.ylabel("Cumulative Revenue ($)")
plt.title("Cumulative Revenue over 5 Years")

# Scientific notation for large values
plt.ticklabel_format(style='sci', axis='y', scilimits=(6, 6))

plt.legend()
plt.grid()

plt.show()