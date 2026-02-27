import pandas as pd
import numpy as np

def compute_metrics(csv_path):
    df = pd.read_csv(csv_path)

    metrics = {}

    # Number of timesteps
    metrics["timesteps"] = len(df)

    # Cumulative profit
    metrics["cumulative_profit"] = df["profit"].sum()

    # Peak charging power
    metrics["peak_power"] = df["total_power"].max()

    # Final average SoH
    soh_cols = [col for col in df.columns if col.startswith("soh")]
    final_avg_soh = df[soh_cols].iloc[-1].mean()
    metrics["final_avg_soh"] = final_avg_soh

    return metrics


baseline_metrics = compute_metrics("output_baseline.csv")
ttm_metrics = compute_metrics("output_ttm.csv")

print("\n=== BASELINE METRICS ===")
for k, v in baseline_metrics.items():
    print(f"{k}: {v}")

print("\n=== TTM METRICS ===")
for k, v in ttm_metrics.items():
    print(f"{k}: {v}")
