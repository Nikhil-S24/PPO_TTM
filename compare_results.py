import pandas as pd

def analyze(file):
    df = pd.read_csv(file)
    df = df.apply(pd.to_numeric, errors="coerce")
    
    total_profit = df["total_revenue"].iloc[-1]
    total_power = df["total_power"].sum()
    total_completed = df["completed"].iloc[-1]

    soh_cols = [col for col in df.columns if col.startswith("soh")]
    final_avg_soh = df[soh_cols].iloc[-1].mean()

    print(f"\nResults for {file}")
    print(f"Total Profit: {total_profit:.2f}")
    print(f"Total Power Used: {total_power:.2f}")
    print(f"Total Completed Jobs: {total_completed}")
    print(f"Final Average SoH: {final_avg_soh:.4f}")

analyze("baseline.csv")
analyze("ppo.csv")
analyze("ppo_ttm.csv")