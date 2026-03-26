import pandas as pd

def analyze(file):
    try:
        df = pd.read_csv(file)
        df = df.apply(pd.to_numeric, errors="coerce")
        
        if df.empty:
            print(f"\nResults for {file}: No data found (file is empty).")
            return

        
        total_profit = df["total_revenue"].iloc[-1]
        
        total_completed = df["completed"].iloc[-1]

        soh_cols = [col for col in df.columns if col.startswith("soh")]
        final_avg_soh = df[soh_cols].iloc[-1].mean()

        print(f"\nResults for {file}")
        print(f"Total Profit: {total_profit:.2f}")
        
        print(f"Total Completed Jobs: {total_completed}")
        print(f"Final Average SoH: {final_avg_soh:.4f}")
        
    except FileNotFoundError:
        print(f"\nCould not find file: {file}")
    except KeyError as e:
         print(f"\nError in {file}: Missing expected column {e}")


analyze("output_baseline.csv")
analyze("output_ppo.csv")
analyze("output_ttm.csv")