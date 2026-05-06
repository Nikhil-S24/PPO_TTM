import os
import pandas as pd

for f in os.listdir("."):
    if f.endswith(".csv"):
        try:
            df = pd.read_csv(f)
            if "total_revenue" in df.columns:
                print(f"{f}: {df['total_revenue'].iloc[-1]}")
        except Exception as e:
            pass
