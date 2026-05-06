import os
import pandas as pd

# Run 1000 steps to see if it gets stuck
os.system("python -m scheduler -a EVAL -c configs/nyc.yaml -p EIGHTYTWENTY -o test_baseline.csv")

df = pd.read_csv("test_baseline.csv")
print("Completed jobs over time:")
print(df["completed"].iloc[100:120].tolist())
print(df["completed"].iloc[500:520].tolist())
print(df["completed"].iloc[900:920].tolist())
