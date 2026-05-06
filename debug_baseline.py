import pandas as pd

df = pd.read_csv("output_baseline_real.csv")
states = [col for col in df.columns if col.startswith("state_")]

print("Completed jobs over time:")
print(df["completed"].head(20).tolist())
print(df["completed"].iloc[100:120].tolist())
print(df["completed"].iloc[500:520].tolist())
print(df["completed"].iloc[1000:1020].tolist())

print("\nFinal states of vehicles:")
print(df[states].iloc[-1].value_counts())
