"""Diagnose why baseline completes so few jobs."""
import csv
import numpy as np

for name in ['baseline_5y.csv', 'ppo_5y.csv', 'ppottm_5y.csv']:
    print(f"\n=== {name} ===")
    with open(name) as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames

        # Find status columns
        status_cols = [c for c in fields if c.startswith('status') or c.startswith('state')]
        soh_cols = [c for c in fields if c.startswith('soh')]
        
        rows = list(reader)
        
        # Check status at step 100, 1000, 10000, and last
        for idx in [100, 1000, 10000, len(rows)-1]:
            row = rows[idx]
            statuses = {}
            for sc in status_cols:
                val = row[sc]
                statuses[val] = statuses.get(val, 0) + 1
            
            col = 'profit' if 'profit' in fields else 'total_revenue'
            cumulative = sum(float(rows[i][col]) for i in range(idx+1))
            print(f"  Step {idx}: completed={row['completed']}, revenue=${cumulative:,.0f}, statuses={statuses}")
