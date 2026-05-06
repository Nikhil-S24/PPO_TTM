"""Extended smoke test: run 100 steps, verify jobs flow through the lifecycle."""
import yaml
from simulator.simulator import TaxiFleetSimulator
from scheduler.policies import EightyTwentyPolicy, DataLogger

with open("configs/nyc.yaml", "r") as f:
    config = yaml.safe_load(f)

env = TaxiFleetSimulator(config)
obs, info = env.reset()

policy = EightyTwentyPolicy()
logger = DataLogger("test_output.csv", config["fleet"]["size"])

print(f"Fleet size: {len(info['fleet'])}")
print(f"Obs shape: {obs.shape}")

for step in range(200):
    logger.write(info)
    action = policy.schedule(obs, info)
    obs, reward, done, truncated, info = env.step(action)
    
    if step % 20 == 0:
        n_inprogress = len(info["inprogress"])
        statuses = {}
        for v in info["fleet"]:
            s = v["status"]
            statuses[s] = statuses.get(s, 0) + 1
        print(
            f"Step {step}: completed={info['completed']}, "
            f"rejected={info['rejected']}, "
            f"inprogress={n_inprogress}, "
            f"statuses={statuses}"
        )

logger.close()

# Check the CSV
import csv
with open("test_output.csv", "r") as f:
    reader = csv.DictReader(f)
    total_profit = 0
    for row in reader:
        total_profit += float(row["profit"])
    print(f"\nCumulative profit from CSV: ${total_profit:,.2f}")
