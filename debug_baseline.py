"""Deep diagnostic: check vehicle SoC + status at key points for baseline."""
import csv
import yaml
from simulator.simulator import TaxiFleetSimulator
from scheduler.policies import EightyTwentyPolicy, DataLogger
import numpy as np

with open('configs/nyc.yaml', 'r') as f:
    config = yaml.safe_load(f)

env = TaxiFleetSimulator(config)
obs, info = env.reset()
policy = EightyTwentyPolicy()

fleet_size = config['fleet']['size']

for step in range(8000):
    action = policy.schedule(obs, info)
    obs, reward, done, _, info = env.step(action)
    
    if step in [100, 1000, 3000, 5000, 6000, 6175, 6200, 7000]:
        obs_2d = obs.reshape((fleet_size, 2))
        socs = obs_2d[:, 1]
        sohs = obs_2d[:, 0]
        statuses = [v.status.name for v in env.fleet]
        status_counts = {}
        for s in statuses:
            status_counts[s] = status_counts.get(s, 0) + 1
        
        n_arrived = len(env.arrived)
        n_inprogress = len(env.inprogress)
        n_assigned = len(env.assigned)
        
        print(f"\nStep {step} (day {step//24}):")
        print(f"  SoC: min={socs.min():.4f}, max={socs.max():.4f}, mean={socs.mean():.4f}")
        print(f"  SoH: min={sohs.min():.6f}, max={sohs.max():.6f}")
        print(f"  Statuses: {status_counts}")
        print(f"  Jobs: arrived={n_arrived}, assigned={n_assigned}, inprogress={n_inprogress}")
        print(f"  Completed={env.completed}, Rejected={env.rejected}, Failed={env.failed}")
    
    if done:
        print(f"\nDone at step {step}")
        break
