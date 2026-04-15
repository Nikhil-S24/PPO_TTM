import yaml
import numpy as np
from simulator.simulator import TaxiFleetSimulator

cfg = yaml.safe_load(open("configs/nyc.yaml"))

def run(mode):
    env = TaxiFleetSimulator(cfg)
    obs, info = env.reset()
    done = False
    while not done:
        fleet_size = len(info["fleet"])
        if mode == "zero":
            action = np.zeros((fleet_size, 2), dtype=np.float32)
        elif mode == "charge":
            action = np.ones((fleet_size, 2), dtype=np.float32)
        else:
            action = np.random.rand(fleet_size, 2).astype(np.float32)
        obs, reward, done, _, info = env.step(action)
    return info["total_revenue"]

print("zero  :", run("zero"))
print("charge:", run("charge"))
print("random:", run("random"))