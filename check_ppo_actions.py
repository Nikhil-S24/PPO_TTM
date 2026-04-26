import yaml
import numpy as np
from simulator.simulator import TaxiFleetSimulator
from stable_baselines3 import PPO

cfg = yaml.safe_load(open("configs/nyc.yaml"))
env = TaxiFleetSimulator(cfg)
model = PPO.load("ppo_final.pt")
obs, info = env.reset()

a0, a1 = [], []
total = 0
charge_cmd = 0
steps = 2000

for _ in range(steps):
    act, _ = model.predict(obs, deterministic=True)
    act = np.array(act).reshape((len(info["fleet"]), 2))
    a0.extend(act[:, 0].tolist())
    a1.extend(act[:, 1].tolist())
    total += act.shape[0]
    charge_cmd += int((act[:, 0] > 0.5).sum())
    obs, _, done, _, info = env.step(act)
    if done:
        break

print("a0_mean", float(np.mean(a0)))
print("a0_max", float(np.max(a0)))
print("pct_charge_cmd_gt_0.5", float(charge_cmd / total))
print("a1_mean", float(np.mean(a1)))
print("a1_p95", float(np.percentile(a1, 95)))
print("pct_a1_ge_0.2", float(np.mean(np.array(a1) >= 0.2)))
