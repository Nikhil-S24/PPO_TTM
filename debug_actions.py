import yaml
import numpy as np
from stable_baselines3 import PPO
from simulator.simulator import TaxiFleetSimulator

cfg = yaml.safe_load(open("configs/nyc.yaml"))
env = TaxiFleetSimulator(cfg)
obs, info = env.reset()

model = PPO.load("ppo_sb3_nyc_v9")

acts = []
term = False
trunc = False
i = 0

while i < 200 and not (term or trunc):
    a, _ = model.predict(obs, deterministic=True)
    acts.append(np.array(a))
    obs, r, term, trunc, info = env.step(a)
    i += 1

A = np.array(acts)
print("shape=", A.shape)
print("mean0=", float(A[:, :, 0].mean()))
print("mean1=", float(A[:, :, 1].mean()))
print("std0=", float(A[:, :, 0].std()))
print("std1=", float(A[:, :, 1].std()))