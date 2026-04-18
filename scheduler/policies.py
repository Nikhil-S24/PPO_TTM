"""Built-in Fleet Scheduling Policies."""

from enum import Enum
from typing import Dict
from collections import deque

import numpy as np
import stable_baselines3

from simulator.job import *
from simulator.vehicle import *
from simulator.charger import *
from simulator.demand import *
from simulator.simulator import *


# ------------------------------------------------------------------
# Base Policy
# ------------------------------------------------------------------
class SchedulePolicy:
    def schedule(self, observation: np.array, info: Dict) -> np.array:
        raise NotImplemented


# ------------------------------------------------------------------
# Baseline: 20–80 Rule
# ------------------------------------------------------------------
class EightyTwentyPolicy(SchedulePolicy):

    def schedule(self, observation: np.array, info: Dict) -> np.array:

        observation = observation.reshape((len(info["fleet"]), 2))
        action = np.zeros((len(info["fleet"]), 2))

        for v in range(len(info["fleet"])):
            if observation[v, 1] < 0.2:
                action[v, 0] = 1.0
                action[v, 1] = 72.1

        return action


# ------------------------------------------------------------------
# Simple TTM
# ------------------------------------------------------------------
class SimpleTTM:
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.history = {}

    def update(self, vehicle_id, value):
        if vehicle_id not in self.history:
            self.history[vehicle_id] = deque(maxlen=self.window_size)
        self.history[vehicle_id].append(value)

    def predict(self, vehicle_id):
        if vehicle_id not in self.history:
            return None
        return sum(self.history[vehicle_id]) / len(self.history[vehicle_id])


# ------------------------------------------------------------------
# TTM Policy
# ------------------------------------------------------------------
class TTMEnhancedPolicy(SchedulePolicy):

    def __init__(self):
        self.ttm_soh = SimpleTTM()
        self.ttm_soc = SimpleTTM()

    def schedule(self, observation: np.array, info: Dict) -> np.array:

        observation = observation.reshape((len(info["fleet"]), 2))
        action = np.zeros((len(info["fleet"]), 2))

        for v in range(len(info["fleet"])):

            soh = observation[v, 0]
            soc = observation[v, 1]

            self.ttm_soh.update(v, soh)
            self.ttm_soc.update(v, soc)

            pred_soh = self.ttm_soh.predict(v) or soh
            pred_soc = self.ttm_soc.predict(v) or soc

            fused_soc = 0.5 * (soc + pred_soc)

            if fused_soc < 0.2:
                action[v, 0] = 1.0
                action[v, 1] = 72.1

        return action


# ------------------------------------------------------------------
# PPO Policy
# ------------------------------------------------------------------
class DnnPolicy(SchedulePolicy):

    def __init__(self, weights: str):
        self.model = stable_baselines3.PPO.load(weights)

    def schedule(self, observation, info):

        action, _ = self.model.predict(observation, deterministic=True)

        fleet_size = len(info["fleet"])
        action = np.array(action).reshape((fleet_size, 2))

        action[:, 0] = np.clip(action[:, 0], 0.0, 1.0)
        action[:, 1] = np.clip(action[:, 1], 0.0, 1.0)

        return action


# ------------------------------------------------------------------
# Data Logger (🔥 FIXED)
# ------------------------------------------------------------------
class DataLogger:

    def __init__(self, logfile, fleet_size):
        self.fleet_size = fleet_size
        self.csvfile = open(logfile, "w")

        # 🔥 ADD profit column FIRST
        header = "profit,total_revenue,total_power,completed,"
        header += ",".join([f"soh_{i}" for i in range(self.fleet_size)]) + ","
        header += ",".join([f"state_{i}" for i in range(self.fleet_size)])

        self.csvfile.write(header + "\n")

        self.prev_revenue = 0  # 🔥 IMPORTANT

        self.p_old = [72.1] * self.fleet_size
        self.retired = [0] * self.fleet_size

    def write(self, info):

        total_power = 0
        p_curr = []
        soh_curr = []
        state = []

        for v in range(self.fleet_size):

            soc_power = info["fleet"][v]["battery"]["soc"] * 72.1
            p_curr.append(soc_power)

            total_power += max(0, soc_power - self.p_old[v])

            if info["fleet"][v]["battery"]["actual_capacity"] / 72.1 <= 0.8:
                self.retired[v] = 1

            soh = (
                info["fleet"][v]["battery"]["actual_capacity"]
                / info["fleet"][v]["battery"]["initial_capacity"]
            )
            soh_curr.append(soh)

            state.append(1 if info["fleet"][v]["status"] == "RECOVERY" else 0)

        self.p_old = p_curr

        # 🔥 FIXED PROFIT CALCULATION
        current_revenue = info["total_revenue"]
        profit = current_revenue - self.prev_revenue
        self.prev_revenue = current_revenue

        completed = info["completed"]

        entry = f"{profit},{current_revenue},{total_power},{completed},"

        for i in range(self.fleet_size):
            entry += f"{soh_curr[i]},"

        entry += ",".join([f"{state[i]}" for i in range(self.fleet_size)])

        self.csvfile.write(entry + "\n")

    def close(self):
        self.csvfile.close()