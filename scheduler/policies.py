"""Built-in Fleet Scheduling Policies.
These classes can be extended for future research.
"""

from enum import Enum
from typing import Dict
from collections import deque

import argparse
import datetime
import json
import logging
import pickle
import random

import gymnasium as gym
import numpy
import yaml
import numpy as np

from scipy import stats

from simulator.job import *
from simulator.vehicle import *
from simulator.charger import *
from simulator.demand import *
from simulator.simulator import *

import stable_baselines3


# ------------------------------------------------------------------
# Base Policy
# ------------------------------------------------------------------
class SchedulePolicy:
    """Abstract Policy Class."""

    def __init__(self) -> None:
        pass

    def schedule(self, observation: numpy.array, info: Dict) -> numpy.array:
        raise NotImplemented


# ------------------------------------------------------------------
# Baseline: 20–80 Rule
# ------------------------------------------------------------------
class EightyTwentyPolicy(SchedulePolicy):
    """Charge vehicles at maximum available rate to 80% SoC, vehicles service
    demand until SoC drops below 20%, at which point they return to the
    nearest charger.
    """

    def __init__(self):
        super().__init__()

    def schedule(self, observation: numpy.array, info: Dict) -> numpy.array:
        fleet_size = len(info["fleet"])
        action = numpy.zeros((fleet_size, 2))
        for v in range(fleet_size):
            if observation[v, 1] < 0.2:
                action[v, 0] = 72.1
                action[v, 1] = 72.1
        return action


# ------------------------------------------------------------------
# TTM-Enhanced Policy (Future-Aware, Same State Size)
# ------------------------------------------------------------------
class SimpleTTM:
    """Lightweight TTM-like predictor using moving average."""

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


class TTMEnhancedPolicy(SchedulePolicy):
    """Uses TTM to predict future SoH and SoC.
    Fuses current and predicted values by averaging.
    """

    def __init__(self):
        super().__init__()
        self.ttm_soh = SimpleTTM(window_size=3)
        self.ttm_soc = SimpleTTM(window_size=3)

    def schedule(self, observation: numpy.array, info: Dict) -> numpy.array:
        observation = observation.reshape((len(info["fleet"]), 2))

        num_vehicles = observation.shape[0]
        action = numpy.zeros((num_vehicles, 2))

        for v in range(num_vehicles):
            current_soh = observation[v, 0]
            current_soc = observation[v, 1]

            # Update TTM history
            self.ttm_soh.update(v, current_soh)
            self.ttm_soc.update(v, current_soc)

            # Predict future values
            pred_soh = self.ttm_soh.predict(v)
            pred_soc = self.ttm_soc.predict(v)

            if pred_soh is None:
                pred_soh = current_soh
            if pred_soc is None:
                pred_soc = current_soc

            # Average fusion
            fused_soc = 0.5 * (current_soc + pred_soc)

            if fused_soc < 0.2:
                action[v, 0] = 72.1
                action[v, 1] = 72.1

        return action


# ------------------------------------------------------------------
# DNN Policy (PPO-trained) — NO hard-coded charging mask
# ------------------------------------------------------------------
class DnnPolicy(SchedulePolicy):
    """PPO-trained policy loaded using Stable-Baselines3.
    
    The PPO network's output is used directly. No manual overrides
    are applied so that PPO can learn a genuinely different strategy
    from the baseline 80-20 rule.
    """

    def __init__(self, weights: str) -> None:
        super().__init__()
        self.model = stable_baselines3.PPO.load(weights)

    def schedule(self, observation, info):
        action, _ = self.model.predict(observation, deterministic=True)

        fleet_size = len(info["fleet"])
        action = np.array(action).reshape((fleet_size, 2))
        action = np.abs(action)

        # Scale action[1] to charger power range (0 → max kW)
        action[:, 0] = np.clip(action[:, 0], 0.0, 1.0)
        action[:, 1] = action[:, 1] * 10.0  # match reference repo scaling

        return action


# ------------------------------------------------------------------
# Data Logger (aligned with reference repo)
# ------------------------------------------------------------------
class DataLogger:
    """Get data for plots — matches reference repo format."""

    def __init__(self, logfile, fleet_size=50):
        self.fleet_size = fleet_size
        self.csvfile = open(logfile, "w")
        self.csvfile.write("profit,total_power,completed,")
        self.csvfile.write(",".join([f"soh{i}" for i in range(self.fleet_size)]))
        self.csvfile.write(",")
        self.csvfile.write(",".join([f"status{i}" for i in range(self.fleet_size)]))
        self.csvfile.write("\n")

        self.p_old = [72.1] * self.fleet_size
        self.retired = [0] * self.fleet_size

    def write(self, info):
        total_power = 0
        p_curr = []
        soh_curr = []
        state = []

        for v in range(self.fleet_size):
            p_curr.append(info["fleet"][v]["battery"]["soc"] * 72.1)
            total_power += max(0, p_curr[-1] - self.p_old[v])

            if info["fleet"][v]["battery"]["actual_capacity"] / 72.1 <= 0.8:
                self.retired[v] = 1

            soh_curr.append(
                info["fleet"][v]["battery"]["actual_capacity"]
                / info["fleet"][v]["battery"]["initial_capacity"]
            )
            state.append(1 if info["fleet"][v]["status"] == "RECOVERY" else 0)

        self.p_old = p_curr

        # Per-step profit: sum fares from in-progress jobs
        # (only from non-retired vehicles)
        profit = 0
        for j in info.get("inprogress", []):
            vehicle_id = j.get("vehicle")
            if vehicle_id is not None and vehicle_id < self.fleet_size:
                if self.retired[vehicle_id] < 1:
                    profit += j["fare"]
            else:
                profit += j["fare"]

        completed = info["completed"]
        entry = f"{profit},{total_power},{completed},"

        for i in range(self.fleet_size):
            entry += f"{soh_curr[i]},"
        entry += ",".join([f"{state[i]}" for i in range(self.fleet_size)])
        self.csvfile.write(entry + "\n")

    def close(self):
        self.csvfile.close()