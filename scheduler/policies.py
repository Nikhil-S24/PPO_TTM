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

import coloredlogs
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
import torch


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
    """
    Vehicles charge when SoC < 20% and charge at max rate until 80%.
    """

    def __init__(self):
        super().__init__()

    def schedule(self, observation: numpy.array, info: Dict) -> numpy.array:
        action = numpy.zeros((len(info["fleet"]), 2))
        for v in range(len(info["fleet"])):
            if observation[v, 1] < 0.2:
                action[v, 0] = 1.0
                action[v, 1] = 72.1
        return action


# ------------------------------------------------------------------
# Simple TTM (Dummy / Moving Average)
# ------------------------------------------------------------------
class SimpleTTM:
    """
    Lightweight TTM-like predictor using moving average.
    Can later be replaced with real Tiny Time Mixer.
    """

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
# TTM-Enhanced Policy (Future-Aware, Same State Size)
# ------------------------------------------------------------------
class TTMEnhancedPolicy(SchedulePolicy):
    """
    Uses TTM to predict future SoH and SoC.
    Fuses current and predicted values by averaging.
    State dimension remains 2.
    """

    def __init__(self):
        super().__init__()
        self.ttm_soh = SimpleTTM(window_size=3)
        self.ttm_soc = SimpleTTM(window_size=3)

    def schedule(self, observation: numpy.array, info: Dict) -> numpy.array:
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

            # Fallback if prediction not available
            if pred_soh is None:
                pred_soh = current_soh
            if pred_soc is None:
                pred_soc = current_soc

            # Average fusion (KEY IDEA)
            fused_soh = 0.5 * (current_soh + pred_soh)
            fused_soc = 0.5 * (current_soc + pred_soc)

            # Charging decision based on fused state
            if fused_soc < 0.2:
                action[v, 0] = 1.0
                action[v, 1] = 72.1

        return action


# ------------------------------------------------------------------
# DNN Policy (PPO-trained)
# ------------------------------------------------------------------
class DnnPolicy(SchedulePolicy):
    """
    PPO-trained neural network policy.
    """

    def __init__(self, weights: str) -> None:
        super().__init__()
        self.dnn = torch.load(weights, weights_only=False).eval()

    def schedule(self, observation, info):
        with torch.no_grad():
            x = torch.from_numpy(observation).unsqueeze(0)
            action = self.dnn(x)[0].squeeze().cpu().numpy()
            
            # Clip charging decision between 0 and 1
            action[:, 0] = np.clip(action[:, 0], 0.0, 1.0)

            # Clip charging power between 0 and 72.1 kW
            action[:, 1] = np.clip(action[:, 1] * 10.0, 0.0, 72.1)
            return action


# ------------------------------------------------------------------
# Data Logger
# ------------------------------------------------------------------
class DataLogger:
    """Logs simulator output to CSV."""

    def __init__(self, logfile):
        self.csvfile = open(logfile, "w")
        header = "total_revenue,total_power,completed,"
        header += ",".join([f"soh_{i}" for i in range(50)]) + ","
        header += ",".join([f"state_{i}" for i in range(50)])
        self.csvfile.write(header + "\n")
        self.csvfile.write("profit,total_power,completed,")
        self.csvfile.write(",".join([f"soh{i}" for i in range(50)]))
        self.csvfile.write(",")
        self.csvfile.write(",".join([f"status{i}" for i in range(50)]))
        self.csvfile.write("\n")
        self.p_old = [72.1] * 50
        self.retired = [0] * 50

    def write(self, info):
        total_power = 0
        p_curr = []
        soh_curr = []
        state = []

        for v in range(50):
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

        profit = info["total_revenue"]

        completed = info["completed"]
        entry = f"{profit},{total_power},{completed},"

        for i in range(50):
            entry += f"{soh_curr[i]},"
        entry += ",".join([f"{state[i]}" for i in range(50)])
        self.csvfile.write(entry + "\n")

    def close(self):
        self.csvfile.close()
