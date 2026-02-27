"""Taxi fleet simulator with Zero-Shot Granite TTM integration."""

from typing import Dict, Tuple
import datetime
import random

import gymnasium as gym
import numpy as np

from simulator.job import *
from simulator.charger import *
from simulator.demand import *
from simulator.region import *
from simulator.vehicle import *

# 🔹 Granite Zero-shot TTM
from ttm.zero_shot_ttm import ZeroShotTTM

random.seed(0)
np.random.seed(0)


class TaxiFleetSimulator(gym.Env):
    """Taxi fleet simulator with TTM-enhanced RL."""

    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.config = config

        # ==================================================
        # LOAD TTM ONLY ONCE (CRITICAL FIX)
        # ==================================================
        # =========================
        # MANUAL TTM TOGGLE
        # =========================
        self.use_ttm = False   # Change to False for baseline

        if self.use_ttm:
            self.ttm = ZeroShotTTM(
                context_length=512,
                prediction_length=96,
            )
        else:
            self.ttm = None

        self.ttm_update_interval = 50
    # ==================================================
    # TTM-ENHANCED OBSERVATION
    # ==================================================
    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((len(self.fleet), 2))

        for idx, v in enumerate(self.fleet):
            current_soh = (
                v.battery.actual_capacity / v.battery.initial_capacity
            )

            predicted_soh = self.predicted_soh.get(v.vid)

            if predicted_soh is None:
                pred_scalar = current_soh
            elif isinstance(predicted_soh, np.ndarray):
                pred_scalar = np.mean(predicted_soh)
            else:
                pred_scalar = predicted_soh

            avg_soh = 0.5 * (current_soh + pred_scalar)

            obs[idx, 0] = avg_soh
            obs[idx, 1] = v.battery.soc

        return obs
    # ==================================================
    # RESET
    # ==================================================
    def reset(self, seed: int = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        # -------------------------------
        # Time
        # -------------------------------
        self.dt = float(self.config["delta t"])
        self.t = datetime.datetime.strptime(
            self.config["start t"], "%Y/%m/%d %H:%M:%S"
        )
        self.t_max = datetime.datetime.strptime(
            self.config["end t"], "%Y/%m/%d %H:%M:%S"
        )
        self.T_a = 25

        # -------------------------------
        # Map & Demand
        # -------------------------------
        self.region = CyclicZoneGraph(self.config["city"])

        self.demand = ReplayDemand(self.config["demand"], self.region)
        self.demand.seek(self.t)
        self.arrived = self.demand.tick(self.dt)

        self.assigned = set()
        self.inprogress = set()
        self.completed = 0
        self.rejected = 0
        self.failed = 0
        self.total_revenue=0

        # -------------------------------
        # Fleet
        # -------------------------------
        self.fleet = []
        for vid in range(self.config["fleet"]["size"]):
            self.fleet.append(
                Vehicle(
                    model=self.config["fleet"]["vehicle"],
                    battery=self.config["fleet"]["battery model"],
                    location=CyclicZoneGraphLocation(
                        random.choice(list(self.region.map.keys())),
                        self.region,
                    ),
                    vid=vid,
                )
            )

        # -------------------------------
        # Charging Network
        # -------------------------------
        self.charging_network = []
        for station in self.config["charging stations"]:
            self.charging_network.append(
                ChargeStation(
                    location=CyclicZoneGraphLocation(
                        station["location"], self.region
                    ),
                    ports=[
                        ChargePort(
                            station["max port power"],
                            station["efficiency"],
                        )
                        for _ in range(station["ports"])
                    ],
                    P_max=station["max total power"],
                )
            )

        # -------------------------------
        # Gym Spaces
        # -------------------------------
        self.observation_space = gym.spaces.Box(
            0, 1, shape=(len(self.fleet), 2)
        )
        self.action_space = gym.spaces.Box(
            0, 1, shape=(len(self.fleet), 2)
        )

        self.step_count = 0

        # -------------------------------
        # RESET TTM STATE (NOT MODEL)
        # -------------------------------
        self.soh_history = {v.vid: [] for v in self.fleet}
        self.predicted_soh = {v.vid: None for v in self.fleet}

        info = {
            "arrived": [j.to_dict() for j in self.arrived],
            "assigned": [],
            "completed": self.completed,
            "rejected": self.rejected,
            "inprogress": [],
            "failed": self.failed,
            "total_revenue": self.total_revenue,
            "charging_network": [
                s.to_dict() for s in self.charging_network
            ],
            "fleet": [v.to_dict() for v in self.fleet],
        }

        return self._get_obs(), info

    # ==================================================
    # STEP
    # ==================================================
    def step(self, action: np.ndarray):

        # -------------------------------
        # Action Execution
        # -------------------------------
        for idx, v in enumerate(self.fleet):

            if (
                action[idx, 0] > 0.5
                and v.status
                in [
                    VehicleStatus.IDLE,
                    VehicleStatus.CHARGING,
                    VehicleStatus.TOCHARGE,
                ]
            ):
                v.charge(
                    min(
                        self.charging_network,
                        key=lambda c: v.location.to(c.location)[0],
                    ),
                    action[idx, 1],
                )

            elif self.arrived and v.status == VehicleStatus.IDLE:
                job = min(
                    self.arrived,
                    key=lambda j: v.location.to(j.pickup_location)[0],
                )
                v.service_demand(job)

        # -------------------------------
        # Vehicle & Charger Dynamics
        # -------------------------------
        for v in self.fleet:
            v.tick(self.dt, {"T_a": self.T_a})
        # Count completed jobs safely
        for v in self.fleet:
            if hasattr(v, "job") and v.job is not None:
                if v.job.status.name == "COMPLETE":
                    self.completed += 1
                    self.total_revenue += v.job.fare
                    v.job.counted=True
        for c in self.charging_network:
            c.tick(self.fleet, self.dt, self.T_a)

        # -------------------------------
        # Update SoH History
        # -------------------------------
        for v in self.fleet:
            soh = v.battery.actual_capacity / v.battery.initial_capacity
            self.soh_history[v.vid].append(soh)

            if len(self.soh_history[v.vid]) > 512:
                self.soh_history[v.vid].pop(0)

        # -------------------------------
        # Zero-Shot Prediction
        # -------------------------------
        for v in self.fleet:
            history = self.soh_history[v.vid]

            if not self.use_ttm:
                self.predicted_soh[v.vid] = history[-1]
                continue

            if len(history) < 512:
                self.predicted_soh[v.vid] = history[-1]
                continue

            if self.step_count % self.ttm_update_interval == 0:
                self.predicted_soh[v.vid] = self.ttm.predict(history)

            elif self.predicted_soh[v.vid] is None:
                self.predicted_soh[v.vid] = history[-1]

        # -------------------------------
        # Demand Update
        # -------------------------------
        self.arrived |= self.demand.tick(self.dt)

        # -------------------------------
        # Time Update
        # -------------------------------
        self.t += datetime.timedelta(seconds=self.dt)
        self.step_count += 1

        if self.step_count % 200 == 0:
            print("Step count:", self.step_count)

        # ==================================================
        # TTM-AWARE REWARD (FIXED)
        # ==================================================
        ALPHA = 1.0
        BETA = 2.0

        # Current SoH reward
        current_soh_reward = sum(
            v.battery.actual_capacity / v.battery.initial_capacity
            for v in self.fleet
        )

        # Future degradation penalty (convert forecast to scalar)
        future_penalty = 0.0

        for v in self.fleet:
            forecast = self.predicted_soh[v.vid]

            if isinstance(forecast, np.ndarray):
                pred_mean = np.mean(forecast)
            else:
                pred_mean = forecast

            future_penalty += max(0.0, 1.0 - pred_mean)

        reward = (
            self.completed
            + ALPHA * current_soh_reward
            - BETA * future_penalty
        )

        # -------------------------------
        # Info Dictionary
        # -------------------------------
        info = {
            "arrived": [j.to_dict() for j in self.arrived],
            "assigned": [j.to_dict() for j in self.assigned],
            "completed": self.completed,
            "rejected": self.rejected,
            "inprogress": [j.to_dict() for j in self.inprogress],
            "failed": self.failed,
            "total_revenue": self.total_revenue,
            "charging_network": [
                s.to_dict() for s in self.charging_network
            ],
            "fleet": [v.to_dict() for v in self.fleet],
        }

        return (
            self._get_obs(),
            reward,
            self.t >= self.t_max,
            self.step_count > 1000,
            info,
        )