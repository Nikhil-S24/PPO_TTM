"""Taxi fleet simulator with Zero-Shot Granite TTM + KDE Demand."""

from typing import Dict, Tuple
import datetime
import random

import gymnasium as gym
import numpy as np

from simulator.job import *
from simulator.charger import *
from simulator.region import *
from simulator.vehicle import *

# KDE
from kde_model import load_and_prepare_data, train_kde, generate_ride

# TTM
from ttm.zero_shot_ttm import ZeroShotTTM

random.seed(0)
np.random.seed(0)


class TaxiFleetSimulator(gym.Env):

    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.config = config
        self.max_port_power = max(
    station["max port power"] for station in config["charging stations"]
)
        # Dynamic TTM control (controlled via YAML config)
        self.use_ttm = config.get("use_ttm", False)

        if self.use_ttm:
            self.ttm = ZeroShotTTM(
                context_length=512,
                prediction_length=96,
            )
        else:
            self.ttm = None

        self.ttm_update_interval = 50

    # ==================================================
    # OBSERVATION (Future-Aware Fusion)
    # ==================================================
    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((len(self.fleet), 2))

        for idx, v in enumerate(self.fleet):
            current_soh = v.battery.actual_capacity / v.battery.initial_capacity

            predicted_soh = self.predicted_soh.get(v.vid)

            if predicted_soh is None:
                pred_scalar = current_soh
            elif isinstance(predicted_soh, np.ndarray):
                pred_scalar = np.mean(predicted_soh)
            else:
                pred_scalar = predicted_soh

            # Standard fusion logic for evaluation
            avg_soh = 0.5 * (current_soh + pred_scalar)

            obs[idx, 0] = avg_soh
            obs[idx, 1] = v.battery.soc

        return obs.flatten()

    # ==================================================
    # RESET (Gymnasium Compatible)
    # ==================================================
    def reset(self, seed: int = None, options: dict = None) -> Tuple[np.ndarray, Dict]:
        # Gymnasium's super().reset handles the seed initialization
        super().reset(seed=seed)

        self.dt = float(self.config["delta t"])
        self.t = datetime.datetime.strptime(
            self.config["start t"], "%Y/%m/%d %H:%M:%S"
        )
        self.t_max = datetime.datetime.strptime(
            self.config["end t"], "%Y/%m/%d %H:%M:%S"
        )
        self.T_a = 25

        self.region = CyclicZoneGraph(self.config["city"])

        # KDE INIT
        data = load_and_prepare_data(self.config["demand"])
        self.kde = train_kde(data)

        self.arrived = set()
        self.assigned = set()
        self.inprogress = set()

        self.completed = 0
        self.rejected = 0
        self.failed = 0
        self.total_revenue = 0

        # Used for per-step incremental reward calculation
        self.prev_revenue = 0
        self.prev_completed = 0

        # Fleet Initialization
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

        # Charging Network Initialization
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

        self.observation_space = gym.spaces.Box(
            0, 1, shape=(len(self.fleet) * 2,)
        )
        self.action_space = gym.spaces.Box(
            0, 1, shape=(len(self.fleet), 2)
        )

        self.step_count = 0

        # Reset TTM State History
        self.soh_history = {v.vid: [] for v in self.fleet}
        self.predicted_soh = {v.vid: None for v in self.fleet}

        info = {
            "fleet": [v.to_dict() for v in self.fleet],
            "charging_network": [s.to_dict() for s in self.charging_network],
            "completed": self.completed,
            "total_revenue": self.total_revenue,
        }

        return self._get_obs(), info

    # ==================================================
    # STEP
    # ==================================================
    def step(self, action: np.ndarray):

        # -------------------------------
        # Action Execution
        # -------------------------------
        idle_charge_penalty = 0.0

        low_soc_penalty = 0.0

        for idx, v in enumerate(self.fleet):

            if (
                (action[idx, 0] > 0.5 or action[idx, 1] > 0.2)
                and v.status in [
                    VehicleStatus.IDLE,
                    VehicleStatus.CHARGING,
                    VehicleStatus.TOCHARGE,
                ]
            ):

                if self.arrived and v.battery.soc > 0.4:
                    idle_charge_penalty += 1.0

                v.charge(
                    min(
                        self.charging_network,
                        key=lambda c: v.location.to(c.location)[0],
                    ),
                    float(action[idx, 1]) * self.max_port_power,
                )

            elif self.arrived and v.status == VehicleStatus.IDLE:
                job = min(
                    self.arrived,
                    key=lambda j: v.location.to(j.pickup_location)[0],
                )
                v.service_demand(job)
                self.arrived.remove(job)
                self.assigned.add(job)
            
            if v.battery.soc < 0.5 and v.status != VehicleStatus.CHARGING:
                low_soc_penalty += (0.5 - v.battery.soc)


        # -------------------------------
        # Vehicle & Charger Dynamics
        # -------------------------------
        for v in self.fleet:
            v.tick(self.dt, {"T_a": self.T_a})

        # Track completed jobs and revenue
        for v in self.fleet:
            if hasattr(v, "job") and v.job is not None:
                if v.job.status.name == "COMPLETE" and not getattr(v.job, "counted", False):
                    self.completed += 1
                    self.total_revenue += v.job.fare
                    v.job.counted = True

        for c in self.charging_network:
            c.tick(self.fleet, self.dt, self.T_a)

        # -------------------------------
        # KDE Demand Generation
        # -------------------------------
        for _ in range(np.random.randint(0, 2)):
            ride = generate_ride(self.kde)
            pickup_time = self.t.strftime("%Y-%m-%d %H:%M:%S")
            drop_time = (
                self.t + datetime.timedelta(minutes=random.randint(5, 20))
            ).strftime("%Y-%m-%d %H:%M:%S")

            data = {
                "pickup_location": ride["pickup_location"],
                "dropoff_location": ride["dropoff_location"],
                "pickup_time": pickup_time,
                "dropoff_time": drop_time,
                "distance": random.uniform(1, 10),
                "fare": random.uniform(5, 30),
            }

            job = Job(data, random.randint(0, 1000000), self.region)
            self.arrived.add(job)

        # -------------------------------
        # TTM Prediction Logic
        # -------------------------------
        for v in self.fleet:
            soh = v.battery.actual_capacity / v.battery.initial_capacity
            self.soh_history[v.vid].append(soh)

            if len(self.soh_history[v.vid]) > 512:
                self.soh_history[v.vid].pop(0)

        # Zero-Shot Prediction (every X steps)
        for v in self.fleet:
            history = self.soh_history[v.vid]

            if not self.use_ttm or len(history) < 512:
                self.predicted_soh[v.vid] = history[-1]
                continue

            if self.step_count % self.ttm_update_interval == 0:
                self.predicted_soh[v.vid] = self.ttm.predict(history)

        # -------------------------------
        # Time Management
        # -------------------------------
        self.t += datetime.timedelta(seconds=self.dt)
        self.step_count += 1

        # ==================================================
        # ✅ TTM-AWARE PPO REWARD (STEP 3 TUNING)
        # ==================================================
        # ADJUST THESE TO MOVE PPO GRAPH BETWEEN BASELINE AND TTM
        ALPHA = 1.0  # Revenue Weight
        BETA = 3.0   # Current Degradation Penalty
        GAMMA = 2.0  # TTM Foresight Penalty (Predictive)

        # 1. Revenue/Job reward (per-step)
        incremental_completed = self.completed - self.prev_completed
        incremental_revenue = self.total_revenue - self.prev_revenue

        # 2. Current SoH penalty
        soh_penalty = sum(
            1 - (v.battery.actual_capacity / v.battery.initial_capacity)
            for v in self.fleet
        )

        # 3. TTM Foresight penalty (Predicting future drops)
        future_penalty = 0.0
        for v in self.fleet:
            forecast = self.predicted_soh[v.vid]
            if isinstance(forecast, np.ndarray):
                pred_mean = np.mean(forecast)
            else:
                pred_mean = forecast

            current_soh = v.battery.actual_capacity / v.battery.initial_capacity
            # Only penalize if we predict health will get worse
            future_penalty += max(0.0, current_soh - pred_mean)

        # Calculate combined reward
        # Paper-aligned simplified reward for PPO behavior debugging
        reward = (
    2 * incremental_completed
    + 0.1 * incremental_revenue
    - 0.5 * idle_charge_penalty
    - 5.0 * low_soc_penalty
)

        # Update per-step tracking variables
        self.prev_revenue = self.total_revenue
        self.prev_completed = self.completed

        # -------------------------------
        # Info & Return
        # -------------------------------
        info = {
            "fleet": [v.to_dict() for v in self.fleet],
            "charging_network": [s.to_dict() for s in self.charging_network],
            "completed": self.completed,
            "total_revenue": self.total_revenue,
        }

        return (
            self._get_obs(),
            reward,
            self.t >= self.t_max,
            self.step_count > 1000,
            info,
        )