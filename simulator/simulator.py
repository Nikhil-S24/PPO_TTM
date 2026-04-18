"""Taxi fleet simulator with PPO + TTM-aware reward (CLEAN VERSION)."""

from typing import Dict
import datetime
import random

import gymnasium as gym
import numpy as np

from simulator.job import *
from simulator.charger import *
from simulator.region import *
from simulator.vehicle import *
from simulator.demand import ReplayDemand

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

        self.use_ttm = config.get("use_ttm", False)

        if self.use_ttm:
            self.ttm = ZeroShotTTM(context_length=512, prediction_length=96)
        else:
            self.ttm = None

        self.ttm_update_interval = 50

    # ==================================================
    # OBSERVATION
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

            avg_soh = 0.5 * (current_soh + pred_scalar)

            obs[idx, 0] = avg_soh
            obs[idx, 1] = v.battery.soc

        return obs.flatten()

    # ==================================================
    # RESET
    # ==================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.dt = float(self.config["delta t"])
        self.t = datetime.datetime.strptime(self.config["start t"], "%Y/%m/%d %H:%M:%S")
        self.t_max = datetime.datetime.strptime(self.config["end t"], "%Y/%m/%d %H:%M:%S")
        self.T_a = 25

        self.region = CyclicZoneGraph(self.config["city"])

        self.demand = ReplayDemand(self.config["demand"], self.region)
        self.demand.seek(self.t)
        self.arrived = self.demand.tick(self.dt)

        self.completed = 0
        self.total_revenue = 0

        self.prev_revenue = 0
        self.prev_completed = 0

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

        self.observation_space = gym.spaces.Box(0, 1, shape=(len(self.fleet) * 2,))
        self.action_space = gym.spaces.Box(0, 1, shape=(len(self.fleet), 2))

        self.step_count = 0

        self.soh_history = {v.vid: [] for v in self.fleet}
        self.predicted_soh = {v.vid: None for v in self.fleet}

        return self._get_obs(), {
            "fleet": [v.to_dict() for v in self.fleet],
            "total_revenue": self.total_revenue,
            "completed": self.completed,
        }

    # ==================================================
    # STEP
    # ==================================================
    def step(self, action):

        for idx, v in enumerate(self.fleet):

            if (
                action[idx, 0] > 0.5
                and v.status in [VehicleStatus.IDLE, VehicleStatus.CHARGING, VehicleStatus.TOCHARGE]
            ):
                v.charge(
                    min(self.charging_network, key=lambda c: v.location.to(c.location)[0]),
                    float(action[idx, 1]) * self.max_port_power,
                )

            elif self.arrived and v.status == VehicleStatus.IDLE:

                jobs = list(self.arrived)

                # 🔥 Sort jobs by profitability (distance proxy)
                jobs.sort(key=lambda j: j.fare, reverse=True)

                # PPO chooses among top jobs
                top_k = min(5, len(jobs))  # only top 5 jobs

                job_idx = int(action[idx, 1] * top_k) % top_k

                job = jobs[job_idx]
                
                v.service_demand(job)
                self.arrived.remove(job)

        for v in self.fleet:
            v.tick(self.dt, {"T_a": self.T_a})

        for v in self.fleet:
            if v.job and v.job.status.name == "COMPLETE" and not getattr(v.job, "counted", False):
                self.completed += 1
                self.total_revenue += v.job.fare
                v.job.counted = True

        for c in self.charging_network:
            c.tick(self.fleet, self.dt, self.T_a)

        self.arrived |= self.demand.tick(self.dt)

        self.t += datetime.timedelta(seconds=self.dt)
        self.step_count += 1

        # -------------------------------
        # FIXED REWARD
        # -------------------------------
        inc_rev = self.total_revenue - self.prev_revenue
        inc_comp = self.completed - self.prev_completed

        future_penalty = 0.0
        for v in self.fleet:
            forecast = self.predicted_soh[v.vid]

            # ✅ FIX APPLIED HERE
            if forecast is None:
                pred_mean = v.battery.actual_capacity / v.battery.initial_capacity
            elif isinstance(forecast, np.ndarray):
                pred_mean = np.mean(forecast)
            else:
                pred_mean = forecast

            current_soh = v.battery.actual_capacity / v.battery.initial_capacity
            future_penalty += max(0.0, current_soh - pred_mean)

        reward = 20.0 * inc_rev + 5.0 * inc_comp - 2.0 * future_penalty

        self.prev_revenue = self.total_revenue
        self.prev_completed = self.completed

        return (
            self._get_obs(),
            reward,
            self.t >= self.t_max,
            self.step_count > 50000,
            {
                "fleet": [v.to_dict() for v in self.fleet],
                "total_revenue": self.total_revenue,
                "completed": self.completed,
            },
        )