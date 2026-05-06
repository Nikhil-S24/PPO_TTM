"""Taxi fleet simulator with optional Zero-Shot Granite TTM forecasting."""

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

# TTM (optional)
try:
    from ttm.zero_shot_ttm import ZeroShotTTM
except ImportError:
    ZeroShotTTM = None

random.seed(0)
np.random.seed(0)


class TaxiFleetSimulator(gym.Env):
    """Taxi fleet simulator.

    Args:
        config: configuration dictionary (see config.yaml for details.)
    """

    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.config = config

        # Dynamic TTM control (controlled via YAML config)
        self.use_ttm = config.get("use_ttm", False)

        if self.use_ttm and ZeroShotTTM is not None:
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
        """Get an observation from the environment."""
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

            # When TTM is active, fuse current + predicted SoH
            if self.use_ttm:
                avg_soh = 0.5 * (current_soh + pred_scalar)
            else:
                avg_soh = current_soh

            obs[idx, 0] = avg_soh
            obs[idx, 1] = v.battery.soc

        return obs

    # ==================================================
    # Helper methods
    # ==================================================
    def get_closest_charger(self, vehicle: Vehicle) -> ChargeStation:
        """Get the closest charger to a vehicle."""
        distances = []
        for charger in self.charging_network:
            d, t = vehicle.location.to(charger.location)
            distances.append(d)
        return self.charging_network[distances.index(min(distances))]

    def get_closest_job(self, vehicle: Vehicle) -> Job:
        """Get the closest job to vehicle that is not inprogress or expired."""
        closest_job = None
        distance = float('inf')
        for job in self.arrived:
            d, t = vehicle.location.to(job.pickup_location)
            if d < distance:
                distance = d
                closest_job = job
        return closest_job

    # ==================================================
    # RESET
    # ==================================================
    def reset(self, seed: int = None, options: dict = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        # Initialize Time
        self.dt = float(self.config["delta t"])
        self.t = datetime.datetime.strptime(
            self.config["start t"], "%Y/%m/%d %H:%M:%S"
        )
        self.t_max = datetime.datetime.strptime(
            self.config["end t"], "%Y/%m/%d %H:%M:%S"
        )
        self.T_a = 25

        # Load Map
        self.region = CyclicZoneGraph(self.config["city"])

        # Load Demand (real CSV replay)
        self.demand = ReplayDemand(self.config["demand"], self.region)
        self.demand.seek(self.t)
        self.arrived = self.demand.tick(self.dt)
        self.assigned = set()
        self.inprogress = set()

        self.completed = 0
        self.rejected = 0
        self.failed = 0

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
            0, 1, shape=(len(self.fleet), 2)
        )
        self.action_space = gym.spaces.Box(
            0, 1, shape=(len(self.fleet), 2)
        )

        self.step_count = 0

        # Reset TTM State History
        self.soh_history = {v.vid: [] for v in self.fleet}
        self.predicted_soh = {v.vid: None for v in self.fleet}

        # Build info
        info = {}
        info["arrived"] = [j.to_dict() for j in self.arrived]
        info["assigned"] = [j.to_dict() for j in self.assigned]
        info["completed"] = self.completed
        info["rejected"] = self.rejected
        info["inprogress"] = [j.to_dict() for j in self.inprogress]
        info["failed"] = self.failed
        info["charging_network"] = [s.to_dict() for s in self.charging_network]
        info["fleet"] = [v.to_dict() for v in self.fleet]

        return self._get_obs(), info

    # ==================================================
    # STEP (aligned with reference repo)
    # ==================================================
    def step(self, action: np.ndarray):

        # -------------------------------
        # Action Execution (from reference repo)
        # -------------------------------
        for idx in range(len(self.fleet)):
            if (
                action[idx, 0] > 0.5
                and self.fleet[idx].status
                in [VehicleStatus.IDLE, VehicleStatus.CHARGING, VehicleStatus.TOCHARGE]
            ):
                self.fleet[idx].charge(
                    self.get_closest_charger(self.fleet[idx]),
                    action[idx, 1],
                )
            elif (
                len(self.arrived) > 0
                and self.fleet[idx].status
                in [VehicleStatus.IDLE, VehicleStatus.CHARGING, VehicleStatus.TOCHARGE]
            ):
                job = self.get_closest_job(self.fleet[idx])
                if job is not None:
                    self.fleet[idx].service_demand(job)
                    self.arrived.discard(job)
                    self.assigned.add(job)

        # -------------------------------
        # Vehicle & Charger Dynamics
        # -------------------------------
        for vehicle in self.fleet:
            vehicle.tick(self.dt, {"T_a": self.T_a})

        for charger in self.charging_network:
            charger.tick(self.fleet, self.dt, self.T_a)

        # -------------------------------
        # Get new arrivals (ReplayDemand)
        # -------------------------------
        self.arrived = self.arrived | self.demand.tick(self.dt)

        # -------------------------------
        # Update jobs in progress
        # -------------------------------
        to_completed = set()
        to_failed = set()
        for job in self.inprogress:
            if job.status == JobStatus.COMPLETE:
                to_completed = to_completed.union({job})
            elif job.status == JobStatus.FAILED:
                to_failed = to_failed.union({job})
        self.inprogress = self.inprogress - to_completed - to_failed
        self.completed += len(to_completed)
        self.failed += len(to_failed)

        # -------------------------------
        # Update assigned jobs
        # -------------------------------
        to_inprogress = set()
        to_failed = set()
        for job in self.assigned:
            if job.status == JobStatus.INPROGRESS:
                to_inprogress = to_inprogress.union({job})
            elif job.status == JobStatus.FAILED:
                to_failed = to_failed.union({job})
        self.assigned = self.assigned - to_inprogress - to_failed
        self.failed += len(to_failed)
        self.inprogress = self.inprogress.union(to_inprogress)

        # -------------------------------
        # Update arrived jobs
        # -------------------------------
        to_assigned = set()
        to_rejected = set()
        for job in self.arrived:
            job.tick(self.dt)
            if job.status == JobStatus.ASSIGNED:
                to_assigned = to_assigned.union({job})
            elif job.status == JobStatus.REJECTED:
                to_rejected = to_rejected.union({job})
            elif job.status == JobStatus.INPROGRESS:
                to_inprogress = to_inprogress.union({job})
        self.arrived = self.arrived - to_assigned - to_rejected - to_inprogress
        self.assigned = self.assigned.union(to_assigned)
        self.inprogress = self.inprogress.union(to_inprogress)
        self.rejected += len(to_rejected)

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

            if not self.use_ttm or self.ttm is None or len(history) < 512:
                self.predicted_soh[v.vid] = history[-1] if history else 1.0
                continue

            if self.step_count % self.ttm_update_interval == 0:
                self.predicted_soh[v.vid] = self.ttm.predict(history)

        # -------------------------------
        # Time Management
        # -------------------------------
        self.t += datetime.timedelta(seconds=self.dt)
        self.step_count += 1

        # -------------------------------
        # Calculate info (aligned with reference)
        # -------------------------------
        info = {}
        info["arrived"] = [j.to_dict() for j in self.arrived]
        info["assigned"] = [j.to_dict() for j in self.assigned]
        info["completed"] = self.completed
        info["rejected"] = self.rejected
        info["inprogress"] = [j.to_dict() for j in self.inprogress]
        info["failed"] = self.failed
        info["charging_network"] = [s.to_dict() for s in self.charging_network]
        info["fleet"] = [v.to_dict() for v in self.fleet]

        # ==================================================
        # Reward (for PPO training)
        # ==================================================
        ALPHA = 1.0
        reward = self.completed + ALPHA * sum(
            [
                v.battery.actual_capacity / v.battery.initial_capacity
                for v in self.fleet
            ]
        )

        return (
            self._get_obs(),
            reward,
            self.t >= self.t_max,
            self.step_count > 1000,
            info,
        )