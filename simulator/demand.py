"""Demand models with KDE-based synthetic demand."""

from typing import Dict, Set
import datetime
import random

from simulator.job import *
from simulator.region import CyclicZoneGraphLocation

# KDE
from kde_model import load_and_prepare_data, train_kde, generate_rides


# --------------------------------------------------
# Base Demand Class
# --------------------------------------------------
class Demand:
    def __init__(self) -> None:
        self.t_min = None
        self.t = None

    def seek(self, t: datetime.datetime) -> None:
        raise NotImplementedError

    def tick(self, dt: float, conditions: Dict = None) -> Set:
        raise NotImplementedError


# --------------------------------------------------
# KDE-based Demand Generator
# --------------------------------------------------
class ReplayDemand(Demand):
    """
    KDE-based demand generator.
    Supports long-term simulation (5 years).
    """

    def __init__(self, path: str, region, loop: bool = False) -> None:
        super().__init__()

        self.path = path
        self.region = region
        self.global_idx = 0

        # -------------------------------
        # Initialize KDE
        # -------------------------------
        data = load_and_prepare_data(path)
        self.kde = train_kde(data)

        self.t = None
        self.t_min = None

    # --------------------------------------------------

    def seek(self, t: datetime.datetime) -> None:
        self.t = t
        self.t_min = t

    # --------------------------------------------------

    def tick(self, dt: float, conditions: Dict = None) -> Set:
        jobs = set()

        if self.t is None:
            return jobs

        # -------------------------------
        # Generate rides from KDE
        # -------------------------------
        rides = generate_rides(self.kde, self.t)

        for ride in rides:

            # Ensure valid zone indices
            pickup_zone = int(ride["pickup_location"]) % 10
            drop_zone = int(ride["dropoff_location"]) % 10

            pickup_loc = CyclicZoneGraphLocation(pickup_zone, self.region)
            drop_loc = CyclicZoneGraphLocation(drop_zone, self.region)

            # -------------------------------
            # Distance & travel time
            # -------------------------------
            distance, travel_time = pickup_loc.to(drop_loc)

            # Safety: avoid zero distance
            distance = max(0.5, distance)

            # Minimum duration = 5 minutes
            duration_seconds = max(300, travel_time)

            drop_time = self.t + datetime.timedelta(seconds=duration_seconds)

            # -------------------------------
            # Realistic fare model
            # -------------------------------
            base_fare = 3.0
            per_km_rate = 2.5
            per_min_rate = 0.4

            fare = (
                base_fare
                + per_km_rate * distance
                + per_min_rate * (duration_seconds / 60.0)
            )

            # Add randomness
            fare *= random.uniform(0.85, 1.15)

            # Ensure minimum fare
            fare = max(5.0, fare)

            # -------------------------------
            # Create job
            # -------------------------------
            job_data = {
                "pickup_location": pickup_zone,
                "dropoff_location": drop_zone,
                "pickup_time": self.t.strftime("%Y-%m-%d %H:%M:%S"),
                "dropoff_time": drop_time.strftime("%Y-%m-%d %H:%M:%S"),
                "distance": distance,
                "fare": fare,
            }

            job = Job(
                job_data,
                job_id=self.global_idx,
                region=self.region
            )

            self.global_idx += 1
            jobs.add(job)

        # -------------------------------
        # Advance simulation time
        # -------------------------------
        self.t += datetime.timedelta(seconds=dt)

        return jobs