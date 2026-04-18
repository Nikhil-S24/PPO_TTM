"""Jobs."""

from typing import Dict, Union
from enum import Enum
import datetime

from simulator.region import *
from simulator.vehicle import *


DATEFMT = "%Y-%m-%d %H:%M:%S"


class JobStatus(Enum):
    ARRIVED = 1
    ASSIGNED = 2
    INPROGRESS = 3
    REJECTED = 4
    COMPLETE = 5
    FAILED = 6


class Job:
    def __init__(self, data: Dict, job_id: int, region: Region) -> None:
        self.id = job_id

        self.pickup_location = CyclicZoneGraphLocation(
            int(data["pickup_location"]), region
        )
        self.dropoff_location = CyclicZoneGraphLocation(
            int(data["dropoff_location"]), region
        )

        self.duration = (
            datetime.datetime.strptime(data["dropoff_time"], DATEFMT)
            - datetime.datetime.strptime(data["pickup_time"], DATEFMT)
        ).total_seconds()

        self.distance = float(data["distance"])
        self.fare = float(data["fare"])

        self.vehicle = None
        self.status = JobStatus.ARRIVED
        self.elapsed_time = 0
        self.counted = False

    def to_dict(self) -> Dict[str, Union[Dict, float, int, str]]:
        return {
            "pickup_location": self.pickup_location.to_dict(),
            "dropoff_location": self.dropoff_location.to_dict(),
            "duration": self.duration,
            "distance": self.distance,
            "fare": self.fare,
            "vehicle": self.vehicle,
            "status": self.status.name,
            "id": self.id,
        }

    def assign_vehicle(self, vehicle: int) -> None:
        self.status = JobStatus.ASSIGNED
        self.vehicle = vehicle

    def inprogress(self) -> None:
        self.status = JobStatus.INPROGRESS
        self.elapsed_time = 0  # reset timer

    def complete(self) -> None:
        self.status = JobStatus.COMPLETE

    def fail(self) -> None:
        self.status = JobStatus.FAILED

    def tick(self, dt: float) -> None:
        """
        Update job state over time.
        """

        # -------------------------------
        # ARRIVED → REJECTED (if ignored)
        # -------------------------------
        if self.status == JobStatus.ARRIVED:
            self.elapsed_time += dt
            if self.elapsed_time > dt:
                self.status = JobStatus.REJECTED

        # -------------------------------
        # INPROGRESS → COMPLETE
        # -------------------------------
        elif self.status == JobStatus.INPROGRESS:
            self.elapsed_time += dt

            if self.elapsed_time >= self.duration:
                self.status = JobStatus.COMPLETE