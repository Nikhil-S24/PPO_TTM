"""Demand models."""

from typing import Dict, Set
import csv
import datetime

from simulator.job import *


class Demand:
    """Abstract class modeling demand."""

    def __init__(self) -> None:
        self.t_min = None
        self.t = None

    def seek(self, t: datetime.datetime) -> None:
        raise NotImplemented

    def tick(self, dt: float, conditions: Dict = None) -> Set:
        raise NotImplemented


class ReplayDemand(Demand):
    """ReplayDemand replays jobs from a CSV file safely."""

    def __init__(self, path: str, region, loop: bool = False) -> None:
        super().__init__()
        self.datefmt = "%Y-%m-%d %H:%M:%S"
        self.path = path
        self.csvfile = open(path, "r")
        self.reader = csv.DictReader(self.csvfile)

        # Read first row safely
        try:
            self.last = next(self.reader)
        except StopIteration:
            self.last = None

        if self.last:
            self.t_min = datetime.datetime.strptime(
                self.last["pickup_time"], self.datefmt
            )
            self.t = self.t_min
        else:
            self.t_min = None
            self.t = None

        self.global_idx = 0
        self.region = region
        self.loop = loop

    def seek(self, t: datetime.datetime) -> None:
        """Move demand pointer to time t safely."""
        if self.t is None:
            return

        if t <= self.t_min:
            self.csvfile.seek(0)
            self.reader = csv.DictReader(self.csvfile)
            self.last = next(self.reader)
            self.t = self.t_min
            return

        while True:
            try:
                if self.t >= t:
                    return
                self.last = next(self.reader)
                self.t = datetime.datetime.strptime(
                    self.last["pickup_time"], self.datefmt
                )
            except StopIteration:
                return

    def tick(self, dt: float, conditions: Dict = None) -> Set:
        """Return jobs released in [t, t + dt)."""
        jobs = set()

        if self.t is None:
            return jobs

        end = self.t + datetime.timedelta(seconds=dt)

        while True:
            try:
                if self.t >= end:
                    break

                jobs.add(
                    Job(self.last, job_id=self.global_idx, region=self.region)
                )
                self.global_idx += 1

                self.last = next(self.reader)
                self.t = datetime.datetime.strptime(
                    self.last["pickup_time"], self.datefmt
                )

            except StopIteration:
                # End of CSV → stop generating demand
                break

        return jobs
