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
    """
    ReplayDemand replays jobs from a CSV file once.
    No looping.
    """

    def __init__(self, path: str, region, loop: bool = False) -> None:
        super().__init__()
        self.datefmt = "%Y-%m-%d %H:%M:%S"
        self.path = path
        self.csvfile = open(path, "r")
        self.reader = csv.DictReader(self.csvfile)

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
        self.exhausted = False   # 🔥 NEW

    def seek(self, t: datetime.datetime) -> None:
        if self.t is None:
            return

        if t <= self.t_min:
            self.csvfile.seek(0)
            self.reader = csv.DictReader(self.csvfile)
            try:
                self.last = next(self.reader)
                self.t = self.t_min
            except StopIteration:
                self.last = None
                self.exhausted = True
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
                self.exhausted = True
                return

    def tick(self, dt: float, conditions: Dict = None) -> Set:
        jobs = set()

        if self.exhausted or self.t is None:
            return jobs

        end = self.t + datetime.timedelta(seconds=dt)

        while not self.exhausted:
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
                self.exhausted = True
                self.csvfile.close()
                break

        return jobs