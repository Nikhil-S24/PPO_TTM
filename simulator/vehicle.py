"""Model of an electric vehicle."""

from typing import Dict, Union
from enum import Enum

from simulator.battery import *
from simulator.region import *


class VehicleStatus(Enum):
    IDLE = 1
    TOPICKUP = 2
    TOCHARGE = 3
    CHARGING = 4
    TOLOC = 5
    ONJOB = 6
    OFFDUTY = 7
    RECOVERY = 8


class Vehicle:
    def __init__(
        self,
        model: Union[str, Dict[str, float]],
        battery: Union[str, Battery],
        location: Location,
        vid: int,
    ) -> None:

        self.model = model
        self.vid = vid
        self.charger = None

        # Vehicle specs
        if self.model.lower() == "byd e6":
            capacity = 71.7
            self.efficiency = 17.1
        else:
            capacity = model["capacity"]
            self.efficiency = model["efficiency"]

        # Battery
        if battery.lower() == "multistage":
            self.battery = MultiStageBattery(capacity)
        else:
            self.battery = battery

        # Location & state
        self.depo = location
        self.location = location
        self.destination = location

        self.time_remaining = 0.0
        self.status = VehicleStatus.IDLE

        self.job = None
        self.preferred_rate = 0.0

    # ------------------------------------------------------

    def to_dict(self) -> Dict:
        return {
            "location": self.location.to_dict(),
            "destination": self.destination.to_dict(),
            "time_remaining": self.time_remaining,
            "status": self.status.name,
            "battery": self.battery.to_dict(),
        }

    # ------------------------------------------------------

    def service_demand(self, job):
        if self.charger:
            self.charger.disconnect(self.vid)

        self.job = job
        self.job.assign_vehicle(self.vid)

        self.destination = job.pickup_location
        self.time_remaining = self.location.to(self.destination)[1]

        self.status = VehicleStatus.TOPICKUP

    # ------------------------------------------------------

    def charge(self, charger, preferred_rate):
        self.charger = charger
        self.destination = charger.location
        self.time_remaining = self.location.to(self.destination)[1]
        self.preferred_rate = preferred_rate

        if self.status != VehicleStatus.CHARGING:
            self.status = VehicleStatus.TOCHARGE
            self.charger.disconnect(self.vid)

    # ------------------------------------------------------

    def initialize_recovery_state(self):
        self.destination = self.depo
        self.time_remaining = 24 * 3600
        self.battery.charge(self.battery.actual_capacity, 3600, T_a=25)

    # ------------------------------------------------------

    def tick(self, dt: float, conditions: Dict[str, int]):

        # -------------------------------
        # IDLE
        # -------------------------------
        if self.status == VehicleStatus.IDLE:
            self.battery.age(dt, conditions["T_a"])

        # -------------------------------
        # GO TO PICKUP
        # -------------------------------
        elif self.status == VehicleStatus.TOPICKUP:
            self.time_remaining -= dt

            if self.time_remaining <= 0:
                prev_location = self.location
                # update location
                self.location = self.destination

                # energy consumption
                dist, _ = prev_location.to(self.destination)
                dW = dist * self.efficiency / 100
                self.battery.discharge(dW, dt, conditions["T_a"])

                if self.battery.soc <= 0:
                    self.status = VehicleStatus.RECOVERY
                    self.job.fail()
                    self.initialize_recovery_state()
                else:
                    # go to dropoff
                    self.destination = self.job.dropoff_location
                    self.time_remaining = self.job.duration

                    self.job.inprogress()
                    self.status = VehicleStatus.ONJOB

        # -------------------------------
        # ON JOB
        # -------------------------------
        elif self.status == VehicleStatus.ONJOB:
            self.time_remaining -= dt

            if self.time_remaining <= 0:
                prev_location = self.location
                self.location = self.destination

                dist, _ = prev_location.to(self.destination)
                dW = dist * self.efficiency / 100
                self.battery.discharge(dW, dt, conditions["T_a"])

                if self.battery.soc <= 0:
                    self.status = VehicleStatus.RECOVERY
                    self.job.fail()
                    self.initialize_recovery_state()
                else:
                    self.status = VehicleStatus.IDLE
                    self.job.complete()

        # -------------------------------
        # GO TO CHARGE
        # -------------------------------
        elif self.status == VehicleStatus.TOCHARGE:
            self.time_remaining -= dt

            if self.time_remaining <= 0:
                prev_location = self.location
                self.location = self.destination

                dist, _ = prev_location.to(self.destination)
                dW = dist * self.efficiency / 100
                self.battery.discharge(dW, dt, conditions["T_a"])

                if self.battery.soc <= 0:
                    self.status = VehicleStatus.RECOVERY
                    self.initialize_recovery_state()
                else:
                    self.status = VehicleStatus.CHARGING

        # -------------------------------
        # CHARGING
        # -------------------------------
        elif self.status == VehicleStatus.CHARGING:
            self.charger.request_charge(self.preferred_rate, self.vid)

        # -------------------------------
        # RECOVERY
        # -------------------------------
        elif self.status == VehicleStatus.RECOVERY:
            self.time_remaining -= dt
            if self.time_remaining <= 0:
                self.status = VehicleStatus.IDLE

        else:
            raise Exception(f"Invalid vehicle state: {self.status}")