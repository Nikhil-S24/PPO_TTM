"""Models for charging infrastructure."""

from typing import Dict, List, Union

from simulator.region import *
from simulator.vehicle import *


class ChargePort:
    """Single port in a charging station.

    Args:
        P_max - maximum instantaneous supply power (kW)
        efficiency - charging efficiency (%)
    """

    def __init__(self, P_max: float, efficiency: float) -> None:
        self.P_max = P_max
        self.efficiency = efficiency
        self.vehicle = None
        self.P_t = 0

    def to_dict(self) -> Dict[str, Union[int, float]]:
        return {
            "P_max": self.P_max,
            "P_t": self.P_t,
            "efficiency": self.efficiency,
            "vehicle": self.vehicle,
        }


class ChargeStation:
    """Charging station."""

    def __init__(
        self, location: Location, ports: List[ChargePort], P_max: float = None
    ) -> None:
        self.location = location
        self.ports = ports
        self.P_max = P_max
        self.vehicle_queue = {}

    def to_dict(self) -> Dict[str, Union[Dict, List, int, float]]:
        return {
            "location": self.location.to_dict(),
            "ports": [p.to_dict() for p in self.ports],
            "P_max": self.P_max,
            "vehicle_queue": [vid for vid in self.vehicle_queue],
        }

    def request_charge(self, preferred_rate: float, vehicle: int) -> None:
        """
        A vehicle requests charging.
        """
        for port in self.ports:
            if port.vehicle == vehicle:
                port.P_t = min(preferred_rate, port.P_max)
                return

        self.vehicle_queue[vehicle] = preferred_rate

    def disconnect(self, vehicle: int) -> None:
        """
        Disconnect vehicle from station.
        """
        for port in self.ports:
            if port.vehicle == vehicle:
                port.vehicle = None
                port.P_t = 0
                return

        if vehicle in self.vehicle_queue:
            del self.vehicle_queue[vehicle]

    def tick(self, fleet: List, dt: float, T_a: float) -> None:
        """
        Update charging state.
        """

        to_charge = list(self.vehicle_queue.keys())
        power_requested = 0.0

        # -------------------------------
        # Assign vehicles to ports
        # -------------------------------
        for port in self.ports:
            if port.vehicle is None and len(to_charge) > 0:
                vehicle = to_charge.pop()
                port.vehicle = vehicle
                port.P_t = min(self.vehicle_queue[vehicle], port.P_max)
                del self.vehicle_queue[vehicle]

            if port.vehicle is not None:
                if power_requested + port.P_t <= self.P_max:
                    power_requested += port.P_t
                else:
                    port.P_t = max(0.0, self.P_max - power_requested)
                    power_requested += port.P_t

        # -------------------------------
        # Apply charging to vehicles
        # -------------------------------
        for port in self.ports:
            if port.vehicle is not None:
                v = fleet[port.vehicle]

                # ✅ Stop charging if battery is full
                if v.battery.soc >= 0.99:
                    port.vehicle = None
                    port.P_t = 0
                    continue

                # ✅ Convert kW → kWh using dt
                energy = port.P_t * (dt / 3600)

                # ✅ Apply efficiency
                energy *= port.efficiency

                # ✅ Charge battery
                v.battery.charge(energy, dt, T_a)