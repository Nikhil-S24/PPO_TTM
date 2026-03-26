"""
Synthetic Multi-Zone Region Map (For Mid-Review Testing)
"""

from typing import Dict, Tuple
from typing_extensions import Self
import math


# ==========================================================
# Base Classes
# ==========================================================

class Location:
    def __init__(self, region: "Region") -> None:
        self.region = region

    def to_dict(self) -> Dict:
        raise NotImplementedError

    def to(self, location: Self) -> Tuple[float, float]:
        return self.region.distance(self, location)


class Region:
    def distance(
        self, start: Location, end: Location, conditions: Dict = None
    ) -> Tuple[float, float]:
        raise NotImplementedError


# ==========================================================
# Location Object
# ==========================================================

class CyclicZoneGraphLocation(Location):
    def __init__(self, zone: int, region: Region) -> None:
        super().__init__(region)
        self.zone = zone  

    def to_dict(self) -> Dict:
        return {"zone": self.zone}

    def to(self, location: Location) -> Tuple[float, float]:
        return self.region.distance(self, location)


# ==========================================================
# Synthetic 10-Zone Circular Graph
# ==========================================================

class CyclicZoneGraph(Region):
    def __init__(self, mapfile: str) -> None:
        print("[DEBUG] Using synthetic 10-zone circular city map")

        self.num_zones = 10
        self.radius = 5.0  # km
        self.speed_kmph = 30.0  # average speed

        # Place zones in circle
        self.coordinates = {}

        for i in range(self.num_zones):
            angle = 2 * math.pi * i / self.num_zones
            x = self.radius * math.cos(angle)
            y = self.radius * math.sin(angle)
            self.coordinates[i] = (x, y)

        # Provide zone list
        self.map = {i: {} for i in range(self.num_zones)}

    # ------------------------------------------------------

    def distance(
        self, start: Location, end: Location, conditions: Dict = None
    ) -> Tuple[float, float]:

        z1 = start.zone % self.num_zones
        z2 = end.zone % self.num_zones

        x1, y1 = self.coordinates[z1]
        x2, y2 = self.coordinates[z2]

        # Euclidean distance
        dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        # Convert distance to travel time (seconds)
        time_hours = dist / self.speed_kmph
        time_seconds = time_hours * 3600

        return (dist, time_seconds)