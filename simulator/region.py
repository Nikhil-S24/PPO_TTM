"""
Synthetic Multi-Zone Region Map (Improved for 5-year simulation)
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
# Synthetic Multi-Zone Circular Graph
# ==========================================================

class CyclicZoneGraph(Region):
    def __init__(self, mapfile: str) -> None:
        print("[DEBUG] Using synthetic circular city map")

        # ✅ Increased zones for realism
        self.num_zones = 50

        self.radius = 10.0  # km (larger city)
        self.speed_kmph = 30.0  # default speed

        # Generate circular coordinates
        self.coordinates = {}

        for i in range(self.num_zones):
            angle = 2 * math.pi * i / self.num_zones
            x = self.radius * math.cos(angle)
            y = self.radius * math.sin(angle)
            self.coordinates[i] = (x, y)

        # Map structure
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

        # --------------------------------------------------
        # Traffic-aware speed
        # --------------------------------------------------
        if conditions and "hour" in conditions:
            hour = conditions["hour"]

            if 7 <= hour <= 10 or 17 <= hour <= 20:
                speed = 20.0   # rush hour (slow)
            elif 0 <= hour <= 5:
                speed = 40.0   # night (fast)
            else:
                speed = 30.0   # normal
        else:
            speed = self.speed_kmph

        # Convert to time
        time_hours = dist / speed
        time_seconds = time_hours * 3600

        return (dist, time_seconds)