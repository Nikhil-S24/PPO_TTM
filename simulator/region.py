"""Region map (SAFE DUMMY VERSION)."""

from typing import Dict, Tuple
from typing_extensions import Self
import pickle


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


class CyclicZoneGraphLocation(Location):
    def __init__(self, zone: int, region: Region) -> None:
        super().__init__(region)
        # FORCE ALL ZONES TO 0
        self.zone = 0

    def to_dict(self) -> Dict:
        return {"zone": self.zone}

    def to(self, location: Location) -> Tuple[float, float]:
        return self.region.distance(self, location)


class CyclicZoneGraph(Region):
    def __init__(self, mapfile: str) -> None:
        print(f"[DEBUG] Initializing CyclicZoneGraph with mapfile: {mapfile}")

        # ALWAYS USE DUMMY SINGLE-ZONE MAP
        self.map = {
            0: {
                0: {
                    "distance": 0.0,
                    "time": 0.0,
                }
            }
        }

        print("[DEBUG] Using dummy single-zone city map")

    def distance(
        self, start: Location, end: Location, conditions: Dict = None
    ) -> Tuple[float, float]:
        return (0.0, 0.0)
