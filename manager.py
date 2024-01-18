import numpy as np
import supervision as sv
from dataclasses import dataclass, field
from typing import List, Dict, Set
from uuid import uuid4, UUID


@dataclass
class Polygon:
    id: UUID
    color: sv.Color
    is_output: bool
    zone: sv.PolygonZone
    detected_ids: set[int] = field(default_factory=set)


class ZonesManager:
    def __init__(self):
        self.input_zones: List[Polygon] = list()
        self.output_zones: List[Polygon] = list()
        self.tracker: Dict[UUID, Dict[UUID, Set[int]]] = dict()

    def create_polygon(self, coords: np.ndarray, zone_is_output: bool, video_info: sv.VideoInfo) -> Polygon:
        zone = sv.PolygonZone(
            coords,
            (video_info.width, video_info.height),
            sv.Position.CENTER
        )
        zone_array = self.output_zones if zone_is_output else self.input_zones
        polygon = Polygon(
            id=uuid4(),
            color=sv.Color(np.random.choice(256), np.random.choice(256), np.random.choice(256)),
            is_output=zone_is_output,
            zone=zone
        )
        zone_array.append(polygon)
        if polygon.is_output:
            self.tracker.setdefault(polygon.id, {})
        return polygon

    def update(self, input_zones: List[sv.Detections], output_zones: List[sv.Detections]):
        for polygon in self.input_zones:
            for tracker_id in input_zones[self.input_zones.index(polygon)].tracker_id:
                polygon.detected_ids.add(tracker_id)

        for polygon in self.output_zones:
            for tracker_id in output_zones[self.output_zones.index(polygon)].tracker_id:
                polygon.detected_ids.add(tracker_id)

                for z in self.input_zones:
                    if tracker_id in z.detected_ids:
                        self.tracker.setdefault(polygon.id, {})
                        self.tracker[polygon.id].setdefault(z.id, set())
                        self.tracker[polygon.id][z.id].add(tracker_id)
