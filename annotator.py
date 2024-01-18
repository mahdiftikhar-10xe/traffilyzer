from ultralytics import YOLO
import supervision as sv
import numpy as np
from manager import ZonesManager


class BoxAnnotator:
    def __init__(self):
        self.__annotator = sv.BoxAnnotator(thickness=1, text_padding=0)

    @staticmethod
    def __create_label(model: YOLO, detections: sv.Detections):
        return [
            f"#{tracker_id} {model.names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, tracker_id
            in detections
        ]

    def annotate(self, frame: np.ndarray, model: YOLO, detections: sv.Detections) -> np.ndarray:
        render = frame.copy()
        self.__annotator.annotate(
            scene=render,
            detections=detections,
            labels=self.__create_label(model, detections)
        )
        return render


class ZoneAnnotator:
    def __init__(self, manager: ZonesManager):
        self.__manager = manager

    def annotate(self, frame: np.ndarray):
        render = frame.copy()
        for zone in self.__manager.input_zones + self.__manager.output_zones:
            zone_center = sv.get_polygon_center(polygon=zone.zone.polygon)
            sv.draw_polygon(
                scene=render,
                polygon=zone.zone.polygon,
                color=zone.color
            )
            sv.draw_text(
                scene=render,
                text=str(len(zone.detected_ids)),
                text_anchor=zone_center,
                background_color=zone.color
            )

            if zone.is_output:
                tracker = self.__manager.tracker[zone.id]
                for i, zone_in_id in enumerate(tracker):
                    count = len(self.__manager.tracker[zone.id][zone_in_id])
                    anchor = sv.Point(x=zone_center.x + 20 * (i + 1), y=zone_center.y + 20 * (i + 1))
                    sv.draw_text(
                        scene=render,
                        text=str(count),
                        text_anchor=anchor,
                        background_color=[x for x in self.__manager.input_zones if x.id == zone_in_id][0].color
                    )

        return render
