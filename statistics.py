from uuid import UUID

import supervision as sv
from dataclasses import dataclass, field
from manager import ZonesManager
import xlsxwriter
import time


@dataclass
class DetectionData:
    detected_ids: set[int] = field(default_factory=set)
    classified: dict[str, int] = field(default_factory=dict)


class Statistics:
    def __init__(self, names: dict[int, str], zone_manager: ZonesManager):
        self.__names = names
        self.__manager = zone_manager
        self.__detect_data = DetectionData()
        for n in names.values():
            self.__detect_data.classified.update({n: 0})

    def update(self, detections: sv.Detections):
        for d in detections:
            class_id = d[3]
            tracker_id = d[4]
            if tracker_id not in self.__detect_data.detected_ids:
                self.__detect_data.detected_ids.add(tracker_id)
                self.__detect_data.classified[self.__names[class_id]] += 1

    def save(self):
        timestamp = time.strftime("%d-%m-%Y_%H-%M-%S")
        workbook = xlsxwriter.Workbook(f"{timestamp}.xlsx")
        detections_sheet = workbook.add_worksheet("Detections")
        zone_sheet = workbook.add_worksheet("Zones")

        for row, (key, value) in enumerate(self.__detect_data.classified.items()):
            detections_sheet.write_row(row, 0, [key, value])
            if row == len(self.__detect_data.classified.items()) - 1:
                detections_sheet.write_row(row + 1, 0, ["Total", len(self.__detect_data.detected_ids)])

        for i, zone in enumerate(self.__manager.output_zones):
            total = str(len(zone.detected_ids))
            zone_sheet.write_row(i + 1, 0, [str(zone.id), total])

        zone_sheet.write(0, 1, "Total")
        for i, zone in enumerate(self.__manager.input_zones):
            zone_sheet.write(0, i + 2, str(zone.id))
            for j, _ in enumerate(self.__manager.output_zones):
                zone_sheet.write(j + 1, i + 2, 0)

        for zone in self.__manager.output_zones:
            tracker = self.__manager.tracker[zone.id]
            for i, zone_in_id in enumerate(tracker):
                count = len(self.__manager.tracker[zone.id][zone_in_id])
                input_pol = [x for x in self.__manager.input_zones if x.id == zone_in_id][0]
                output_pol = [x for x in self.__manager.output_zones if x.id == zone.id][0]
                col = self.__manager.input_zones.index(input_pol) + 2
                row = self.__manager.output_zones.index(output_pol) + 1
                zone_sheet.write(row, col, count)

        workbook.close()
