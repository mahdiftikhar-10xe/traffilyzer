import time
from statistics import Statistics  # type: ignore

import cv2
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO

import annotator
import manager
from Yolov8n import Yolov8n
from dvapi.infer_session import quantize, dequantize


class Video:
    def __init__(
        self, model: str, video_file: str, conf_threshold: float = 0.3, iou: float = 0.7
    ):
        self.__conf = conf_threshold
        self.__iou = iou
        self.__stream = cv2.VideoCapture(video_file)
        self.__video_info = sv.VideoInfo.from_video_path(video_file)
        self.__render = np.zeros([self.__video_info.width, self.__video_info.height, 3])
        self.__paused = False
        self.__click = 0
        self.__polygon = np.zeros([4, 2], dtype=int)

        self.__cuda = "cuda" if torch.cuda.is_available() else "cpu"
        self.__model = YOLO("yolov8n.pt").to(self.__cuda)
        self.custom_model = Yolov8n("model.dvm")
        self.__tracker = sv.ByteTrack(
            track_buffer=self.__video_info.fps * 2,
            frame_rate=self.__video_info.fps,
        )
        self.__zones_manager = manager.ZonesManager()
        self.__box_annotator = annotator.BoxAnnotator()
        self.__zone_annotator = annotator.ZoneAnnotator(self.__zones_manager)
        self.__breadcrumbs = sv.TraceAnnotator(
            color=sv.ColorPalette.default(),
            position=sv.Position.CENTER,
            trace_length=100,
            thickness=2,
        )
        self.__stats = Statistics(self.__model.names, self.__zones_manager)

        self.__times = {}

    @staticmethod
    def __mouse_callback(event, x, y, flags, param):
        if event in [cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN] and param.__paused:
            mgr = param.__zones_manager
            param.__polygon[param.__click] = [x, y]
            param.__click += 1
            zone_type = event == cv2.EVENT_RBUTTONDOWN
            if param.__click == 4:
                param.__click = 0
                mgr.create_polygon(param.__polygon, zone_type, param.__video_info)
                param.__render = param.__zone_annotator.annotate(param.__render)
                cv2.imshow("Traffic Analysis", param.__render)

    def process(self):
        from tqdm import tqdm

        writer = cv2.VideoWriter(
            "output/output.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.__video_info.fps,
            (self.__video_info.width, self.__video_info.height),
        )

        pbar = tqdm(total=self.__video_info.total_frames, ncols=100)

        timers = []

        while self.__stream.isOpened():
            pbar.update(1)

            ret, frame = self.__stream.read()
            if not ret:
                break
            timer = time.perf_counter()
            detections = self.__detect(frame)
            timers.append(time.perf_counter() - timer)
            self.__render = self.__annotate(frame, detections)
            self.__stats.update(detections)

            writer.write(self.__render)

        self.__stats.save()
        self.__stream.release()
        writer.release()
        pbar.close()
        print()
        print(f"Average inference time: {np.mean(timers) * 1000:.2f}ms")
        print(f"Average FPS: {self.__video_info.total_frames / np.sum(timers):.2f}")
        print(
            f"\t{'Average preprocess latency:':<40} {np.mean(self.__times['preprocess']) * 1000:.2f} ms"
        )
        # print(
        #     f"\t{'Average quantize latency:':<40} {np.mean(self.__times['quantize']) * 1000:.2f} ms"
        # )
        print(
            f"\t{'Average inference latency:':<40} {np.mean(self.__times['inference']) * 1000:.2f} ms"
        )
        print(
            f"\t{'Average dequantize latency:':<40} {np.mean(self.__times['dequantize']) * 1000:.2f} ms"
        )
        print(
            f"\t{'Average postprocess latency:':<40} {np.mean(self.__times['postprocess']) * 1000:.2f} ms"
        )
        print(
            f"\t{'Average tracking latency:':<40} {np.mean(self.__times['tracking']) * 1000:.2f} ms"
        )

        # cv2.destroyAllWindows()

    def old_preocess(self):
        cv2.namedWindow("Traffic Analysis", cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback("Traffic Analysis", self.__mouse_callback, self)

        while self.__stream.isOpened():
            if not self.__paused:
                ret, frame = self.__stream.read()
                cv2.imwrite("test.jpg", frame)
                if ret:
                    detections = self.__detect(frame)
                    self.__render = self.__annotate(frame, detections)
                    self.__stats.update(detections)

                    cv2.imwrite("render.jpg", self.__render)

                    cv2.imshow("Traffic Analysis", self.__render)
            key_press = cv2.waitKey(1) & 0xFF
            if key_press == ord("q"):
                self.__stats.save()
                break
            if key_press == ord("p"):
                self.__paused = not self.__paused

        self.__stats.save()
        self.__stream.release()
        # cv2.destroyAllWindows()

    def __annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        render = self.__box_annotator.annotate(frame, self.__model, detections)
        render = self.__zone_annotator.annotate(render)
        render = self.__breadcrumbs.annotate(render, detections)
        render = sv.draw_text(
            render, f"Device: {self.__cuda}", sv.Point(100, 10), sv.Color.red(), 1, 2
        )

        return render

    def __detect(self, frame: np.ndarray) -> sv.Detections:
        # result = self.__model(
        #     frame,
        #     conf=self.__conf,
        #     iou=self.__iou,
        #     classes=[2, 3, 4, 5, 6, 7, 8, 9],
        #     verbose=True,
        #     imgsz=640,
        # )[0]
        preprocess_times = self.__times.get("preprocess", [])
        inference_times = self.__times.get("inference", [])
        postprocess_times = self.__times.get("postprocess", [])
        # quantize_times = self.__times.get("quantize", [])
        dequantize_times = self.__times.get("dequantize", [])
        tracking_times = self.__times.get("tracking", [])

        timer = time.perf_counter()
        quantized = self.custom_model.preprocess(frame)
        preprocess_times.append(time.perf_counter() - timer)

        # timer = time.perf_counter()
        # quantized = quantize(
        #     preprocessed_frame, self.custom_model.session.get_model_input_params()
        # )
        # quantize_times.append(time.perf_counter() - timer)

        timer = time.perf_counter()
        outputs = self.custom_model.infer(quantized)
        inference_times.append(time.perf_counter() - timer)

        timer = time.perf_counter()
        dequantized = dequantize(outputs)
        dequantize_times.append(time.perf_counter() - timer)

        timer = time.perf_counter()
        xyxy, conf, class_id = self.custom_model.postprocess(dequantized, frame.shape)
        postprocess_times.append(time.perf_counter() - timer)

        # result = self.custom_model(frame)
        # xyxy, conf, class_id = result
        xyxy = xyxy.astype(int)
        class_id = class_id.reshape((-1))

        timer = time.perf_counter()
        detections = sv.Detections(xyxy=xyxy, confidence=conf, class_id=class_id)

        detections = self.__tracker.update_with_detections(detections)
        input_detections = list()
        output_detections = list()
        for polygon in (
            self.__zones_manager.input_zones + self.__zones_manager.output_zones
        ):
            result = detections[polygon.zone.trigger(detections=detections)]
            zone_type = output_detections if polygon.is_output else input_detections
            zone_type.append(result)

        self.__zones_manager.update(input_detections, output_detections)
        tracking_times.append(time.perf_counter() - timer)

        self.__times = {
            "preprocess": preprocess_times,
            "inference": inference_times,
            "postprocess": postprocess_times,
            # "quantize": quantize_times,
            "dequantize": dequantize_times,
            "tracking": tracking_times,
        }
        return detections
