import sys
import traceback
import argparse
import typing as typ
import time
import attr
import cv2 
import numpy as np

from gstreamer import GstContext, GstPipeline, GstApp, Gst, GstVideo
import gstreamer.utils as utils

video_file="/home/edgepi/omair/traffilyzer/uav0000137_00458_v.mp4"
model_path="/home/edgepi/omair/traffilyzer/model.dvm"
CLASSES = {0: 'pedestrian', 1: 'people', 2: 'bicycle', 3: 'car', 4: 'van', 5: 'truck', 6: 'tricycle', 7: 'awning-tricycle', 8: 'bus', 9: 'motor'}

DEFAULT_PIPELINE = f"uridecodebin uri=file://{video_file} ! videoconvert ! videoscale ! queue ! pre ! queue ! dvmodel model={model_path} ! queue ! cpppost silent=True ! queue ! appsink emit-signals=True"

print(DEFAULT_PIPELINE)
#working pipeline
# DEFAULT_PIPELINE = 'rtspsrc location=rtsp://admin:abc12345@172.17.0.1:8554/colgate latency=0 retry=50 transport=tcp ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvideoconvert nvbuf-memory-type=3 ! capsfilter caps="video/x-raw, format=BGR" ! videorate ! video/x-raw, width=3840,height=2160,format=BGR, framerate=15/1 ! appsink emit-signals=True'

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--pipeline", required=False,
                default=DEFAULT_PIPELINE, help="Gstreamer pipeline without gst-launch")

args = vars(ap.parse_args())

command = args["pipeline"]
height = 480
width = 640
channels = 3
image_size = channels*width*height

def extract_buffer(sample: Gst.Sample) -> np.ndarray:
    """Extracts Gst.Buffer from Gst.Sample and converts to np.ndarray"""

    buffer = sample.get_buffer()  # Gst.Buffer

    # print(buffer.pts, buffer.dts, buffer.offset)
    
    memory1 = buffer.peek_memory(1)  # Equivalent to gst_buffer_peek_memory(inbuf, 1)
    memory2 = buffer.peek_memory(2)  # Equivalent to gst_buffer_peek_memory(inbuf, 2)

    # Map memory blocks for reading
    success2, info2 = memory2.map(Gst.MapFlags.READ)  # Equivalent to gst_memory_map(memory2, &info, GST_MAP_READ)
    success1, info1 = memory1.map(Gst.MapFlags.READ)  # Equivalent to gst_memory_map(memory1, &info1, GST_MAP_READ)

    if success2 and success1:
        # Access data in memory2 and memory1 via info2 and info1
        data2 = info2.data
        data1 = info1.data
        
        image_np = np.frombuffer(data1, dtype=np.uint8).reshape((height, width, channels))
        detections_np = np.frombuffer(data2, dtype=np.float32)
        # After processing, unmap the memory
        memory2.unmap(info2)  # Equivalent to gst_memory_unmap(memory2, &info)
        memory1.unmap(info1)  # Equivalent to gst_memory_unmap(memory1, &info1)

        return image_np, detections_np.reshape(-1,6)

    return None,None  # remove single dimension if exists


def on_buffer(sink: GstApp.AppSink, data: typ.Any) -> Gst.FlowReturn:
    """Callback on 'new-sample' signal"""
    # Emit 'pull-sample' signal
    # https://lazka.github.io/pgi-docs/GstApp-1.0/classes/AppSink.html#GstApp.AppSink.signals.pull_sample

    sample = sink.emit("pull-sample")  # Gst.Sample

    if isinstance(sample, Gst.Sample):
        img, dets = extract_buffer(sample)
        # cv2.imwrite(f"{time.time()}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        print(
            "Received {type} with shape {shape} of type {dtype}".format(type=type(img),
                                                                        shape=img.shape,
                                                                        dtype=img.dtype))
        print(
            "Received {type} with shape {shape} of type {dtype}".format(type=type(dets),
                                                                        shape=dets.shape,
                                                                        dtype=dets.dtype), dets)
        
        # draw detections on img 
        img = np.copy(img)
        for det in dets:
            x1,y1,x2,y2,conf,cls = det
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            # set font scale according to bounding box size
            font_scale = 1
            font_thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(f'{int(cls)}:{conf}', cv2.FONT_HERSHEY_COMPLEX_SMALL, font_scale, font_thickness)
            # string format conf to 2 decimal places
            conf = "{:.2f}".format(conf)
            cv2.putText(img, f'{CLASSES[int(cls)]}:{conf}', (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imwrite(f"{time.time()}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return Gst.FlowReturn.OK

    return Gst.FlowReturn.ERROR


with GstContext():  # create GstContext (hides MainLoop)
    # create GstPipeline (hides Gst.parse_launch)
    with GstPipeline(command) as pipeline:
        appsink = pipeline.get_by_cls(GstApp.AppSink)[0]  # get AppSink
        print("appsink = ", appsink)
        # subscribe to <new-sample> signal
        appsink.connect("new-sample", on_buffer, None)
        while not pipeline.is_done:
            time.sleep(.1)
