from ultralyticsplus import YOLO
from onnx.utils import extract_model

# load model
# model = YOLO("mshamrai/yolov8n-visdrone")
# model.export(format="onnx", opset=11)

# create postprocessing graph
onodes = ["/model.22/dfl/conv/Conv_output_0", "/model.22/Sigmoid_output_0"]
model_path = "yolov8n.onnx"
extract_model(
    model_path, "extracted.onnx", input_names=onodes, output_names=["output0"]
)

# # set model parameters
# model.overrides["conf"] = 0.25  # NMS confidence threshold
# model.overrides["iou"] = 0.45  # NMS IoU threshold
# model.overrides["agnostic_nms"] = False  # NMS class-agnostic
# model.overrides["max_det"] = 1000  # maximum number of detections per image

# # set image
# image = "https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg"

# # perform inference
# results = model.predict(image)


# # observe results
# print(results[0].boxes)
# # render = render_result(model=model, image=image, result=results[0])
# # render.show()
