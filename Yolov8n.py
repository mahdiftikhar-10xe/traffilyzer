import numpy as np
import onnxruntime as ort
import cv2

from supervision.detection.core import Detections

np.set_printoptions(suppress=True)


class Yolov8n:
    def __init__(self, model_path) -> None:

        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        h, w = self.input_shape[2:]
        im_h, im_w, _ = image.shape

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 640))

        gain = min(h / im_h, w / im_w)

        new_h = int(im_h * gain)
        new_w = int(im_w * gain)

        dh = (h - new_h) // 2
        dw = (w - new_w) // 2

        padded_image = np.ones((h, w, 3), dtype=np.uint8) * 114
        padded_image[dh : new_h + dh, dw : new_w + dw, :] = cv2.resize(
            image, (new_w, new_h)
        )

        image = padded_image.transpose(2, 0, 1)
        image = image / 255
        image = image.astype(np.float32)
        image = np.expand_dims(image, axis=0)
        return image

    @staticmethod
    def nms(dets: np.ndarray, scores: np.ndarray, thresh: float) -> list[int]:
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = scores

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort(kind="mergesort")[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep

    @staticmethod
    def xywh2xyxy(x):
        # Convert boxes with shape [n, 4] from [x, y, w, h] to [x1, y1, x2, y2] where x1y1 is top-left, x2y2=bottom-right
        # y = x.clone() if isinstance(x, torch.Tensor) else numpy.copy(x)
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    @staticmethod
    def xyxy2xywh(x):
        # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
        # y = x.clone() if isinstance(x, torch.Tensor) else numpy.copy(x)
        y = np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y

    def non_max_suppression(
        self,
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,  # number of classes (optional)
        max_time_img=0.05,
        max_nms=30000,
        max_wh=7680,
    ):
        """
        Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

        Args:
            prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
                containing the predicted boxes, classes, and masks. The tensor should be in the format
                output by a model, such as YOLO.
            conf_thres (float): The confidence threshold below which boxes will be filtered out.
                Valid values are between 0.0 and 1.0.
            iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
                Valid values are between 0.0 and 1.0.
            classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
            agnostic (bool): If True, the model is agnostic to the number of classes, and all
                classes will be considered as one.
            multi_label (bool): If True, each box may have multiple labels.
            labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
                list contains the apriori labels for a given image. The list should be in the format
                output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
            max_det (int): The maximum number of boxes to keep after NMS.
            nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
            max_time_img (float): The maximum time (seconds) for processing one image.
            max_nms (int): The maximum number of boxes into torchvision.ops.nms().
            max_wh (int): The maximum box width and height in pixels

        Returns:
            (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
                shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
                (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
        """
        bs = prediction.shape[0]  # bs 1
        nc = prediction.shape[1] - 4  # nc 80
        nm = prediction.shape[1] - nc - 4  # 0
        mi = 4 + nc  # 84
        xc = (
            np.amax(prediction[:, 4:mi], 1) > conf_thres
        )  # candidates equivalent to pred[:, 4:mi] > conf_thres
        # xc = pred[..., 4] > nms_score_threshold  # candidates

        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

        prediction = prediction.transpose(0, -1, -2)

        output = [np.zeros((0, 6))] * bs

        for xi, x in enumerate(prediction):  # image index, image inference
            x = x[xc[xi]]  # confidence
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = np.zeros((len(lb), nc + nm + 5))
                v[:, :4] = lb[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                x = np.concatenate((x, v), 0)

            box = x[:, :4]
            cls = x[:, 4:]
            mask = x[:, nc:nm]
            # box, cls, mask = np.split(x, [4, pred.shape[1]], 1)
            box = self.xywh2xyxy(box)

            if multi_label:
                i, j = (cls > conf_thres).nonzero()
                x = np.concatenate(
                    (box[i], x[i, j + 4, None], j[:, None].astype(np.float), mask[i]), 1
                )
            else:  # best class onlyx
                conf = np.zeros((x.shape[0], 1))
                j = np.zeros((x.shape[0], 1))
                index = 0
                for i in range(0, x.shape[0]):
                    conf[index][0] = x[index][5:].max()
                    j[index][0] = np.where(x[index][5:] == x[index][5:].max())[0][0]
                    index = index + 1
                arr = np.concatenate((box, conf, j, mask), 1)
                x = arr[np.where(arr[:, 4] > conf_thres)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            x = x[(-x[:, 4]).argsort(kind="mergesort")[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = self.nms(boxes, scores, iou_thres)
            i = np.array(i)

            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            output[xi] = x[i]

        return output

    def scale_coords(self, coords: np.ndarray, image_shape: tuple) -> np.ndarray:

        boxes: np.ndarray = coords.copy()
        im_h, im_w, _ = image_shape
        h, w = self.input_shape[2:]
        gain = min(h / im_h, w / im_w)  # gain  = old / new
        pad_x = (w - im_w * gain) / 2  # width padding
        pad_y = (h - im_h * gain) / 2  # height padding

        boxes[:, [0, 2]] -= pad_x
        boxes[:, [1, 3]] -= pad_y
        boxes /= gain

        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, im_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, im_h)

        return boxes

    def postprocess(self, result, frame):
        result = self.non_max_suppression(result)[0]

        boxes = result[:, :4]
        scores = result[:, 4]
        classes = result[:, 5:]

        boxes = self.scale_coords(boxes, frame.shape)

        return [boxes, scores, classes]

    def __forward(self, input: np.ndarray) -> np.ndarray:
        result = self.session.run(None, {self.input_name: input})[0]
        return result

    def __call__(self, input: np.ndarray) -> np.ndarray:
        return self.__forward(input)


if __name__ == "__main__":
    model = Yolov8n("yolov8n.onnx")
    image = cv2.imread("test.jpg")
    xyxy, conf, class_id = model(image)
    print(xyxy)
