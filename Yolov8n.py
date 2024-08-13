import numpy as np
import onnxruntime as ort
import cv2


class Yolov8n:
    def __init__(self, model_path) -> None:

        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    @staticmethod
    def preprocess(image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 640))
        image = image.transpose(2, 0, 1)
        image = image / 255
        image = image.astype(np.float32)
        image = np.expand_dims(image, axis=0)
        print(image.shape)
        return image

    def __forward(self, input: np.ndarray) -> np.ndarray:
        preprocessed = self.preprocess(input)
        result = self.session.run(None, {self.input_name: preprocessed})[0]
        print(result.shape)
        return result

    def __call__(self, input: np.ndarray) -> np.ndarray:
        return self.__forward(input)


if __name__ == "__main__":
    model = Yolov8n("yolov8n.onnx")
    image = cv2.imread("test.jpg")
    result = model(image)
