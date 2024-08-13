from pprint import pprint
import glob
import os
import logging
import numpy
import copy

from . import dvapi


class DvApiException(Exception):
    def __init__(self, error=None):
        super().__init__(error)


class HardwareSession:
    endpoint = None
    session = None
    model = None
    shmfile = None
    shmfile_size = 256 * 1024 * 1024  # ? make this configurable
    dv_shm_desc = None
    model_path = None
    mode = None
    input_list = []
    output_list = []
    session_started: bool = False

    def __init__(self, model_path):
        # self.shmfile_path = config["application"]["shm_mem"]
        # self.socket = config["application"]["sock"]

        # output_path = os.path.join(os.getcwd(), self.config["out"])
        # model_paths = list(glob.glob(f"{output_path}/**/model.dvm"))
        # if not len(model_paths):
        #     raise FileNotFoundError(
        #         "No `model.dvm` found in specefied output directory"
        #     )

        # self.model_path = model_paths[0]
        self.model_path = model_path
        self.socket = "/var/run/dvproxy.sock"

    def __create_session(self):
        if ":" in self.socket:
            args = self.socket.split(":")
            ret, session = dvapi.DVSession.create_via_tcp_ipv4_socket(
                args[0], int(args[1])
            )
        else:
            ret, session = dvapi.DVSession.create_via_unix_socket(self.socket)

        if ret == dvapi.dv_status_code.DV_CONNECTION_ERROR:
            raise ConnectionError("Failed to establish connection with proxy")

        logging.info("Sucessfully connected to dvproxy.sock")
        self.session = session

    def __create_shared_memory(self):
        self.shmfile = open(self.shmfile_path, "w+")
        self.shmfile.truncate(self.shmfile_size)
        self.shmfile.seek(0, 0)
        status, self.dv_shm_desc = self.session.register_shmfd(
            self.shmfile.fileno(), self.shmfile_size, 0
        )
        if status != dvapi.dv_status_code.DV_SUCCESS:
            raise DvApiException(f"Failed to register shared file: {self.shmfile_path}")

        logging.info(f"Successfully registered shared memory file {self.shmfile_path}")

    def __get_endpoints(self):
        status, endpoints = self.session.get_endpoint_list()
        if not len(endpoints) or status != dvapi.dv_status_code.DV_SUCCESS:
            raise DvApiException("No endpoints connected")

        self.endpoint = endpoints[0]

    def __load_model(self):
        status, model = self.session.load_model_from_file(
            self.endpoint, self.model_path
        )
        if status != dvapi.dv_status_code.DV_SUCCESS:
            raise DvApiException(
                f"Error loading model {dvapi.dv_stringify_status_code(status)}"
            )

        self.model = model

    def start(self):
        if self.session_started:
            logging.info("HardwareSession already started")
            return

        self.__create_session()
        # self.__create_shared_memory()
        self.__get_endpoints()
        self.__load_model()

        self.session_started = True

        return self

    def __create_input_tensor(self, inputs: list[numpy.ndarray]):
        input_tensor = []
        for i in range(self.model.num_inputs):
            input_buf = numpy.zeros(self.model.input_param[i].size, dtype=numpy.int8)
            input_buf[:] = inputs[i].flatten()[:]
            input_tensor.append(dvapi.DVTensor(input_buf, self.model.input_param[i]))
        return input_tensor

    def __create_output_tensor(self):
        output_tensor = []
        for i in range(self.model.num_outputs):
            output_buf = numpy.zeros(self.model.output_param[i].size, dtype=numpy.int8)
            output_tensor.append(dvapi.DVTensor(output_buf, self.model.output_param[i]))
        return output_tensor

    def run_sync(self, inputs: list[numpy.ndarray], timeout=5000):

        input_tensors = self.__create_input_tensor(inputs)
        output_tensors = self.__create_output_tensor()

        status, _ = self.model.infer_sync(
            input_tensors, output_tensors, self.endpoint, timeout
        )

        if status != dvapi.dv_status_code.DV_SUCCESS:
            raise DvApiException(status)

        return output_tensors

    def __del__(self):
        self.close()

    def close(self):
        if not self.session_started:
            logging.info("HardwareSession not started")
            return
        self.model.unload()
        if self.dv_shm_desc is not None:
            self.dv_shm_desc.unregister()
            os.remove(self.shmfile_path)
            self.shmfile.close()

        self.session.close()
        self.session_started = False

    def get_model_input_params(self) -> list[dvapi.DVModelInputParam]:
        if not self.session_started:
            raise DvApiException("HardwareSession not started")

        return self.model.input_param

    def get_model_output_params(self):
        if not self.session_started:
            raise DvApiException("HardwareSession not started")

        return self.model.output_param


def quantize(
    image: numpy.ndarray, input_param: list[dvapi.DVModelInputParam]
) -> numpy.ndarray:

    pm = input_param[0].preprocess_param
    image = image.flatten()

    if pm.bpp == 1:
        image = numpy.floor(image * pm.qn * pm.output_scale + 0.5)
        if pm.is_signed:
            image = numpy.clip(image, -128, 127)
        else:
            image = numpy.clip(image, 0, 255)
        return image.astype(numpy.int8)
    else:
        image = numpy.floor(image * pm.qn * pm.output_scale + 0.5)
        if pm.is_signed:
            image = numpy.clip(image, -32768, 32767)
        else:
            image = numpy.clip(image, 0, 65535)
        return image.astype(numpy.int16)


def dequantize(outputs: list[dvapi.DVTensor]) -> numpy.ndarray:

    num_outputs = len(outputs)
    dequantized_outputs = []

    for i in range(num_outputs):

        output = copy.deepcopy(outputs[i].numpy_data)
        out_param = outputs[i].params
        if (
            not out_param.postprocess_param.is_struct_format
            and not out_param.postprocess_param.is_float
        ):
            outputBytes = output.tobytes()
            bpp = out_param.bpp
            intype = numpy.int8

            if bpp == 1:
                intype = (
                    numpy.int8 if out_param.postprocess_param.is_signed else numpy.uint8
                )
            elif bpp == 2:
                intype = (
                    numpy.int16
                    if out_param.postprocess_param.is_signed
                    else numpy.uint16
                )
            elif bpp == 4:
                intype = (
                    numpy.int32
                    if out_param.postprocess_param.is_signed
                    else numpy.uint32
                )
            out = (
                (
                    numpy.frombuffer(outputBytes, dtype=intype)
                    + out_param.postprocess_param.offset
                )
                / (
                    out_param.postprocess_param.qn
                    * out_param.postprocess_param.output_scale
                )
            ).astype(numpy.float32)

            dequantized_outputs.append(out)
        else:
            dequantized_outputs.append(output)

    return dequantized_outputs
