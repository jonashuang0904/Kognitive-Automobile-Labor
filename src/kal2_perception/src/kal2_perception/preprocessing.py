import time
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path

from typing import Tuple

import cv2 as cv

class ImagePreprocessor(ABC):
    def __init__(self, verbose: bool = False) -> None:
        super().__init__()
        self._verbose = verbose

        self._last_duration = None
        self._average_duration = None


    def process(self, color_image: np.ndarray, depth_image: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        result = self._do_process(color_image, depth_image)
        dt = time.perf_counter() - t0
        self._average_duration = (self._average_duration or dt) * 0.9 + dt * 0.1
        self._last_duration = dt
        return result
    
    def get_last_duration(self) -> float:
        return self._last_duration
    
    def get_average_duration(self) -> float:
        return self._average_duration


    @abstractmethod
    def _do_process(self, color_image: np.ndarray, depth_image: np.ndarray) -> np.ndarray:
        pass


class InferenceSession(ABC):
    def __init__(self, model_path: Path, input_shape: Tuple[int, int, int, int]) -> None:
        super().__init__()
        self._model_path = model_path
        self._input_shape = input_shape

    @property
    def model_path(self) -> str:
        return self._model_path
    
    def run(self, input_data: np.ndarray) -> np.ndarray:
        if input_data.shape != self._input_shape:
            raise ValueError(f"Invalid input shape: {input_data.shape} != {self._input_shape}")

        return self._do_run(input_data)

    @abstractmethod
    def _do_run(self, input_data: np.ndarray) -> np.ndarray:
        pass

class OpenVinoInferenceSession(InferenceSession):
    def __init__(self, *, model_path: Path, input_shape: Tuple[int, int, int, int], device: str = 'CPU') -> None:
        super().__init__(model_path, input_shape)
        self._model_path = model_path
        self._device = device

        from openvino import Core, Layout, Type
        from openvino.preprocess import PrePostProcessor, ResizeAlgorithm

        self._core = Core()
        model = self._core.read_model(model_path)

        preprocessor = PrePostProcessor(model)
        _, h, w, _ = input_shape

        preprocessor.input().tensor() \
            .set_shape(input_shape) \
            .set_element_type(Type.f32) \
            .set_layout(Layout('NHWC'))

        preprocessor.input().preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR) # TODO: Check if irrelevant
        preprocessor.input().model().set_layout(Layout('NHWC'))                 # TODO: Check if irrelevant
        preprocessor.output().tensor().set_element_type(Type.f32)

        model = preprocessor.build()
        self._compiled_model = self._core.compile_model(model, device)


    def _do_run(self, input_data: np.ndarray) -> np.ndarray:
        return self._compiled_model.infer_new_request({0: input_data})[0]

class OnnxInferenceSession(InferenceSession):
    def __init__(self, *, model_path: str, input_shape: Tuple[int, int, int, int]) -> None:
        super().__init__(model_path, input_shape)
        
        from onnxruntime import InferenceSession
        self._inference_session = InferenceSession(model_path)

    def _do_run(self, input_data: np.ndarray) -> np.ndarray:
        return self._inference_session.run(None, {'input_1': input_data})[0]


class UnetPreprocessor(ImagePreprocessor):
    def __init__(self, *, model_path: Path, horizon: int = 200, runtime: str = "onnx", input_shape: Tuple[int, int, int, int] = (1, 224, 224, 3)) -> None:
        super().__init__()
        self._horizon = horizon
        self._input_shape = input_shape
        self._image_shape = (720, 1280)
        self._interpolation = cv.INTER_LINEAR

        if runtime == "onnx":
            self._inference_session = OnnxInferenceSession(model_path=model_path, input_shape=input_shape)
        elif runtime == "openvino":
            self._inference_session = OpenVinoInferenceSession(model_path=model_path, input_shape=input_shape, device='GPU')
        else:
            raise ValueError(f"Invalid runtime: {runtime}")

    def _resize_for_inference(self, color_image: np.ndarray) -> np.ndarray:
        _, h, w, _ = self._input_shape
        cropped = cv.resize(color_image[self._horizon:], (h, w), interpolation=self._interpolation)

        return np.expand_dims(cropped, axis=0).astype(np.float32) / 255.0
    
    def _restore_shape_after_inference(self, segmented_image: np.ndarray) -> np.ndarray:
        target_height, target_width = self._image_shape

        resized = cv.resize(segmented_image, (target_width, target_height-self._horizon), interpolation=self._interpolation)

        result = np.zeros(self._image_shape, dtype=np.uint8)
        result[self._horizon:] = (resized * 255).astype(np.uint8)

        return result

    def _do_process(self, color_image: np.ndarray, depth_image: np.ndarray) -> np.ndarray:
        h, w = self._image_shape

        if color_image.shape != (h, w, 3):
            raise ValueError(f"Invalid color image shape: {color_image.shape} != {(h, w, 3)}")
        
        if depth_image.shape != (h, w):
            raise ValueError(f"Invalid depth image shape: {depth_image.shape} != {(h, w)}")

        X = self._resize_for_inference(color_image)
        y = self._inference_session.run(X)

        if y.shape != (*self._input_shape[:3], 1):
            raise ValueError(f"Invalid output shape: {y.shape} != {(*self._input_shape[:3], 1)}")
        
        return self._restore_shape_after_inference(y[0])

