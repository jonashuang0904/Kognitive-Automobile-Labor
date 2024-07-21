from abc import ABC, abstractmethod
from typing import Tuple
from pathlib import Path
import string

import numpy as np
import cv2 as cv
import pytesseract

import rospy
import openvino as ov

from kal2_perception.preprocessing import OpenVinoInferenceSession


def crop_to_bbox(image: np.ndarray, rect, scale_factor=1.0):
    box = cv.boxPoints(rect).astype(np.float32)
    center = np.mean(box, axis=0)
    scaled_box = np.array([center + (point - center) * scale_factor for point in box], dtype=np.float32)
    sorted_indices = np.argsort(scaled_box[:, 1])
    lowest_idx = sorted_indices[0]
    second_lowest_idx = sorted_indices[1]
    a = scaled_box[lowest_idx]
    b = scaled_box[second_lowest_idx]
    ab = b - a
    angle_to_x_positive = np.arctan2(ab[1], ab[0])
    angle_to_x_negative = np.arctan2(ab[1], ab[0]) - np.pi if ab[1] != 0 else np.pi
    angle = angle_to_x_positive if abs(angle_to_x_positive) < abs(angle_to_x_negative) else angle_to_x_negative
    M = cv.getRotationMatrix2D(tuple(a), np.degrees(angle), 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_width = int(image.shape[1] * cos + image.shape[0] * sin)
    new_height = int(image.shape[1] * sin + image.shape[0] * cos)
    M[0, 2] += (new_width / 2) - a[0]
    M[1, 2] += (new_height / 2) - a[1]
    rotated = cv.warpAffine(image, M, (new_width, new_height))
    box = (cv.transform(np.array([box]), M)).astype(np.int64)[0]
    x, y, w, h = cv.boundingRect(box)
    cropped = rotated[y : y + h, x : x + w]
    return cropped


class BaseCityDetector:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def detect(self, image: np.ndarray, box: np.ndarray) -> Tuple[str, np.ndarray]:
        pass


class TesseractCityDetector(BaseCityDetector):
    def __init__(self, threshold: int = 100) -> None:
        self._treshold = threshold

    def detect(self, image: np.ndarray, box: np.ndarray) -> Tuple[str, np.ndarray]:
        cropped_image = crop_to_bbox(image, box, scale_factor=0.5)
        cropped_image = cv.cvtColor(cropped_image, cv.COLOR_BGR2GRAY)

        _, binary_image = cv.threshold(image, self._treshold, 255, cv.THRESH_BINARY)
        text = pytesseract.image_to_string(binary_image)

        return text.replace("\n", "").replace(" ", ""), cropped_image


class CRNNCityDetector(BaseCityDetector):
    def __init__(self, model_path: Path, craft_model_path: Path, input_shape=(1, 31, 200, 1)) -> None:
        self._inference_session = OpenVinoInferenceSession(model_path=model_path, input_shape=input_shape)
        self._input_shape = input_shape
        self._use_craft = craft_model_path is not None

        self._alphabet = string.digits + string.ascii_lowercase

        if craft_model_path is not None:
            self._core = ov.Core()
            ov_model = ov.convert_model(craft_model_path)
            self._model =  self._core.compile_model(ov_model, "CPU")

            #ov_model = ov.convert_model(craft_model_path)
            #self._crnn = self._core.compile_model(ov.convert_model(model_path), "CPU")


    def _find_roi(self, cropped_image: np.ndarray) -> np.ndarray:
        input_image = cropped_image.copy()
        input_image = input_image / 255.0
        try:
            res = self._model.infer_new_request({"input_1": np.expand_dims(input_image, 0)})[0]
        except RuntimeError as e:
            # rospy.logerr(e)
            rospy.logerr(input_image.shape)
            return cropped_image

        thresh = (res[0, ..., 0] > 0.5).astype(np.uint8)  * 255
        thresh = thresh[..., np.newaxis]
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return cropped_image

        boxes = []
        for contour in contours:
            rect = cv.minAreaRect(contour)
            box = cv.boxPoints(rect)
            boxes.append(box)

        boxes = np.concatenate(boxes).astype(int)
        x_min = boxes[:, 0].min() * 2 - 10
        x_max = boxes[:, 0].max() * 2 + 10
        y_min = boxes[:, 1].min() * 2 - 10
        y_max = boxes[:, 1].max() * 2 + 10
        cropped_image = cropped_image[y_min:y_max, x_min:x_max]

        return cropped_image

    def detect(self, image: np.ndarray, box: np.ndarray) -> Tuple[str, np.ndarray]:
        cropped_image = crop_to_bbox(image, box)

        if self._use_craft:
            cropped_image = self._find_roi(cropped_image)
            
            if cropped_image.shape[0] < 20 or cropped_image.shape[1] < 80:
                rospy.logwarn_throttle(1, f"Image too small: {cropped_image.shape}")
                return "", cropped_image

        _, h, w, _ = self._input_shape

        if h == 0 or w == 0 or cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
            return "", cropped_image
        
        input_image = np.ones((h*3, w*3, 3)) * 255
        input_image[:cropped_image.shape[0], :cropped_image.shape[1], :] = cropped_image
        input_image = cv.resize(cropped_image, (w, h))
        #input_image = cropped_image
        input_image = input_image.astype(np.uint8)
        input_image = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)[..., np.newaxis]
        input_image = input_image.astype(np.float64) / 255.0
        #result = self._crnn.infer_new_request(np.expand_dims(input_image, 0))[0]
        result = self._inference_session.run(np.expand_dims(input_image, 0))
        
        text = "".join([self._alphabet[idx] for idx in result[0].astype(int) if idx not in [len(self._alphabet), -1]])
        return text, (255 * input_image).astype(np.uint8)
