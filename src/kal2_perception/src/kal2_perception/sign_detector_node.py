from collections import Counter, deque
from pathlib import Path

import cv2 as cv
import numpy as np

from scipy.ndimage import binary_fill_holes
from openvino.runtime import Core

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import pytesseract

from kal2_perception.preprocessing import OpenVinoInferenceSession, OnnxInferenceSession
from kal2_perception.ocr import TesseractCityDetector, CRNNCityDetector, crop_to_bbox
from kal2_control.state_machine import TurningDirection, Zone
from kal2_util.node_base import NodeBase

from kal2_msgs.msg import DetectedSign
from kal2_msgs.msg import MainControllerState # type: ignore



def apply_hsv_threshold(
    image: np.ndarray,
    min_area: int = 4000,
    lower: np.ndarray = np.array([15, 100, 100]),
    upper: np.ndarray = np.array([50, 255, 255]),
):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hsv[:100] = (0, 0, 0)  # ignore upper most part of the image

    mask = cv.inRange(hsv, lower, upper)
    mask = binary_fill_holes(mask).astype(np.uint8) * 255
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    region_proposals = []
    areas = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > min_area:
            rect = cv.minAreaRect(cnt)
            region_proposals.append(rect)
            areas.append(area)
    return region_proposals, areas


def increase_sharpness(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv.filter2D(image, -1, kernel)


def increase_contrast(image):
    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv.merge((cl, a, b))
    return cv.cvtColor(limg, cv.COLOR_LAB2BGR)


class SignDetectorNode(NodeBase):
    def __init__(self):
        super().__init__(name="sign_detector_node")
        rospy.loginfo("Starting sign detector node...")

        self.bridge = CvBridge()

        self._inference_session = OpenVinoInferenceSession(
            model_path=Path(self.params.model_path), input_shape=(1, 224, 224, 3)
        )

        if self.params.use_tesseract:
            rospy.loginfo("Using tesseract.")
            self._city_detector = TesseractCityDetector(threshold=self.params.tessaract_binary_treshold)
        else:
            rospy.loginfo("Using CRNN + CRAFT.")
            self._city_detector = CRNNCityDetector(self.params.ocr_model_path, self.params.craft_model_path)

        self._predictions = deque(maxlen=self.params.prediction_queue_size)
        self._current_zone = Zone.RecordedZone

        self._hsv_lower = np.array(self.params.hsv_lower)
        self._hsv_upper = np.array(self.params.hsv_upper)
        self._min_area = self.params.min_area

        self._result_publisher = rospy.Publisher("/kal2/detected_sign", DetectedSign, queue_size=10)
        self._debug_publisher = rospy.Publisher("/kal2/debug/signs", Image, queue_size=10)
        self._debug_ocr_publisher = rospy.Publisher("/kal2/debug/signs_ocr", Image, queue_size=10)
        self._image_subscriber = rospy.Subscriber(
            "/camera_front/color/image_raw", Image, self._image_callback, queue_size=1
        )
        self._state_subscriber = rospy.Subscriber("/kal2/main_controller_state", MainControllerState, self._state_callback)


        rospy.loginfo("Sign detector node started.")

    def _image_callback(self, color_image_msg: Image) -> None:
        if self._current_zone != Zone.SignDetectionZone and not self.params.detect_always:
            return

        color_image = self.bridge.imgmsg_to_cv2(color_image_msg, desired_encoding="bgr8")
        city, direction = self.process_image(color_image)

        if city != "unknown":
            rospy.loginfo(f"Saw {city, direction.value}")
            self._predictions.append((city, direction))

        if len(self._predictions) == 0:
            return

        (predicted_city, predicted_direction), count = Counter(self._predictions).most_common(1)[0]

        if count >= 2:
            self._publish_result(predicted_city, predicted_direction)

        rospy.loginfo(f"Most common prediction: {predicted_city}, {predicted_direction.value}")

    def _state_callback(self, msg: MainControllerState):
        try:
            zone = Zone(msg.current_zone)
        except ValueError as e:
            self._is_initialized = False
            return

        self._current_zone = zone


    def _publish_debug_image(self, image, boxes, largest_box):
        debug_image = image.copy()
        for box in boxes:
            box = cv.boxPoints(box)
            cv.drawContours(debug_image, [box.astype(int)], contourIdx=0, color=(0, 0, 255), thickness=5)

        box = cv.boxPoints(largest_box)
        cv.drawContours(debug_image, [box.astype(int)], contourIdx=0, color=(0, 255, 0), thickness=5)

        debug_image = self.bridge.cv2_to_imgmsg(debug_image)
        self._debug_publisher.publish(debug_image)
        #self._predictions.clear()

    def _publish_result(self, predicted_city, predicted_direction):
        result_msg = DetectedSign(city=predicted_city, direction=predicted_direction.value)
        self._result_publisher.publish(result_msg)

    def run_inference(self, image):
        input_image = cv.resize(image, (224, 224))
        input_image = input_image.astype(np.float32) / 255.0
        input_image = np.expand_dims(input_image, axis=0)
        result = self._inference_session.run(input_image)
        return result

    def process_image(self, image):
        boxes, areas = apply_hsv_threshold(image, self._min_area, self._hsv_lower, self._hsv_upper)
        if len(boxes) == 0:
            return "unknown", TurningDirection.Unknown

        indices = np.argsort(areas)[::-1]
        largest_box = boxes[indices[0]]

        self._publish_debug_image(image, boxes, largest_box)

        detected_city = self.detect_city(image, largest_box)
        if detected_city and detected_city != "unknown":
            rospy.loginfo(f"Detected city: {detected_city}")

        cropped_image = crop_to_bbox(image, largest_box, scale_factor=1.5)
        if cropped_image is None or cropped_image.size == 0:
            rospy.loginfo("Empty cropped image.")
            return "unknown", TurningDirection.Unknown

        inference_result = self.run_inference(cropped_image)
        predicted_class = np.argmax(inference_result)

        return detected_city, TurningDirection.Left if predicted_class == 0 else TurningDirection.Right

    def detect_city(self, image, rect):
        try:
            text, cropped_image = self._city_detector.detect(image=image, box=rect)
        except (RuntimeError, ValueError) as e:
            rospy.logerr(e)
            return "unknown"

        debug_image = self.bridge.cv2_to_imgmsg(cropped_image)
        self._debug_ocr_publisher.publish(debug_image)

        city_names = ["Munchen", "Karlsruhe", "Koln", "Hildesheim"]

        def count_common_letters(ref: str, other: str) -> int:
            return len(set(ref.lower()).intersection(set(other.lower())))

        common_letter_count = [count_common_letters(city, text) for city in city_names]
        indices = np.argsort(common_letter_count)[::-1]

        if common_letter_count[indices[0]] == common_letter_count[indices[1]]:
            if common_letter_count[indices[0]] == 0:
                return "unknown"
        
            best_match = None
            lowest_diff = 1000
            for city in city_names:
                diff = len(city) - len(text)
                if diff < lowest_diff:
                    best_match = city
                    lowest_diff = diff
            if lowest_diff != 1000:
                return best_match
            
            return "unknown"
        else:
            if city_names[indices[0]] == city_names[-1] and len(text) < 5:
                return city_names[2]
            return city_names[indices[0]]
