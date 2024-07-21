#!/usr/bin/env python3

from collections import Counter

import cv2 as cv
import numpy as np

from scipy.ndimage import binary_fill_holes
from openvino.runtime import Core

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import pytesseract

from kal2_msgs.msg import DetectedSign


class SignDetectorNode:
    def __init__(self):
        rospy.init_node("sign_detector_node", anonymous=True)
        rospy.loginfo("Starting sign detector node...")

        self.bridge = CvBridge()
        self.core = Core()
        self.model_path = rospy.get_param("~model_path")
        self.compiled_model, self.input_layer = self.load_model(self.model_path)
        self.predictions = []
        self.detected_city = "unknown"

        self.image_sub = rospy.Subscriber("/camera_front/color/image_raw", Image, self._image_callback)
        self._result_publisher = rospy.Publisher("/kal2/detected_sign", DetectedSign, queue_size=10)
        self._debug_publisher = rospy.Publisher("/kal2/debug/signs", Image, queue_size=10)

    def _image_callback(self, color_image_msg: Image) -> None:
        color_image = self.bridge.imgmsg_to_cv2(color_image_msg, desired_encoding="bgr8")
        detected_sign = self.process_image(color_image)
        if detected_sign and detected_sign != "unknown":
            self.publish_result(detected_sign)

    def load_model(self, model_path):
        model = self.core.read_model(model_path)
        compiled_model = self.core.compile_model(model, device_name="CPU")
        input_layer = compiled_model.input(0)
        return compiled_model, input_layer

    def apply_hsv_threshold(self, image):
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        hsv[:100] = (0, 0, 0)
        lower = np.array([15, 100, 100])
        upper = np.array([50, 255, 255])
        mask = cv.inRange(hsv, lower, upper)
        mask = binary_fill_holes(mask).astype(np.uint8) * 255
        result = cv.bitwise_and(image, image, mask=mask)
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        region_proposals = []
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area > 4000:
                rect = cv.minAreaRect(cnt)
                box = cv.boxPoints(rect)
                cv.drawContours(result, [box.astype(int)], 0, (0, 0, 255), thickness=5)
                region_proposals.append(rect)
        return result, region_proposals

    def crop_to_bbox(self, image, rect, scale_factor=1.0):
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

    def increase_sharpness(self, image):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv.filter2D(image, -1, kernel)

    def increase_contrast(self, image):
        lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
        l, a, b = cv.split(lab)
        clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv.merge((cl, a, b))
        return cv.cvtColor(limg, cv.COLOR_LAB2BGR)

    def binarize_image(self, image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        return binary

    def perform_ocr(self, image):
        return pytesseract.image_to_string(image)

    def run_inference(self, image):
        input_image = cv.resize(image, (224, 224))
        input_image = input_image.astype(np.float32) / 255.0
        input_image = np.expand_dims(input_image, axis=0)
        result = self.compiled_model([input_image])[self.compiled_model.output(0)]
        return result

    def process_image(self, image):
        hsv_result, boxes = self.apply_hsv_threshold(image)
        if len(boxes) == 0:
            rospy.loginfo("No valid regions found in the image.")
            return "unknown"
        
        debug_image = image.copy()
        for box in boxes:
            box = cv.boxPoints(box)
            cv.drawContours(debug_image, [box.astype(int)], contourIdx=0, color=(0, 0, 255), thickness=5)
        debug_image = self.bridge.cv2_to_imgmsg(debug_image)
        self._debug_publisher.publish(debug_image)


        self.detected_city = self.detect_city(image, boxes[0])
        if self.detected_city and self.detected_city != "unknown":
            rospy.loginfo(f"Detected city: {self.detected_city}")

        cropped_image = self.crop_to_bbox(image, boxes[0], scale_factor=1.5)
        if cropped_image is None or cropped_image.size == 0:
            rospy.loginfo("Empty cropped image.")
            return "unknown"

        inference_result = self.run_inference(cropped_image)
        predicted_class = np.argmax(inference_result)

        if self.detected_city != "unknown":
            self.predictions.append(predicted_class)
        #rospy.loginfo(f"inference: {inference_result}")
        #rospy.loginfo(f"Predicted class: {predicted_class}")
        if predicted_class == 0:
            return "left"
        elif predicted_class == 1:
            return "right"
        else:
            return "unknown"

    def detect_city(self, image, rect):
        cropped_image = self.crop_to_bbox(image, rect, scale_factor=1.0)
        contrast_image = self.increase_contrast(cropped_image)
        sharp_image = self.increase_sharpness(contrast_image)
        binary_image = self.binarize_image(sharp_image)
        text = self.perform_ocr(binary_image)
        city_names = ["Munchen", "Karlsruhe", "Koln", "Hildesheim"]
        for city in city_names:
            if any(city[i : i + 3].lower() in text.lower() for i in range(len(city) - 2)):
                return city
        return "unknown"

    def publish_result(self, detected_sign):
        if len(self.predictions) == 0:
            rospy.loginfo("No predictions made.")
            return

        most_common_prediction = Counter(self.predictions).most_common(1)[0][0]
        direction = "left" if most_common_prediction == 0 else "right"
        result_msg = DetectedSign(city=self.detected_city, direction=direction)

        self._result_publisher.publish(result_msg)
        rospy.loginfo(f"Most common predicted direction: {direction}")
        self.predictions = []

    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            rate.sleep()


if __name__ == "__main__":
    node = SignDetectorNode()
    node.run()
