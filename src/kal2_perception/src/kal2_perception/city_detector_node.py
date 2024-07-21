#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
import message_filters
from cv_bridge import CvBridge
from kal2_perception.node_base import NodeBase
from tf2_ros import TransformListener, Buffer
import cv2 as cv
import numpy as np
from scipy.ndimage import binary_fill_holes
import pytesseract

class CityDetectorNode(NodeBase):
    def __init__(self) -> None:
        super().__init__(name="city_detector_node")
        rospy.loginfo("Starting city detector node...")

        self._color_image_subscriber = message_filters.Subscriber("/camera_front/color/image_raw", Image)
        self._image_subscriber = message_filters.TimeSynchronizer([self._color_image_subscriber], queue_size=10)
        self._image_subscriber.registerCallback(self._image_callback)

        self._cv_bridge = CvBridge()
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer)

        self._result_publisher = rospy.Publisher("/detected_city", String, queue_size=10)

    def _image_callback(self, color_image_msg: Image) -> None:
        color_image = self._cv_bridge.imgmsg_to_cv2(color_image_msg, desired_encoding="bgr8")
        city_name = self.process_image(color_image)
        self._result_publisher.publish(city_name)

    def process_image(self, image):
        hsv, boxes = self.apply_hsv_threshold(image)
        if len(boxes) == 0:
            return "Unknown"
        
        cropped_image = self.crop_to_bbox(image, boxes[0])
        contrast_image = self.increase_contrast(cropped_image)
        sharp_image = self.increase_sharpness(contrast_image)
        binary_image = self.binarize_image(sharp_image)
        text = self.perform_ocr(binary_image)

        city_names = ['Munchen', 'Karlsruhe', 'Koln', 'Hildesheim']
        for city in city_names:
            if any(city[i:i+3].lower() in text.lower() for i in range(len(city) - 2)):
                return city
        return "Unknown"

    def apply_hsv_threshold(self, image):
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        lower = np.array([15, 100, 100]) 
        upper = np.array([35, 255, 255]) 
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

    def crop_to_bbox(self, image, rect):
        box = cv.boxPoints(rect).astype(np.float32)
        center = np.mean(box, axis=0)
        sorted_indices = np.argsort(box[:, 1])
        lowest_idx = sorted_indices[0]
        second_lowest_idx = sorted_indices[1]
        a = box[lowest_idx]
        b = box[second_lowest_idx]
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
        box = np.int0(cv.transform(np.array([box]), M))[0]
        x, y, w, h = cv.boundingRect(box)
        cropped = rotated[y:y+h, x:x+w]
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

if __name__ == "__main__":
    node = CityDetectorNode()
    rospy.spin()
