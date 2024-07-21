#! /usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from kal2_msgs.msg import sign
from geometry_msgs.msg import Vector3
from cv_bridge import CvBridge
from collections import Counter
import cv2 as cv
import numpy as np
from scipy.ndimage import binary_fill_holes
from openvino.runtime import Core

class DirectionDetectorNode:
    def __init__(self):
        rospy.init_node('direction_detector_node', anonymous=True)
        rospy.loginfo("Starting direction detector node...")

        self.bridge = CvBridge()
        self.core = Core()
        self.model_path = rospy.get_param('~model_path')
        self.compiled_model, self.input_layer = self.load_model(self.model_path)
        self.predictions = []

        self.image_sub = rospy.Subscriber('/camera_front/color/image_raw', Image, self._image_callback)
        self._result_publisher = rospy.Publisher("/detected_direction", String, queue_size=10)

    def _image_callback(self, color_image_msg: Image) -> None:
        color_image = self.bridge.imgmsg_to_cv2(color_image_msg, desired_encoding="bgr8")
        direction = self.process_image(color_image)
        
        self._result_publisher.publish(direction)

    def load_model(self, model_path):
        model = self.core.read_model(model_path)
        compiled_model = self.core.compile_model(model, device_name="CPU")
        input_layer = compiled_model.input(0)
        return compiled_model, input_layer

    def apply_hsv_threshold(self, image):
        hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)
        lower = np.array([15, 100, 100])
        upper = np.array([100, 255, 255])
        mask = cv.inRange(hsv, lower, upper)

        mask = (
            binary_fill_holes(
                mask,
            ).astype(np.uint8)
            * 255
        )

        result = cv.bitwise_and(image, image, mask=mask)
        contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        region_proposals = []

        for cnt in contours:
            area = cv.contourArea(cnt)

            if area > 4000:
                rect = cv.minAreaRect(cnt)
                box = cv.boxPoints(rect)
                cv.drawContours(result, [box.astype(int)], 0, (0, 0, 255), thickness=5)
                region_proposals.append(rect)

        return result, region_proposals

    def crop_to_bbox(self, image, rect, scale_factor=1.5):
        box = cv.boxPoints(rect).astype(np.float32)
        center = np.mean(box, axis=0)

        scaled_box = np.array([
            center + (point - center) * scale_factor for point in box
        ], dtype=np.float32)

        sorted_indices = np.argsort(scaled_box[:, 1])
        lowest_idx = sorted_indices[0]
        second_lowest_idx = sorted_indices[1]
        a = scaled_box[lowest_idx]
        b = scaled_box[second_lowest_idx]
        
        ab = b - a
        angle_to_x_positive = np.degrees(np.arctan2(ab[1], ab[0]))
        angle_to_x_negative = angle_to_x_positive - 180 if angle_to_x_positive > 0 else angle_to_x_positive + 180
        
        if abs(angle_to_x_positive) < abs(angle_to_x_negative):
            angle = angle_to_x_positive
        else:
            angle = angle_to_x_negative

        M = cv.getRotationMatrix2D(tuple(center), angle, 1.0)

        rotated = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))

        rotated_scaled_box = cv.transform(np.array([scaled_box]), M)[0]
        rotated_scaled_box = np.intp(rotated_scaled_box)

        x, y, w, h = cv.boundingRect(rotated_scaled_box)

        x = max(x, 0)
        y = max(y, 0)
        w = min(w, rotated.shape[1] - x)
        h = min(h, rotated.shape[0] - y)

        if w == 0 or h == 0:
            print("Warning: Width or height of the bounding box is zero.")
            return None

        cropped = rotated[y:y+h, x:x+w]
        
        return cropped

    def run_inference(self, image):
        input_image = cv.resize(image, (224, 224))
        input_image = input_image.astype(np.float32) / 255.0  # Normalize to [0, 1]
        input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

        result = self.compiled_model([input_image])[self.compiled_model.output(0)]
        return result

    def process_image(self, image):
        hsv_result, boxes = self.apply_hsv_threshold(image)
        if len(boxes) == 0:
            rospy.loginfo("No valid regions found in the image.")
            return "unknown"

        cropped_image = self.crop_to_bbox(image, boxes[0])
        if cropped_image is None or cropped_image.size == 0:
            rospy.loginfo("Empty cropped image.")
            return "unknown"

        inference_result = self.run_inference(cropped_image)
        predicted_class = np.argmax(inference_result)
        self.predictions.append(predicted_class)

        rospy.loginfo(f"Predicted class: {predicted_class}")

        if predicted_class == 0:
            return "left"
        elif predicted_class == 1:
            return "right"
        else:
            return "unknown"

    def publish_result(self):
        if len(self.predictions) == 0:
            rospy.loginfo("No predictions made.")
            return

        most_common_prediction = Counter(self.predictions).most_common(1)[0][0]
        direction = "left" if most_common_prediction == 0 else "right"
        self._result_publisher.publish(direction)
        rospy.loginfo(f"Most common predicted direction: {direction}")
        self.predictions = []

    def run(self):
        rate = rospy.Rate(1)  # 1 Hz
        while not rospy.is_shutdown():
            self.publish_result()
            rate.sleep()

if __name__ == "__main__":
    node = DirectionDetectorNode()
    node.run()
