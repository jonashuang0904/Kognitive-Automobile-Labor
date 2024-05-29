import numpy as np
import cv2 as cv
import open3d as o3d
from dataclasses import dataclass

import seaborn as sns
import matplotlib.pyplot as plt

import rospy
from sensor_msgs.msg import Image

import message_filters
from cv_bridge import CvBridge

from kal2_perception.node_base import NodeBase
from kal2_perception.birds_eye_view import BevTransformer, BevRoi

from tf2_ros import TransformListener, Buffer
from ros_numpy import numpify


def line_kernel(width: int, rows: int):
    kernel = np.ones((rows, width * 3)) * 2
    kernel[:, :width] = -1
    kernel[:, -width:] = -1
    kernel /= np.sum(np.abs(kernel))

    return kernel


class LaneDector:
    def __init__(self, intrinsic_matrix: np.ndarray, extrinsic_matrix: np.ndarray, region_of_interest: BevRoi):
        self._intrinsic_matrix = intrinsic_matrix
        self._extrinsic_matrix =extrinsic_matrix
        self.region_of_interest = region_of_interest

        self._bev_transformer = BevTransformer.from_roi(region_of_interest, intrinsic_matrix, extrinsic_matrix, scale=240)

        self._cv_bridge = CvBridge()
        self._publisher = rospy.Publisher("/debug/bev", Image, queue_size=10)
        self._publisher2 = rospy.Publisher("/debug/bev2", Image, queue_size=10)

    def detect(self, color_image: np.ndarray, depth_image: np.ndarray, timestamp: rospy.Time) -> np.ndarray:
        # Implement lane detection here

        gray_image = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)
        bev_image = self._bev_transformer.transform(gray_image).astype(np.float32)
                
        activation_image = cv.filter2D(bev_image, -1, line_kernel(5, 3)) * 2.0
        activation_image[activation_image < 0.0] = 0

        edge_image = cv.Canny(gray_image, 50, 150)
        edge_image = self._bev_transformer.transform(edge_image)

        self._publisher.publish(self._cv_bridge.cv2_to_imgmsg(activation_image.astype(np.uint8), encoding="mono8"))
        self._publisher2.publish(self._cv_bridge.cv2_to_imgmsg(edge_image, encoding="mono8"))

        return None

class LaneDetectorNode(NodeBase):
    def __init__(self) -> None:
        super().__init__(name="lane_detector_node")

        self._color_image_subscriber = message_filters.Subscriber(self.params.color_image_topic, Image)
        self._depth_image_subscriber = message_filters.Subscriber(self.params.depth_image_topic, Image)

        self._image_subscriber = message_filters.TimeSynchronizer(
            [self._color_image_subscriber, self._depth_image_subscriber], queue_size=10
        )
        self._image_subscriber.registerCallback(self._image_callback)

        self._cv_bridge = CvBridge()

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer)

        transform = self._tf_buffer.lookup_transform(source_frame="vehicle_rear_axle", target_frame="camera_front_aligned_depth_to_color_frame", time=rospy.Time(), timeout=rospy.Duration(1.0))
        extrinsic_matrix = numpify(transform.transform)
        intrinsic_matrix = np.array([[913.75, 0.0, 655.42], [0.0, 911.92, 350.49], [0.0, 0.0, 1.0]])
        region_of_interest = BevRoi(min_distance=0.5, max_distance=3, width=3)
        self._lane_detector = LaneDector(intrinsic_matrix=intrinsic_matrix, extrinsic_matrix=extrinsic_matrix, region_of_interest=region_of_interest)

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def _image_callback(self, color_image_msg: Image, depth_image_msg: Image) -> None:
        color_image = self._cv_bridge.imgmsg_to_cv2(color_image_msg, desired_encoding="bgr8")
        depth_image = self._cv_bridge.imgmsg_to_cv2(depth_image_msg, desired_encoding="16UC1")

        path = self._lane_detector.detect(color_image, depth_image, color_image_msg.header.stamp)
