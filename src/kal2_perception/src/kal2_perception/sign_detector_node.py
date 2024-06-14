import rospy
from sensor_msgs.msg import Image

import message_filters
from cv_bridge import CvBridge

from kal2_perception.node_base import NodeBase

from tf2_ros import TransformListener, Buffer, LookupException, ExtrapolationException


class SignDetectorNode(NodeBase):
    def __init__(self) -> None:
        super().__init__(name="sign_detector_node")
        rospy.loginfo("Starting sign detector node...")

        self._color_image_subscriber = message_filters.Subscriber(self.params.color_image_topic, Image)
        self._depth_image_subscriber = message_filters.Subscriber(self.params.depth_image_topic, Image)

        self._image_subscriber = message_filters.TimeSynchronizer(
            [self._color_image_subscriber, self._depth_image_subscriber], queue_size=10
        )
        self._image_subscriber.registerCallback(self._image_callback)

        self._cv_bridge = CvBridge()
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer)

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def _image_callback(self, color_image_msg: Image, depth_image_msg: Image) -> None:
        color_image = self._cv_bridge.imgmsg_to_cv2(color_image_msg, desired_encoding="bgr8")
        depth_image = self._cv_bridge.imgmsg_to_cv2(depth_image_msg, desired_encoding="16UC1")
