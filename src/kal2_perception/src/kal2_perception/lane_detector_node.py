import numpy as np
from functools import cached_property
from pathlib import Path

import cv2 as cv
import rospy
import rospkg
from sensor_msgs.msg import Image

import message_filters
from cv_bridge import CvBridge

from kal2_perception.node_base import NodeBase
from kal2_perception.birds_eye_view import PerspectiveBevTransformer, PointcloudBevTransformer, BevRoi
from kal2_perception.lane_tracking import LaneMap
from kal2_perception.preprocessing import UnetPreprocessor
from kal2_perception.feature_extraction import HoughFeatureExtractor, transform_features
from kal2_perception.camera import CameraInfo
from kal2_perception.visualization import LaneFeatureOverlay, ProjectedErrorsOverlay, LaneSegmentOverlay
from kal2_perception.util import convert_images_to_arrays

from kal2_util.transforms import TransformProvider


class LaneDetectorNode(NodeBase):
    def __init__(self) -> None:
        super().__init__(name="lane_detector_node")
        rospy.loginfo("Starting lane detector node...")

        self._rospack = rospkg.RosPack()
        self._transform_provider = TransformProvider()
        self._camera_info = CameraInfo.from_ros_topic(self.params.color_image_info_topic)
        tf_camera_to_vehicle = self._transform_provider.get_transform_camera_to_vehicle()

        model_path = self.package_path / self.params.model_path
        self._image_preprocessor = UnetPreprocessor(model_path=model_path, runtime="openvino")
        self._bev_transformer = PointcloudBevTransformer(
            camera_info=self._camera_info, extrinsic_matrix=tf_camera_to_vehicle
        )
        self._feature_extractor = HoughFeatureExtractor()
        self._lane_map =LaneMap(initial_pose=np.array((3, 0.5, 0)))

        self._cv_bridge = CvBridge()
        self._debug_publisher_preprocessed = rospy.Publisher("/kal2/debug/preprocessed_image", Image, queue_size=10)
        self._debug_publisher_bev = rospy.Publisher("/kal2/debug/bev_image", Image, queue_size=10)
        self._debug_publisher_features = rospy.Publisher("/kal2/debug/features_image", Image, queue_size=10)

        self._color_image_subscriber = message_filters.Subscriber(self.params.color_image_topic, Image)
        self._depth_image_subscriber = message_filters.Subscriber(self.params.depth_image_topic, Image)

        self._image_subscriber = message_filters.TimeSynchronizer(
            [self._color_image_subscriber, self._depth_image_subscriber], queue_size=10
        )
        self._image_subscriber.registerCallback(self._image_callback)

        rospy.loginfo("Lane detector node started.")

    @cached_property
    def package_path(self) -> Path:
        return Path(self._rospack.get_path("kal2_perception"))

    def _publish_image(self, image: np.ndarray, publisher: rospy.Publisher, encoding: str) -> None:
        image_msg = self._cv_bridge.cv2_to_imgmsg(image, encoding=encoding)
        publisher.publish(image_msg)

    @convert_images_to_arrays("bgr8", "16UC1")
    def _image_callback(self, color_image: np.ndarray, depth_image: np.ndarray) -> None:
        # 1. Preprocessing: Convert color image into a binary mask
        preprocessed_image = self._image_preprocessor.process(color_image, depth_image)
        rospy.logdebug(f"Preprocessing took {self._image_preprocessor.get_last_duration():.3f}s")

        # 2. Transform the binary mask into a bird's eye view image
        depth_image[preprocessed_image < 100] = 0
        bev_image = self._bev_transformer.transform(color_image, depth_image)

        # 3. Extract features from the bird's eye view image
        features = self._feature_extractor.extract(bev_image)
        tf_vehicle_to_world = self._transform_provider.named_transform("vehicle_to_world")
        transformed_features = transform_features(features, self._bev_transformer.intrinsic_matrix, tf_vehicle_to_world)

        # 4. Track lanes
        # TODO: Implement lane tracking
        grid = self._lane_map.update(transformed_features, vehicle_pose=tf_vehicle_to_world)

        # 5. Publish debug images
        if self.params.publish_debug_images:
            self._publish_image(preprocessed_image, self._debug_publisher_preprocessed, "mono8")
            self._publish_image(grid, self._debug_publisher_bev, "mono8")

            # timestamp = rospy.Time.now()
            # cv.imwrite(str(self.package_path/f"notebooks/bev_images/bev_image_{timestamp}.png"), bev_image)

            tf_world_to_camera = self._transform_provider.named_transform("world_to_camera")
            overlay = LaneFeatureOverlay(self._camera_info.intrinsic_matrix, tf_world_to_camera)
            color_image = overlay.draw(color_image, transformed_features)

            proj_overlay = LaneSegmentOverlay(self._camera_info.intrinsic_matrix, tf_world_to_camera)
            color_image = proj_overlay.draw(color_image, self._lane_map._lane_segments)
            self._publish_image(color_image, self._debug_publisher_features, "bgr8")
