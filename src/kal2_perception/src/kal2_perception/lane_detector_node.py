import numpy as np
from functools import cached_property
from pathlib import Path
import time

import cv2 as cv
import rospy
import rospkg
from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import Path as NavPath
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Header

import message_filters
from cv_bridge import CvBridge

from tf2_ros import ExtrapolationException

from kal2_perception.node_base import NodeBase
from kal2_perception.birds_eye_view import PerspectiveBevTransformer, PointcloudBevTransformer, BevRoi
# from kal2_perception.lane_tracking import LaneMap
from kal2_perception.lane_tracking_v2 import LaneMap
from kal2_perception.preprocessing import UnetPreprocessor
from kal2_perception.feature_extraction import HoughFeatureExtractor, transform_features
from kal2_perception.camera import CameraInfo
from kal2_perception.visualization import LaneFeatureOverlay, ProjectedErrorsOverlay, LaneSegmentOverlay, LaneMapOverlay
from kal2_perception.util import convert_images_to_arrays, create_point_cloud_msg

from kal2_util.transforms import TransformProvider, CoordinateFrames


class LaneDetectorNode(NodeBase):
    def __init__(self) -> None:
        super().__init__(name="lane_detector_node")
        rospy.loginfo("Starting lane detector node...")

        self._rospack = rospkg.RosPack()
        self._transform_provider = TransformProvider()
        self._camera_info = CameraInfo.from_ros_topic(self.params.color_image_info_topic)
        # self._transform_provider.wait_for_transform(source_frame=CoordinateFrames.WORLD, target_frame=CoordinateFrames.VEHICLE)
        tf_camera_to_vehicle = self._transform_provider.get_transform_camera_to_vehicle()

        model_path = self.package_path / self.params.model_path
        self._image_preprocessor = UnetPreprocessor(model_path=model_path, runtime="openvino")
        self._bev_transformer = PointcloudBevTransformer(
            camera_info=self._camera_info, extrinsic_matrix=tf_camera_to_vehicle
        )
        self._feature_extractor = HoughFeatureExtractor()

        tf_vehicle_to_world = self._transform_provider.named_transform("vehicle_to_world")
        t = tf_camera_to_vehicle[:2, 3]
        theta = np.arctan2(tf_camera_to_vehicle[1, 0], tf_camera_to_vehicle[0, 0])
        self._lane_map = LaneMap(initial_pose=np.array((t[0], 0.5, np.pi)))

        self._cv_bridge = CvBridge()
        self._debug_publisher_preprocessed = rospy.Publisher("/kal2/debug/preprocessed_image", Image, queue_size=10)
        self._debug_publisher_bev = rospy.Publisher("/kal2/debug/bev_image", Image, queue_size=10)
        # self._debug_publisher_features = rospy.Publisher("/kal2/debug/features_image", Image, queue_size=10)
        self._debug_publisher_features = rospy.Publisher("/kal2/debug/features", PointCloud2, queue_size=10)
        self._local_features_publisher = rospy.Publisher("/kal2/local_features", PointCloud2, queue_size=10)

        self._center_line_publisher = rospy.Publisher("/kal2/center_line", NavPath, queue_size=10)
        self._trajectory_publisher = rospy.Publisher("/kal2/trajectory", NavPath, queue_size=10)

        self._color_image_subscriber = message_filters.Subscriber(self.params.color_image_topic, Image)
        self._depth_image_subscriber = message_filters.Subscriber(self.params.depth_image_topic, Image)

        self._image_subscriber = message_filters.TimeSynchronizer(
            [self._color_image_subscriber, self._depth_image_subscriber], queue_size=2
        )
        self._image_subscriber.registerCallback(self._image_callback)

        rospy.loginfo("Lane detector node started.")

    @cached_property
    def package_path(self) -> Path:
        return Path(self._rospack.get_path("kal2_perception"))

    def _publish_image(self, image: np.ndarray, publisher: rospy.Publisher, encoding: str) -> None:
        image_msg = self._cv_bridge.cv2_to_imgmsg(image, encoding=encoding)
        publisher.publish(image_msg)

    def _publish_path(self, points, publisher):
        if not points.ndim == 2 or not points.shape[0] == 2:
            raise ValueError(f"Expected points to have shape (2, N), got {points.shape}")

        header = Header(frame_id="stargazer", stamp=rospy.Time.now())

        msg = NavPath()
        msg.header = header

        for point in points.T:
            pose = PoseStamped()
            pose.header = header
            pose.pose.position = Point(x=point[0], y=point[1], z=0)
            msg.poses.append(pose)

        publisher.publish(msg)

    def _publish_features(self, features: np.ndarray, publisher: rospy.Publisher, frame_id: str = "stargazer", timestamp: rospy.Time = None):
        msg = create_point_cloud_msg(features[:2], features[2:])
        msg.header.frame_id = frame_id
        msg.header.stamp = timestamp if timestamp is not None else rospy.Time.now()

        publisher.publish(msg)

    @convert_images_to_arrays("bgr8", "16UC1")
    def _image_callback(self, color_image: np.ndarray, depth_image: np.ndarray, timestamp: rospy.Time) -> None:
        now = rospy.Time.now()
        t0 = time.time()
        # 1. Preprocessing: Convert color image into a binary mask
        preprocessed_image = self._image_preprocessor.process(color_image, depth_image)
        rospy.logdebug(f"Preprocessing took {self._image_preprocessor.get_last_duration():.3f}s")

        # 2. Transform the binary mask into a bird's eye view image
        preprocessed_image[preprocessed_image < 200] = 0
        depth_image[preprocessed_image < 200] = 0
        bev_image = self._bev_transformer.transform(color_image, depth_image)

        # 3. Extract features from the bird's eye view image
        features = self._feature_extractor.extract(bev_image)
        try:
            tf_vehicle_to_world = self._transform_provider.named_transform("vehicle_to_world", timestamp)
        except ExtrapolationException:
            tf_vehicle_to_world = self._transform_provider.named_transform("vehicle_to_world")
        transformed_features = transform_features(features, self._bev_transformer.intrinsic_matrix, tf_vehicle_to_world)

        local_features = transform_features(features, self._bev_transformer.intrinsic_matrix, np.eye(4))
        self._publish_features(local_features, self._local_features_publisher, frame_id="base_link", timestamp=now)
        t1 = time.time()

        # 4. Track lanes
        # TODO: Implement lane tracking
        center_points = self._lane_map.update(transformed_features, vehicle_pose=tf_vehicle_to_world)
        t2 = time.time()

        self._publish_path(center_points, self._trajectory_publisher)
        try:
            self._publish_path(np.vstack(self._lane_map._center_line).T, self._center_line_publisher)
        except ValueError as e:
            rospy.logerr(e)

        recent_observations = self._lane_map.get_recent_observations(2)
        features = np.concatenate([o.features for o in recent_observations], axis=1)
        self._publish_features(features=features, publisher=self._debug_publisher_features)

        # 5. Publish debug images
        if self.params.publish_debug_images:
            # pass
            self._publish_image(preprocessed_image, self._debug_publisher_preprocessed, "mono8")
            self._publish_image(bev_image, self._debug_publisher_bev, "mono8")

            # timestamp = rospy.Time.now()
            # cv.imwrite(str(self.package_path/f"notebooks/bev_images/bev_image_{timestamp}.png"), bev_image)

            # tf_world_to_camera = self._transform_provider.named_transform("world_to_camera")
            # overlay = LaneFeatureOverlay(self._camera_info.intrinsic_matrix, tf_world_to_camera)
            # color_image = overlay.draw(color_image, transformed_features)

            # map_overlay = LaneMapOverlay(self._camera_info.intrinsic_matrix, tf_world_to_camera)
            # grid = map_overlay.draw(color_image, self._lane_map)
            # self._publish_image(grid, self._debug_publisher_bev, "mono8")

            # proj_overlay = LaneSegmentOverlay(self._camera_info.intrinsic_matrix, tf_world_to_camera)
            # color_image = proj_overlay.draw(color_image, self._lane_map._lane_segments)
            # self._publish_image(color_image, self._debug_publisher_features, "bgr8")

        t3 = time.time()
        # print(f"Preprocessing: {t1-t0:.03f}, Tracking: {t2-t1:.03f}, Publishing: {t3-t2:.03f}, total: {t3-t0:.03f}")
