import time

from functools import cached_property
from pathlib import Path
from threading import Lock
from typing import List, Union

import numpy as np
import sympy as sp

import rospy
import rospkg

from tf2_msgs.msg import TFMessage
from tf2_ros import TransformBroadcaster, Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException  # type: ignore
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path as NavPath
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import (
    Vector3,
    Point,
    Quaternion,
    Transform,
    TransformStamped,
    PoseStamped,
    Pose,
    PoseWithCovarianceStamped,
)


from ros_numpy import msgify, numpify


from kal2_perception.node_base import NodeBase
from kal2_util.transforms import TransformProvider, CoordinateFrames


def _wrap_angle(alpha):
    return (alpha + np.pi) % (2.0 * np.pi) - np.pi


def _convert_to_transform_msg(pose_2d: np.ndarray):
    x, y, theta = tuple(pose_2d)
    translation = msgify(Vector3, np.array([x, y, 0]))
    rotation = msgify(Quaternion, np.array([0, 0, np.sin(theta * 0.5), np.cos(theta * 0.5)]))
    return Transform(translation=translation, rotation=rotation)


def _convert_to_pose_msg(pose_2d: np.ndarray):
    x, y, theta = tuple(pose_2d)
    translation = msgify(Point, np.array([x, y, 0]))
    rotation = msgify(Quaternion, np.array([0, 0, np.sin(theta * 0.5), np.cos(theta * 0.5)]))
    return Pose(position=translation, orientation=rotation)


def _convert_to_pose2d(transform: TransformStamped):
    tf = numpify(transform.transform)
    xy = tf[:2, 3]
    theta = np.arctan2(tf[1, 0], tf[0, 0])
    return np.hstack((xy, theta))


class MapRecorderNode(NodeBase):
    def __init__(self) -> None:
        super().__init__(name="map_recorder_node")
        rospy.loginfo("Starting map recorder node...")

        self._stargazer_period = 0.1
        self._odometry_period = 0.02

        self._rospack = rospkg.RosPack()
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer)

        self._accumulated_points_stargazer = []
        self._accumulated_points_ekf = []
        self._stargazer_path = []
        self._ekf_path = []

        self._feature_subscriber = rospy.Subscriber("/kal2/local_features", PointCloud2, self._features_callback)

        rospy.loginfo("Map recorder node started.")

    @cached_property
    def package_path(self) -> Path:
        return Path(self._rospack.get_path("kal2_perception"))
    
    def save(self, path: str):
        stargazer = np.concatenate(self._accumulated_points_stargazer, axis=1)
        stargazer_path = np.concatenate(self._stargazer_path, axis=1)
        ekf = np.concatenate(self._accumulated_points_ekf, axis=1)
        ekf_path = np.concatenate(self._ekf_path, axis=1)
        print(ekf_path.shape)
        np.savez(path, stargazer=stargazer, ekf=ekf, stargazer_path=stargazer_path, ekf_path=ekf_path)

    def _features_callback(self, msg: PointCloud2):
        features3d = numpify(msg)

        # print(features3d)
        x, y = features3d['x'], features3d['y']
        normal_x, normal_y = features3d['normal_x'], features3d['normal_y']

        points = np.vstack((x, y))
        normals = np.vstack((normal_x, normal_y))

        try:
            transform = self._tf_buffer.lookup_transform(
                source_frame="vehicle_rear_axle", target_frame="stargazer", time=msg.header.stamp, timeout=rospy.Duration.from_sec(0.5)
            )
            tf = numpify(transform.transform)
            points_stargazer = tf[:2, :2] @ points + tf[:2, 3].reshape(2, 1)
            self._accumulated_points_stargazer.append(points_stargazer)
            self._stargazer_path.append(tf[:2, 3].reshape(2, 1))
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            rospy.logwarn(e)

        try:
            transform = self._tf_buffer.lookup_transform(
                source_frame="base_link", target_frame="stargazer", time=msg.header.stamp,  timeout=rospy.Duration.from_sec(0.5)
            )
            tf = numpify(transform.transform)
            points_ekf = tf[:2, :2] @ points + tf[:2, 3].reshape(2, 1)
            self._accumulated_points_ekf.append(points_ekf)
            self._ekf_path.append(tf[:2, 3].reshape(2, 1))
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            rospy.logwarn(e)
