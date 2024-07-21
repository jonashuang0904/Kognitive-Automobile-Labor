import time

from functools import cached_property
from pathlib import Path
from threading import Lock
from typing import List, Union
from scipy.spatial.distance import mahalanobis

import numpy as np
import sympy as sp

import rospy
import rospkg

from tf2_msgs.msg import TFMessage
from tf2_ros import TransformBroadcaster, Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException  # type: ignore
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path as NavPath
from std_msgs.msg import Header
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


class PoseEstimator:
    def __init__(self, std_vel: float = 0.5, std_yaw_rate: float = 0.1, std_pose: float = 0.1) -> None:
        self._x = np.array([0.0, 0.0, 0.0])
        self._P = np.eye(3) * std_pose**2

        self._Q = np.diag([std_vel**2, std_yaw_rate**2])  # system noise
        self._R = np.eye(3) * std_pose**2  # measurement noise
        self._H = np.eye(3)  # observation model

        x, y, theta, v, omega, dt = sp.symbols("x, y, theta, v, omega, dt")
        state = sp.Matrix([x, y, theta])
        odometry = sp.Matrix([v, omega])
        state_transition = sp.Matrix(
            [
                x + v * sp.cos(theta + 0.5 * omega * dt) * dt,
                y + v * sp.sin(theta + 0.5 * omega * dt) * dt,
                theta + omega * dt,
            ]
        )
        state_jacobian = state_transition.jacobian(state)
        model_jacobian = state_transition.jacobian(odometry)

        self._state_transition = sp.lambdify([state, odometry, dt], state_transition)
        self._state_jacobian = sp.lambdify([state, odometry, dt], state_jacobian)
        self._model_jacobian = sp.lambdify([state, odometry, dt], model_jacobian)

    @property
    def state(self):
        return self._x

    @property
    def covariance(self):
        return self._P

    @property
    def transform(self):
        T = np.eye(4)
        T[:2, 3] = self.state[:2]

        cos_theta = np.cos(self.state[-1])
        sin_theta = np.cos(self.state[-1])
        T[:2, :2] = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        return T

    def reset(self, initial_pose: np.ndarray, inital_covariance=None):
        self._x = initial_pose
        self._P = inital_covariance if inital_covariance is not None else self._R

    def predict(self, speed: float, yaw_rate: float, dt: float):
        x = self._x
        u = np.array([speed, yaw_rate])

        x_next = self._state_transition(x, u, dt).squeeze()
        x_next[-1] = _wrap_angle(x_next[-1])
        F = self._state_jacobian(x, u, dt)
        V = self._model_jacobian(x, u, dt)
        P = self._P

        self._x = x_next
        self._P = F @ P @ F.T + V @ self._Q @ V.T

    def update(self, pose: np.ndarray):
        z = pose.squeeze()
        H = self._H
        P = self._P
        R = self._R

        y = z - H @ self._x
        y[-1] = _wrap_angle(y[-1])
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)

        self._x += K @ y
        self._x[-1] = _wrap_angle(self._x[-1])
        self._P = (np.eye(3) - K @ H) @ P

class PathPublisher:
    def __init__(self, topic: str, frame_id: str, queue_size: int = 10) -> None:
        self._publisher = rospy.Publisher(topic, NavPath, queue_size=queue_size)
        self._path: List[PoseStamped] = []
        self._frame_id = frame_id

    @property
    def header(self):
        return Header(frame_id=self._frame_id, stamp=rospy.Time.now())
    
    @property
    def poses(self):
        return self._path

    def append_pose(self, pose: Union[Pose, PoseStamped]):
        if pose._type == "geometry_msgs/Pose":
            self._path.append(PoseStamped(header=self.header, pose=pose))
        elif pose._type == "geometry_msgs/PoseStamped":
            self._path.append(pose) # type: ignore

    def publish(self):
        path = NavPath(header=self.header, poses=self.poses)
        self._publisher.publish(path)


class LocalizationNode(NodeBase):
    def __init__(self) -> None:
        super().__init__(name="localization_node")
        rospy.loginfo("Starting localization node...")

        self._stargazer_period = 0.1
        self._odometry_period = 0.02

        self._rospack = rospkg.RosPack()
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer)

        self._filter = PoseEstimator(std_vel=0.5, std_yaw_rate=0.5, std_pose=0.025)
        self._filter_mutex = Lock()
        self._last_odometry_callback = None
        self._last_pose_callback = None
        self._driven_path = NavPath()
        self._driven_path.header = Header(frame_id="stargazer", stamp=rospy.Time.now())
        self._frame_counter = 0
        self._outlier_count = 0

        self._tf_broadcaster = TransformBroadcaster()
        # self._path_publisher = rospy.Publisher("/kal2/driven_path", NavPath, queue_size=10)
        self._driven_path_publisher = PathPublisher("/kal2/driven_path", frame_id="stargazer")
        self._stargazer_path_publisher = PathPublisher("/kal2/stargazer_path", frame_id="stargazer")
        self._pose_publisher = rospy.Publisher("/kal2/estimated_pose", PoseWithCovarianceStamped, queue_size=10)

        self._odom_subscriber = rospy.Subscriber("/anicar/vesc/odom", Odometry, self._odometry_callback)

        rospy.loginfo("Localization node started.")

    @cached_property
    def package_path(self) -> Path:
        return Path(self._rospack.get_path("kal2_perception"))

    def reset(self):
        rospy.loginfo("Resetting localization state..")
        self._last_pose_callback = None

        try:
            transform = self._tf_buffer.lookup_transform(
                source_frame="vehicle_rear_axle", target_frame="stargazer", time=rospy.Time(0)
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            rospy.logwarn(e)
            rospy.logwarn("Failed to reset localization state because of missing transforms.")
            return

        pose = _convert_to_pose2d(transform)
        self._filter.reset(initial_pose=pose, inital_covariance=np.eye(3))
        self._last_pose_callback = transform.header.stamp
        rospy.loginfo("Localization state reset.")

    def _publish_current_estimate(self, stamp):
        state = self._filter.state

        tf = TransformStamped()
        tf.child_frame_id = "base_link"
        tf.header = Header(frame_id="stargazer", stamp=stamp, seq=self._frame_counter)
        tf.transform = _convert_to_transform_msg(state)
        self._tf_broadcaster.sendTransform(tf)

        pose_with_cov = PoseWithCovarianceStamped()
        pose_with_cov.header = Header(frame_id="stargazer", stamp=stamp, seq=self._frame_counter)
        pose_with_cov.pose.pose = _convert_to_pose_msg(state)

        estimated_covariance = self._filter.covariance
        covariance = np.eye(6) * 0.00
        covariance[:2, :2] = estimated_covariance[:2, :2]
        covariance[-1, -1] = estimated_covariance[-1, -1]
        covariance[-1, :2] = estimated_covariance[-1, :2]
        covariance[:2, -1] = estimated_covariance[:2, -1]

        pose_with_cov.pose.covariance = covariance.flatten()
        self._pose_publisher.publish(pose_with_cov)
        self._frame_counter += 1

    def _predict_with_odometry(self, msg: Odometry):
        speed = msg.twist.twist.linear.x
        yaw_rate = msg.twist.twist.angular.z

        self._filter.predict(speed=speed, yaw_rate=yaw_rate, dt=self._odometry_period)

    def _update_with_transform(self):
        if self._last_pose_callback is None:
            self.reset()
            return

        try:
            transform = self._tf_buffer.lookup_transform(
                source_frame="vehicle_rear_axle", target_frame="stargazer", time=rospy.Time(0)
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            rospy.logwarn(e)
            return

        dt = (transform.header.stamp - self._last_pose_callback).to_sec()

        if dt > self._stargazer_period / 2:
            measured_pose = _convert_to_pose2d(transform)

            dist = mahalanobis(measured_pose[:2], self._filter.state[:2], VI=np.eye(2))

            if dist > 0.25 and self._outlier_count <= 10:
                rospy.logwarn(f"Detected outlier with distance {dist}.")
                self._outlier_count += 1
                return
            elif self._outlier_count > self.params.max_outliers:
                rospy.logwarn(f"Found more than {self._outlier_count} outliers. Resetting..")
                self.reset()
            
            self._outlier_count = 0
            self._filter.update(measured_pose)

            self._driven_path_publisher.append_pose( _convert_to_pose_msg(self._filter.state))
            self._driven_path_publisher.publish()

            self._stargazer_path_publisher.append_pose(_convert_to_pose_msg(measured_pose))
            self._stargazer_path_publisher.publish()

    def _odometry_callback(self, msg: Odometry):
        now = msg.header.stamp

        if self._last_odometry_callback is None:
            self._last_odometry_callback = now
            return

        self._predict_with_odometry(msg=msg)
        self._update_with_transform()
        self._publish_current_estimate(msg.header.stamp)

        self._last_odometry_callback = now