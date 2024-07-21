from dataclasses import dataclass
from pathlib import Path
from functools import cached_property

import rospy
import numpy as np
import yaml

from nav_msgs.msg import Path as NavMsgPath
from tf2_ros import TransformListener, Buffer, LookupException, ExtrapolationException
from tf.transformations import euler_from_quaternion
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped, PoseStamped, Point, Quaternion

from kal2_control.pure_pursuit import PurePursuitController
from kal2_control.stanley_controller import StanleyController
from kal2_control.state_machine import Zone, TurningDirection
from kal2_control.trajectory_planning import (
    create_trajectory_from_path,
    create_trajectory_from_path_with_control_points,
)
from kal2_control.turning import calculate_turningpath
from kal2_util.node_base import NodeBase
from kal2_msgs.msg import MainControllerState  # type: ignore

# roslaunch command for own recorded path: roslaunch kal demo_path_follower.launch path_file:=/home/kal2/path_wholeloop_ic_smooved.yaml


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float

    @property
    def position(self):
        return np.array([self.x, self.y])

    @property
    def rotation_matrix(self):
        cos_theta = np.cos(self.yaw)
        sin_theta = np.sin(self.yaw)
        return np.array([[cos_theta, sin_theta], [sin_theta, -cos_theta]])

    @staticmethod
    def from_pose_msg(msg: TransformStamped):
        x = msg.transform.translation.x
        y = msg.transform.translation.y
        q = msg.transform.rotation
        (_, _, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])

        return Pose2D(x=x, y=y, yaw=yaw)


@dataclass
class CarState:
    steering_angle: float
    speed: float

    def as_ackermann_msg(self):
        msg = AckermannDrive()
        msg.steering_angle = self.steering_angle
        msg.speed = self.speed

        return msg


def _extract_xyz_from_pose(pose: PoseStamped):
    return pose.pose.position.x, pose.pose.position.y, pose.pose.position.z


def _load_recorded_path(file_path: str) -> NavMsgPath:
    with Path(file_path).open() as file:
        path_dict = yaml.safe_load(file)

    path_msg = NavMsgPath()
    path_msg.header.seq = path_dict["header"]["seq"]
    path_msg.header.stamp.secs = path_dict["header"]["stamp"]["secs"]
    path_msg.header.stamp.nsecs = path_dict["header"]["stamp"]["nsecs"]
    path_msg.header.frame_id = path_dict["header"]["frame_id"]

    for pose_dict in path_dict["poses"]:
        pose_stamped = PoseStamped()
        pose_stamped.header.seq = pose_dict["header"]["seq"]
        pose_stamped.header.stamp.secs = pose_dict["header"]["stamp"]["secs"]
        pose_stamped.header.stamp.nsecs = pose_dict["header"]["stamp"]["nsecs"]
        pose_stamped.header.frame_id = pose_dict["header"]["frame_id"]

        pose_stamped.pose.position = Point(
            pose_dict["pose"]["position"]["x"], pose_dict["pose"]["position"]["y"], pose_dict["pose"]["position"]["z"]
        )

        pose_stamped.pose.orientation = Quaternion(
            pose_dict["pose"]["orientation"]["x"],
            pose_dict["pose"]["orientation"]["y"],
            pose_dict["pose"]["orientation"]["z"],
            pose_dict["pose"]["orientation"]["w"],
        )

        path_msg.poses.append(pose_stamped)

    return path_msg


class ControllerNode(NodeBase):
    def __init__(self):
        super().__init__(name="vehicle_control_node", log_level=rospy.INFO)
        rospy.loginfo("Starting controller node...")

        controller_params = self.params.controller

        self._purePursuit = StanleyController(
            target_speed=self.params.target_speed,
            k=self.params.stanley_k,
            look_ahead_index=self.params.look_ahead_index,
            adaptive_speed=self.params.use_adaptive_speed,
        )

        self._recorded_path = _load_recorded_path(self.params.recorded_path)
        self._recorded_lane = None
        self._path_points = None

        self._use_recorded_path = True
        self._is_initialized = False
        self._is_driving_cw = None
        self._is_turning = False
        self._turning_direction = TurningDirection.Unknown
        self._current_zone = Zone.RecordedZone

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer)

        self._recorded_path_publisher = rospy.Publisher("/kal2/recorded_path", NavMsgPath, queue_size=10, latch=True)
        self._control_output_publisher = rospy.Publisher(
            self.params.control_output_topic, AckermannDriveStamped, queue_size=10
        )
        self._path_subscriber = rospy.Subscriber(self.params.path_topic, NavMsgPath, self._path_callback)
        self._state_subscriber = rospy.Subscriber(
            "/kal2/main_controller_state", MainControllerState, self._state_callback
        )

        self.period = rospy.Duration.from_sec(1 / controller_params["rate"])
        self._control_timer = rospy.Timer(self.period, self._control_callback)

    @cached_property
    def recorded_path(self):
        if not self._is_initialized:
            raise ValueError("Recorded path is only valid once initialized.")

        map_offset = np.array([self.params.map_offset_x, self.params.map_offset_y]).reshape(2, 1)
        offset = -self.params.lane_offset if self._is_driving_cw else self.params.lane_offset

        if not self.params.use_control_points:
            self._recorded_lane = create_trajectory_from_path(
                path=self._recorded_path,
                lane_offset=offset,
                map_offset=map_offset,
                n_samples=self.params.map_samples,
                v_min=self.params.map_vmin,
                v_max=self.params.map_vmax,
            )
        else:
            n_points = (int(self.params.n_points),)
            n_gap = (int(self.params.n_gap),)
            ct1 = (np.array([self.params.ct1_x, self.params.ct1_y]),)
            ct2 = (np.array([self.params.ct2_x, self.params.ct2_y]),)
            ct3 = np.array([self.params.ct3_x, self.params.ct3_y])

            try:
                self._recorded_lane = create_trajectory_from_path_with_control_points(
                    path=self._recorded_path,
                    lane_offset=offset,
                    map_offset=map_offset,
                    n_samples=self.params.map_samples,
                    v_min=self.params.map_vmin,
                    v_max=self.params.map_vmax,
                    n_points=n_points,
                    n_gap=n_gap,
                    ct1=ct1,
                    ct2=ct2,
                    ct3=ct3,
                )
            except (ValueError, IndexError) as e:
                rospy.logerr(e)
                self._recorded_lane = create_trajectory_from_path(
                    path=self._recorded_path,
                    lane_offset=offset,
                    map_offset=map_offset,
                    n_samples=self.params.map_samples,
                    v_min=self.params.map_vmin,
                    v_max=self.params.map_vmax,
                )

        path = np.array([_extract_xyz_from_pose(pose) for pose in self._recorded_lane.poses])
        path = path[::-1] if self.params.invert_path else path

        return path[::-1] if self._is_driving_cw else path[::1]

    def _get_current_pose(self, timeout: rospy.Duration = rospy.Duration(1.0)) -> Pose2D:
        transform = self._tf_buffer.lookup_transform(
            target_frame="stargazer", source_frame=self.params.vehicle_frame, time=rospy.Time(0), timeout=timeout
        )

        return Pose2D.from_pose_msg(transform)

    def _state_callback(self, msg: MainControllerState):
        try:
            zone = Zone(msg.current_zone)
        except ValueError as e:
            self._is_initialized = False
            return

        self._use_recorded_path = zone in [Zone.RecordedZone, Zone.SignDetectionZone]
        self._current_zone = zone
        self._is_driving_cw = msg.is_driving_cv
        self._is_initialized = msg.is_initialized
        self._turning_direction = TurningDirection(msg.turning_direction)

    def _path_callback(self, msg: NavMsgPath):
        self._path_points = [_extract_xyz_from_pose(pose) for pose in msg.poses]

    def _publish_control_output(self, steering_angle: float, speed: float) -> None:
        if np.isnan(steering_angle) or np.isnan(speed):
            rospy.logerr(f"Speed or steering angle is nan: steering_angle={steering_angle}, speed={speed}")
            return

        msg = AckermannDriveStamped()
        msg.header = Header(frame_id="vehicle_rear_axle", stamp=rospy.Time.now())
        msg.drive = CarState(steering_angle, speed).as_ackermann_msg()
        self._control_output_publisher.publish(msg)

    def _control_callback(self, _):
        rospy.loginfo_once("Controller started.")
        if self._recorded_lane is not None:
            self._recorded_path_publisher.publish(self._recorded_lane)

        if not self._is_initialized:
            rospy.loginfo_throttle(2, "Not initialized.")
            return

        if not self._use_recorded_path and (self._path_points is None or len(self._path_points) < 2):
            rospy.logwarn_throttle(period=5, msg=f"Missing path points: {self._path_points}.")
            return

        try:
            current_pose = self._get_current_pose(timeout=rospy.Duration(0.1))
        except (LookupException, ExtrapolationException) as e:
            rospy.logwarn_throttle(5, e)
            return

        if self._turning_direction != TurningDirection.Unknown:
            path = calculate_turningpath(self._turning_direction, self._is_driving_cw)
            steering_angle, speed = self._purePursuit.update(
                current_pose.rotation_matrix, current_pose.position, path.T
            )
            self._publish_control_output(steering_angle=steering_angle, speed=self.params.turning_speed)
            return

        if not self._use_recorded_path:
            rospy.loginfo_throttle(1, "Using perception.")

        path = self.recorded_path[:, :3] if self._use_recorded_path else np.array(self._path_points)[:2]
        steering_angle, speed = self._purePursuit.update(current_pose.rotation_matrix, current_pose.position, path.T)

        if self._current_zone == Zone.SignDetectionZone:
            speed = self.params.sign_detection_speed

        self._publish_control_output(steering_angle, speed)
