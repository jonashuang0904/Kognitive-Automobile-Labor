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
from kal2_control.state_machine import Zone
from kal2_control.trajectory_planning import create_trajectory_from_path
from kal2_util.node_base import NodeBase
from kal2_msgs.msg import MainControllerState # type: ignore

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
        return np.array([[cos_theta, sin_theta],
                         [sin_theta, -cos_theta]])

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
    path_msg.header.seq = path_dict['header']['seq']
    path_msg.header.stamp.secs = path_dict['header']['stamp']['secs']
    path_msg.header.stamp.nsecs = path_dict['header']['stamp']['nsecs']
    path_msg.header.frame_id = path_dict['header']['frame_id']

    for pose_dict in path_dict['poses']:
        pose_stamped = PoseStamped()
        pose_stamped.header.seq = pose_dict['header']['seq']
        pose_stamped.header.stamp.secs = pose_dict['header']['stamp']['secs']
        pose_stamped.header.stamp.nsecs = pose_dict['header']['stamp']['nsecs']
        pose_stamped.header.frame_id = pose_dict['header']['frame_id']
        
        pose_stamped.pose.position = Point(
            pose_dict['pose']['position']['x'],
            pose_dict['pose']['position']['y'],
            pose_dict['pose']['position']['z']
        )
        
        pose_stamped.pose.orientation = Quaternion(
            pose_dict['pose']['orientation']['x'],
            pose_dict['pose']['orientation']['y'],
            pose_dict['pose']['orientation']['z'],
            pose_dict['pose']['orientation']['w']
        )
        
        path_msg.poses.append(pose_stamped)

    return path_msg


class ControllerNode(NodeBase):
    def __init__(self):
        super().__init__(name="vehicle_control_node", log_level=rospy.INFO)
        rospy.loginfo("Starting controller node...")

        controller_params = self.params.controller
       
        self._purePursuit = StanleyController()

        self._recorded_path = _load_recorded_path(self.params.recorded_path)
        self._recorded_lane = None
        self._path_points = None
        self._use_recorded_path = True
        self._is_initialized = False
        self._is_driving_cw = None
        self._car_state = CarState(steering_angle=0, speed=controller_params['speed'])

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer)

        self._recorded_path_publisher = rospy.Publisher("/kal2/recorded_path", NavMsgPath, queue_size=10, latch=True)
        self._control_output_publisher = rospy.Publisher(
            self.params.control_output_topic, AckermannDriveStamped, queue_size=10
        )
        self._path_subscriber = rospy.Subscriber(self.params.path_topic, NavMsgPath, self._path_callback)
        self._state_subscriber = rospy.Subscriber("/kal2/main_controller_state", MainControllerState, self._state_callback)
        
        self.period = rospy.Duration.from_sec(1 / controller_params['rate'])
        self._control_timer = rospy.Timer(self.period, self._control_callback)

    @cached_property
    def recorded_path(self):
        if not self._is_initialized:
            raise ValueError("Recorded path is only valid once initialized.")
        
        offset = -0.2 if self._is_driving_cw else 0.2
        self._recorded_lane = create_trajectory_from_path(self._recorded_path, offset)
        
        path = np.array([_extract_xyz_from_pose(pose) for pose in self._recorded_lane.poses])
        return path[::-1] if self._is_driving_cw else path[::1]
            

    def _get_current_pose(self, timeout: rospy.Duration = rospy.Duration(1.0)) -> Pose2D:
        transform = self._tf_buffer.lookup_transform(
            target_frame="stargazer", source_frame="base_link", time=rospy.Time(0), timeout=timeout
        )

        return Pose2D.from_pose_msg(transform)
    
    def _state_callback(self, msg: MainControllerState):
        try:
            zone = Zone(msg.current_zone)
        except ValueError as e:
            self._is_initialized = False
            return

        self._use_recorded_path = zone in [Zone.RecordedZone, Zone.SignDetectionZone]
        self._is_driving_cw = msg.is_driving_cv
        self._is_initialized = msg.is_initialized

    def _path_callback(self, msg: NavMsgPath):
        self._path_points = [_extract_xyz_from_pose(pose) for pose in msg.poses]
        
    def _publish_control_output(self, speed) -> None:
        msg = AckermannDriveStamped()
        msg.header = Header(frame_id="vehicle_rear_axle", stamp=rospy.Time.now())
        self._car_state.speed = speed
        msg.drive = self._car_state.as_ackermann_msg()
        self._control_output_publisher.publish(msg)

    def _control_callback(self, _):
        rospy.loginfo_once("Controller started.")
        if self._recorded_lane is not None:
            self._recorded_path_publisher.publish(self._recorded_lane)

        if not self._is_initialized:
            return

        if not self._use_recorded_path and (self._path_points is None or len(self._path_points) < 2):
            rospy.logwarn_throttle(period=5, msg=f"Missing path points: {self._path_points}.")
            return

        try:
            current_pose = self._get_current_pose(timeout=rospy.Duration(0.1))
        except (LookupException, ExtrapolationException) as e:
            rospy.logwarn_throttle(5, e)
            return

        self._use_recorded_path = True

        if not self._use_recorded_path:
            rospy.loginfo_throttle(1, "Using perception.")
        
        path = self.recorded_path[:, :3] if self._use_recorded_path else np.array(self._path_points)[:2]
        self._car_state.steering_angle, speed = self._purePursuit.update(current_pose.rotation_matrix, current_pose.position, path.T)
        self._publish_control_output(speed)
