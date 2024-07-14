from dataclasses import dataclass

import rospy
import numpy as np

from nav_msgs.msg import Path
from tf2_ros import TransformListener, Buffer, LookupException, ExtrapolationException
from tf.transformations import euler_from_quaternion
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped

from kal2_control.pp_controller_node import PurePursuitController

from kal2_util.node_base import NodeBase

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


class ControllerNode(NodeBase):
    def __init__(self):
        super().__init__(name="vehicle_control_node", log_level=rospy.DEBUG)
        rospy.loginfo("Starting controller node...")

        controller_params = self.params.controller
       
        self._purePursuit = PurePursuitController()

        self._path_points = None
        self._car_state = CarState(steering_angle=0, speed=controller_params['speed'])

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer)

        self._control_output_publisher = rospy.Publisher(
            self.params.control_output_topic, AckermannDriveStamped, queue_size=10
        )
        self._path_subscriber = rospy.Subscriber(self.params.path_topic, Path, self._path_callback)
        
        self.period = rospy.Duration.from_sec(1 / controller_params['rate'])
        self._control_timer = rospy.Timer(self.period, self._control_callback)

    def _get_current_pose(self, timeout: rospy.Duration = rospy.Duration(1.0)) -> Pose2D:
        transform = self._tf_buffer.lookup_transform(
            target_frame="stargazer", source_frame="vehicle_rear_axle", time=rospy.Time(0), timeout=timeout
        )

        return Pose2D.from_pose_msg(transform)

    def _path_callback(self, msg: Path):
        def extract_xy(pose):
            return pose.pose.position.x, pose.pose.position.y

        self._path_points = [extract_xy(pose) for pose in msg.poses]
        rospy.loginfo(f"Received path points: {self._path_points}")

    def _publish_control_output(self) -> None:
        rospy.logdebug(f"Publishing control outputs: speed={self._car_state.speed:.02f}, steering_angle={self._car_state.steering_angle:.02f}.")

        msg = AckermannDriveStamped()
        msg.header = Header(frame_id="vehicle_rear_axle", stamp=rospy.Time.now())
        msg.drive = self._car_state.as_ackermann_msg()
        self._control_output_publisher.publish(msg)

    def _control_callback(self, _):
        rospy.loginfo_once("Controller started.")

        if self._path_points is None or len(self._path_points) < 2:
            rospy.logwarn_throttle(period=5, msg=f"Missing path points: {self._path_points}.")
            return

        try:
            current_pose = self._get_current_pose(timeout=rospy.Duration(0.1))
        except (LookupException, ExtrapolationException) as e:
            rospy.logwarn_throttle(5, e)
            return
        
        self._car_state.steering_angle = self._purePursuit.update(current_pose.rotation_matrix, current_pose.position, np.array(self._path_points).T)
        self._publish_control_output()
