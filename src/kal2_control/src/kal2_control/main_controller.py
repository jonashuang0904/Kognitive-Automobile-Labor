from typing import Any, List, Optional
from enum import Enum, auto
from collections import deque

import numpy as np

import rospy
from tf2_ros import TransformBroadcaster, Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException  # type: ignore
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, Point, Quaternion

from ros_numpy import numpify  # type: ignore
from kal2_util.node_base import NodeBase
from kal2_control.state_machine import VehicleStateMachine, VehicleState, Zone, TargetCity, StatesLogger, TurningDirection
from kal2_msgs.msg import MainControllerState, DetectedSign

try:
    from kal2_srvs.srv import StartDriving
except ImportError as e:
    print(e)


class TrackGate:
    def __init__(self, name: str, zone: Zone, gate_position: np.ndarray, radius: float, epsilon: float = 0.3) -> None:
        self._name = name
        self._zone = zone
        self._gate_position = gate_position
        self._radius = radius
        self._epsilon = epsilon
        self._was_inside = False

    @property
    def inside_gate(self):
        return self._was_inside

    @property
    def name(self):
        return self._name

    @property
    def zone(self):
        return self._zone

    def is_inside_gate(self, current_pose: np.ndarray):
        distance = np.linalg.norm(self._gate_position - current_pose)

        in_inner_circle = distance < (self._radius + self._epsilon)
        in_outer_circle = distance < self._radius

        if in_inner_circle:
            self._was_inside = True
        elif not in_outer_circle:
            self._was_inside = False

        return self._was_inside


class InitialPoseEstimator:
    def __init__(self, n_poses: int) -> None:
        self._poses = deque(maxlen=n_poses)

    def append(self, pose):
        self._poses.append(pose)

    def get_estimate(self):
        if not self.has_estimate():
            raise ValueError("No estimate available.")

        poses = np.array(self._poses)

        return np.median(poses, axis=0)

    def has_estimate(self):
        return len(self._poses) == self._poses.maxlen

    def reset(self):
        self._poses.clear()


class MainControllerNode(NodeBase):
    def __init__(self) -> None:
        super().__init__(name="sign_detector_node")
        rospy.loginfo("Starting main controller node...")

        self._sm = VehicleStateMachine(VehicleState(), listeners=[StatesLogger()], allow_event_without_transition=True)

        self._initial_pose_estimator = InitialPoseEstimator(n_poses=5)
        radius = self.params.gate_radius

        if self.params.use_lane_detection:
            self._track_gates = [
                TrackGate("LD1", Zone.LaneDetectionZone, np.array([4.0, 0.7]), radius=radius),
                TrackGate("LD2", Zone.LaneDetectionZone, np.array([0.8, 2.0]), radius=radius),
                TrackGate("SD1", Zone.SignDetectionZone, np.array([2.0, 4.0]), radius=radius),
                TrackGate("SD2", Zone.SignDetectionZone, np.array([6.0, 2.2]), radius=radius),
            ]
        else:
            self._track_gates = [
                TrackGate("SD1", Zone.SignDetectionZone, np.array([2.0, 4.0]), radius=radius),
                TrackGate("SD2", Zone.SignDetectionZone, np.array([6.0, 2.2]), radius=radius),
            ]

        self._loop_closure_gate = None
        self._lap_counter = 0

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer)
        self._state_publisher = rospy.Publisher("/kal2/main_controller_state", MainControllerState, queue_size=10)
        self._marker_array_publisher = rospy.Publisher("/kal2/gates", MarkerArray, queue_size=10, latch=True)

        self._start_service = rospy.Service("start_driving", StartDriving, self._start_driving_handler)
        self._sign_detection_subscriber = rospy.Subscriber("/kal2/detected_sign", DetectedSign, self._sign_detection_callback)
        self._timer = rospy.Timer(rospy.Duration.from_sec(0.1), callback=self._loop_callback)

        rospy.loginfo("Main controller node started.")

    @property
    def vehicle_state(self) -> VehicleState:
        return self._sm.model  # type: ignore

    @property
    def current_state(self):
        return self._sm.current_state

    @property
    def current_zone(self) -> Optional[Zone]:
        return self.vehicle_state.current_zone
    
    def _publish_current_state(self):
        msg = MainControllerState()

        target_city = self.vehicle_state.target_city
        current_zone = self.current_zone
        msg.current_zone = current_zone.name if current_zone is not None else "None"
        msg.target_city = target_city.name if target_city is not None else "None"
        msg.is_driving_cv = bool(self.vehicle_state.is_driving_cw)
        msg.is_initialized = self.vehicle_state.is_initialized
        msg.lap_counter = self._lap_counter
        msg.turning_direction = self.vehicle_state.turning_direction.value

        self._state_publisher.publish(msg)

    def _publish_gate_markers(self):
        from std_msgs.msg import ColorRGBA, Header
        from geometry_msgs.msg import Vector3, Quaternion, Point

        marker_array = MarkerArray()

        for i, gate in enumerate(self._track_gates):
            x, y = tuple(gate._gate_position)
            radius = gate._radius

            marker = Marker()
            marker.header = Header(frame_id="stargazer", stamp=rospy.Time.now())
            marker.ns = "spheres"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = Point(x=x, y=y, z=0)
            marker.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            marker.scale = Vector3(x=2 * radius, y=2 * radius, z=0.01)
            marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)

            marker_array.markers.append(marker)

        self._marker_array_publisher.publish(marker_array)

    def _start_driving_handler(self, msg):
        try:
            target_city = TargetCity(msg.target_city)
        except ValueError as e:
            rospy.logerr(e)
            return False

        self._sm.start_driving(target_city=target_city)

        return True
    
    def _sign_detection_callback(self, msg: DetectedSign):
        if msg.city == "unknown":
            return

        try:
            detected_city = TargetCity(msg.city)
        except ValueError:
            if msg.city in ["Koln", "Köln"]:
                detected_city = TargetCity("Köln")
            elif msg.city in ["Munchen", "München"]:
                detected_city = TargetCity("Köln")
            else:
                rospy.logerr(f"Invalid city detected: {msg.city}")
                return
            
        try:
            turning_direction = TurningDirection(msg.direction)
        except ValueError as e:
            rospy.logerr(e)
            return
            
        self._sm.sign_detected(detected_city, turning_direction)

    def _detect_current_zone(self, current_pose):
        current_position = current_pose[:2, 3]

        if self.current_zone is None:
            raise ValueError(f"Current zone should not be None in state {self.current_state}.")

        for gate in self._track_gates:
            was_inside_gate = gate.inside_gate
            is_inside_gate = gate.is_inside_gate(current_position)

            if is_inside_gate and not was_inside_gate:
                if self.current_zone == Zone.RecordedZone:
                    rospy.loginfo(f"Passed gate: {gate.name}, entering zone: {gate.zone} from {self.current_zone}.")
                    self._sm.zone_detected(gate.zone)
                else:
                    rospy.loginfo(f"Passed gate: {gate.name}, entering zone: {Zone.RecordedZone} from {gate.zone}.")
                    self._sm.zone_detected(Zone.RecordedZone)

                break

    def _detect_loop_closure(self, current_pose):
        if self._loop_closure_gate is None:
            return
        
        current_position = current_pose[:2, 3]
        was_inside_gate = self._loop_closure_gate.inside_gate
        is_inside_gate = self._loop_closure_gate.is_inside_gate(current_position)

        if is_inside_gate and not was_inside_gate:
            self._lap_counter += 1
            rospy.loginfo(f"Loop closure detected: lap={self._lap_counter}")

    def _loop_callback(self, _):
        self._publish_gate_markers()
        self._publish_current_state()

        try:
            transform = self._tf_buffer.lookup_transform(
                source_frame="vehicle_rear_axle", target_frame="stargazer", time=rospy.Time(0)
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            rospy.logwarn_throttle(2, e)
            return

        current_pose = numpify(transform.transform)

        if self.current_state == VehicleStateMachine.initializing:
            angle = np.arctan2(current_pose[1, 0], current_pose[0, 0])
            pose2d = np.hstack((current_pose[:2, 3], angle))
            self._initial_pose_estimator.append(pose2d)

            if self._initial_pose_estimator.has_estimate():
                estimate = self._initial_pose_estimator.get_estimate()
                rospy.loginfo(f"Initial pose: {estimate}")
                self._sm.initial_pose_detected(initial_pose=estimate)
                self._loop_closure_gate = TrackGate("initial_position", Zone.InitialPosition, estimate[:2], radius=0.5)
                self._loop_closure_gate._was_inside = True

        if self.current_state in [
            VehicleStateMachine.in_sign_detection_zone,
            VehicleStateMachine.in_lane_detection_zone,
            VehicleStateMachine.in_recorded_zone,
        ]:
            self._detect_current_zone(current_pose)
            self._detect_loop_closure(current_pose)
