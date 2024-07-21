from enum import Enum, auto
from typing import Optional

import numpy as np

from statemachine import StateMachine, State

import rospy


class TargetCity(Enum):
    Hildesheim = "Hildesheim"
    Karlsruhe = "Karlsruhe"
    Köln = "Köln"
    München = "München"

class TurningDirection(Enum):
    Unknown = "unknown"
    Left = "left"
    Right = "right"


class Zone(Enum):
    RecordedZone = "RecordedZone"
    LaneDetectionZone = "LaneDetectionZone"
    SignDetectionZone = "SignDetectionZone"
    InitialPosition = "InitialPosition"


class StatesLogger:
    def __init__(self) -> None:
        pass

    def after_transition(self, event, source, target):
        rospy.loginfo(f"Transition from '{source}' to '{target}' due to event '{event}'")


class VehicleState:
    def __init__(self) -> None:
        self._target_city: Optional[TargetCity] = None
        self._current_zone: Optional[Zone] = None
        self._initial_pose: Optional[np.ndarray] = None
        self._is_driving_cw: Optional[bool] = None
        self._is_initialized = False
        self._turning_direction: TurningDirection = TurningDirection.Unknown

    @property
    def current_zone(self):
        return self._current_zone

    @property
    def target_city(self):
        return self._target_city

    @property
    def is_driving_cw(self):
        return self._is_driving_cw
    
    @property
    def is_initialized(self):
        return self._is_initialized
    
    @property
    def turning_direction(self):
        return self._turning_direction

    def on_start_driving(self, target_city: TargetCity):
        if not isinstance(target_city, TargetCity):
            target_city = TargetCity(target_city)

        rospy.loginfo(f"Driving to {target_city.value}.")
        self._target_city = target_city

    def on_initial_pose_detected(self, initial_pose: np.ndarray):
        if initial_pose.shape != (3,):
            raise ValueError(f"Expected pose to have shape (x, y, theta) -> (3,), got {initial_pose.shape}.")

        self._initial_pose = initial_pose
        self._is_driving_cw = np.abs(initial_pose[2]) > np.pi / 2

        rospy.loginfo(f"Driving {'CW' if self._is_driving_cw else 'CCW'}.")

    def on_sign_detected(self, target_city: TargetCity, turning_direction: TurningDirection):
        rospy.loginfo(f"Detected {target_city.value} on the {turning_direction.value} side.")
        self._turning_direction = turning_direction

    def on_exit_initializing(self):
        self._is_initialized = True

    def on_enter_in_recorded_zone(self):
        rospy.loginfo("Entered recorded zone.")
        self._current_zone = Zone.RecordedZone

    def on_enter_in_lane_detection_zone(self):
        rospy.loginfo("Entered lane detection zone.")
        self._current_zone = Zone.LaneDetectionZone

    def on_enter_in_sign_detection_zone(self):
        rospy.loginfo("Entered sign detection zone.")
        self._current_zone = Zone.SignDetectionZone

    def is_recorded_zone(self, zone: Zone):
        return zone == Zone.RecordedZone

    def is_lane_detection_zone(self, zone: Zone):
        return zone == Zone.LaneDetectionZone

    def is_sign_detection_zone(self, zone: Zone):
        return zone == Zone.SignDetectionZone

    def is_target_city(self, city):
        if self._target_city != city:
            rospy.loginfo(f"Wrong city detected: {city}, expected: {self._target_city.value}")
        return self._target_city == city  # type: ignore


class VehicleStateMachine(StateMachine):
    uninitialized = State("Uninitialized", initial=True)
    initializing = State("Initializing")
    in_recorded_zone = State("In Recorded Zone")
    in_lane_detection_zone = State("In Lane Detection Zone")
    in_sign_detection_zone = State("In Sign Detection Zone")
    turning = State("Turning")
    off_track = State()
    goal_reached = State("Goal Reached", final=True)

    start_driving = uninitialized.to(initializing)  # , on="on_start_driving")
    initial_pose_detected = initializing.to(in_recorded_zone)
    zone_detected = (
        # initializing.to(in_recorded_zone, cond="is_recorded_zone")
        in_recorded_zone.to(in_lane_detection_zone, cond="is_lane_detection_zone")
        | in_lane_detection_zone.to(in_recorded_zone, cond="is_recorded_zone")
        | in_sign_detection_zone.to(in_recorded_zone, cond="is_recorded_zone")
        | in_recorded_zone.to(in_sign_detection_zone, cond="is_sign_detection_zone")
        | off_track.to(in_recorded_zone, cond="is_recorded_zone")
    )
    sign_detected = in_sign_detection_zone.to(turning, cond="is_target_city")
    off_track_detected = in_lane_detection_zone.to(off_track)
    finished = turning.to(goal_reached)
