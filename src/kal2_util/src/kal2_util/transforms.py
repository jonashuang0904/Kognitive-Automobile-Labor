import numpy as np
import rospy

from tf2_ros import Buffer, TransformListener, LookupException, ExtrapolationException
from enum import Enum, auto
from typing import Union

from functools import lru_cache as cache


class CoordinateFrames(Enum):
    WORLD = auto()
    VEHICLE_FRONT = auto()
    VEHICLE = auto()
    CAMERA = auto()

    @staticmethod
    def default_mapping():
        return {
            CoordinateFrames.WORLD: "stargazer",
            CoordinateFrames.VEHICLE: "vehicle_rear_axle",
            CoordinateFrames.CAMERA: "camera_front_aligned_depth_to_color_frame",
        }


class TransformProvider:
    def __init__(self, mapping: dict = None):
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer)

        self._mapping = mapping or CoordinateFrames.default_mapping()

        rospy.loginfo("Waiting for transforms from tf...")
        self.wait_for_transform(source_frame=CoordinateFrames.WORLD, target_frame=CoordinateFrames.VEHICLE)
        rospy.loginfo("Transforms available.")

    def wait_for_transform(
        self,
        *,
        source_frame: Union[str, CoordinateFrames],
        target_frame: Union[str, CoordinateFrames],
        timeout: rospy.Duration = rospy.Duration(10),
    ):
        try:
            self.lookup_transform(source_frame=source_frame, target_frame=target_frame, timeout=timeout)
        except (LookupException, ExtrapolationException) as e:
            rospy.logwarn(f"Failed to lookup transform after {timeout.to_sec()}s: {e}")
            return False
        return True

    def lookup_transform(
        self,
        *,
        source_frame: Union[str, CoordinateFrames],
        target_frame: Union[str, CoordinateFrames],
        time: rospy.Time = rospy.Time(0),
        timeout: rospy.Duration = rospy.Duration(0.0),
    ) -> np.ndarray:
        if isinstance(source_frame, CoordinateFrames):
            source_frame = self._mapping[source_frame]

        if isinstance(target_frame, CoordinateFrames):
            target_frame = self._mapping[target_frame]

        transform = self._tf_buffer.lookup_transform(
            source_frame=source_frame, target_frame=target_frame, time=time, timeout=timeout
        )

        from ros_numpy import numpify
        return numpify(transform.transform)
    
    def named_transform(self, name: str) -> np.ndarray:
        try:
            source, _, target = name.upper().split('_')
        except ValueError:
            raise ValueError(f"Invalid transform name: {name}. Expected format: <source>_to_<target>.")
        
        return self.lookup_transform(source_frame=CoordinateFrames[source], target_frame=CoordinateFrames[target])

    def get_transform_vehicle_to_world(self) -> np.ndarray:
        return self.lookup_transform(source_frame=CoordinateFrames.VEHICLE, target_frame=CoordinateFrames.WORLD)

    def get_transform_world_to_vehicle(self) -> np.ndarray:
        return self.lookup_transform(source_frame=CoordinateFrames.VEHICLE, target_frame=CoordinateFrames.WORLD)

    @cache
    def get_transform_camera_to_vehicle(self) -> np.ndarray:
        return self.lookup_transform(source_frame=CoordinateFrames.CAMERA, target_frame=CoordinateFrames.VEHICLE)

    @cache
    def get_transform_vehicle_to_camera(self) -> np.ndarray:
        return self.lookup_transform(source_frame=CoordinateFrames.VEHICLE, target_frame=CoordinateFrames.CAMERA)
