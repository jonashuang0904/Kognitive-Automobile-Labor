from typing import Tuple

import numpy as np

class CameraInfo:
    def __init__(self, height: int, width: int, intrinsic_matrix: np.ndarray, distortion_coeffs: np.ndarray) -> None:
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid image size: {height, width}")
        
        if intrinsic_matrix.shape != (3, 3):
            raise ValueError(f"Invalid intrinsic matrix shape: {intrinsic_matrix.shape}")

        self._width = width
        self._height = height
        self._intrinsic_matrix = intrinsic_matrix
        self._distortion_coeffs = distortion_coeffs

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        return self._intrinsic_matrix
    
    @property
    def image_size(self) -> Tuple[int, int]:
        return self._height, self._width

    @staticmethod
    def from_ros_topic(topic: str) -> "CameraInfo":
        import sensor_msgs
        import rospy        

        try:
            rospy.loginfo(f"Waiting for camera info message on topic '{topic}'...")
            msg = rospy.wait_for_message(topic, sensor_msgs.msg.CameraInfo, timeout=5)
            rospy.loginfo(f"Received camera info message.")
        except rospy.exceptions.ROSException as e:
            rospy.logerr(f"Failed to get camera info: {e}. Using default camera info.")
            return CameraInfo.from_default()

        return CameraInfo(msg.height, msg.width, np.array(msg.K).reshape(3, 3), np.array(msg.D))
    

    @staticmethod
    def from_default():
        intrinsic_matrix = np.array([[913.75, 0.0, 655.42], [0.0, 911.92, 350.49], [0.0, 0.0, 1.0]])
        return CameraInfo(720, 1280, intrinsic_matrix, np.zeros(5))
