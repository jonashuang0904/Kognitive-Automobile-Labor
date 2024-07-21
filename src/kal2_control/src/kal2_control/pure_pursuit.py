from typing import Tuple
import numpy as np

import rospy


class PurePursuitController:
    def __init__(self, look_ahead_distance: float = 0.7, target_speed: float = 1.0, wheel_base: float = 0.3) -> None:
        self._look_ahead_distance = look_ahead_distance
        self._wheel_base = wheel_base
        self._target_speed = target_speed

    def update(self, rotation_matrix: np.ndarray, translation: np.ndarray, path_points: np.ndarray) -> Tuple[float, float]:

        if path_points.shape[0] == 3:
            target_speed = path_points[-1]
            path_points = path_points[:2]
        else:
            target_speed = np.ones_like(path_points[0]) * self._target_speed

        translation = translation.reshape(2, 1)
        local_points = np.linalg.inv(rotation_matrix) @ (path_points - translation)        
        
        distances = np.linalg.norm(local_points, axis=0)
        current_index = np.argmin(distances)
        speed = target_speed[current_index]

        distances[local_points[0] < 0] = np.inf
    
        if local_points.shape[1] == 0:
            raise ValueError("No pose infront of vehicle")

        look_ahead_distance = 0.5 + (2.0 - 0.5) * (speed - 1.0) #min(speed * 0.8, 1.0)
        distances[distances < look_ahead_distance] = np.inf

        look_ahead_index = np.argmin(distances)
        if (look_ahead_index - current_index) % len(distances) > 10:
            rospy.loginfo(f"Index difference: {look_ahead_index - current_index}, {current_index}, {look_ahead_index}")
            look_ahead_index = (current_index + 10) % len(distances)

        look_ahead_point = local_points[:, look_ahead_index]
        alpha = np.arctan2(look_ahead_point[1], look_ahead_point[0])
        
        delta = - np.arctan2(2.0 * self._wheel_base * np.sin(alpha), look_ahead_distance)
        delta = np.clip(delta, np.radians(-20), np.radians(20))
    
        return delta, np.clip(speed, 0.0, 1.9)

