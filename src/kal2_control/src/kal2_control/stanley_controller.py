from typing import Tuple

import numpy as np

import rospy
class StanleyController:
    def __init__(self, target_speed: float, k: float, look_ahead_index: int, adaptive_speed: bool) -> None:
        self._target_speed = 1.0
        self._k = 0.3
        self._k_soft = 0.0
        self._adaptive_speed = adaptive_speed
        self._look_ahead_index = int(look_ahead_index)

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

        try:
            next_index = (current_index + self._look_ahead_index) % len(distances)
        except IndexError as e:
            print(f"{current_index}, {len(distances)}")
            return 0.0, 0.0

        trajectory_heading = local_points[:, next_index] - local_points[:, current_index]
        trajectory_heading /= np.linalg.norm(trajectory_heading)
        x_axis = np.array([1.0, 0.0])
        y_axis = np.array([0.0, 1.0])
        heading_error = np.arccos(x_axis.dot(trajectory_heading)) * np.sign(np.cross(x_axis, trajectory_heading))
        
        crosstrack_error = y_axis.dot(local_points[:, current_index])
        speed = target_speed[current_index]

        steering_angle = - np.arctan2(self._k * crosstrack_error, speed + self._k_soft) - heading_error 
        steering_angle = np.clip(steering_angle, np.radians(-25), np.radians(25))

        if self._adaptive_speed:
            return steering_angle, np.clip(speed, 0.0, 1.9)
        else:
            return steering_angle, self._target_speed

