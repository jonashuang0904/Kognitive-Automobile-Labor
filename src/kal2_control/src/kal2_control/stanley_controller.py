import numpy as np

import rospy
class StanleyController:
    def __init__(self) -> None:
        self._look_ahead_distance = 0.7
        self._wheel_base = 0.3
        self._target_speed = 1.0
        self._last_speed = 1.0
        self._last_index = 0
        self._k = 0.3
        self._k_soft = 0.0

    def update(self, rotation_matrix: np.ndarray, translation: np.ndarray, path_points: np.ndarray) -> float:

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
            next_index = (current_index + 2) % len(distances)
        except IndexError as e:
            print(f"{current_index}, {len(distances)}")
        trajectory_heading = local_points[:, next_index] - local_points[:, current_index]
        trajectory_heading /= np.linalg.norm(trajectory_heading)
        x_axis = np.array([1.0, 0.0])
        y_axis = np.array([0.0, 1.0])
        heading_error = np.arccos(x_axis.dot(trajectory_heading)) * np.sign(np.cross(x_axis, trajectory_heading))

        # heading_error = np.arctan2(trajectory_heading[1], trajectory_heading[0])
        
        crosstrack_error = y_axis.dot(local_points[:, current_index])
        speed = target_speed[current_index]

        steering_angle = - np.arctan2(self._k * crosstrack_error, speed + self._k_soft) - heading_error 
        steering_angle = np.clip(steering_angle, np.radians(-15), np.radians(15))
        #rospy.loginfo(f"index: {current_index}, cte: {crosstrack_error:.02f}, heading: {heading_error:.02f}, steering: {np.degrees(steering_angle):.02f}")

        #rospy.loginfo(f"{look_ahead_distance:.02f}, {np.degrees(delta):.02f}"
    
        return steering_angle, 1.0#np.clip(speed, 0.0, 1.9)

