import numpy as np
from kal2_control.pid import k_smallest_indices

class PurePursuitController:
    def __init__(self) -> None:
        self._wheelbase = 0.3
        self._look_ahead_point = 0.0

    def calculate_steeringangle (self, heading, vehicle_position: np.array, path_points: np.array):
        if vehicle_position.shape != (2,):
            raise ValueError(f"Expected vehicle_position.shape to be (2,), got {vehicle_position.shape}.")
        
        if path_points.ndim != 2 or path_points.shape[0] != 2:
            raise ValueError(f"Expected path.shape to be (2, N), got {path_points.shape}.")
        
        look_ahead_distance = self.calculate_look_ahead_distance(vehicle_position, path_points)
        vec_look_ahead_point = self._look_ahead_point - vehicle_position
        vec_xaxis = [np.sin(heading), np.cos(heading)]
        alpha = self.get_angle_between_vectors(vec_xaxis, vec_look_ahead_point)
        steering_angle = np.arctan(2.0 * self._wheelbase*np.sin(alpha)/look_ahead_distance)
        return steering_angle

    def calculate_look_ahead_distance (self, vehicle_position: np.array, path: np.array) -> float:
        if vehicle_position.shape != (2,):
            raise ValueError(f"Expected vehicle_position.shape to be (2,), got {vehicle_position.shape}.")

        if path.ndim != 2 or path.shape[0] != 2:
            raise ValueError(f"Expected path.shape to be (2, N), got {path.shape}.")

        if path.shape[1] < 2:
            raise ValueError(f"Minimum of 2 points required, got {path.shape[1]}.")

        distances = np.linalg.norm(path.T - vehicle_position, axis=1)
        closest_point_index = k_smallest_indices(distances, 1)
        print("closest_point_index: ", closest_point_index)
        self._look_ahead_point = path[:, closest_point_index + 10]
        print("self._look_ahead_point: ", self._look_ahead_point)
        look_ahead_distance = np.linalg.norm(self._look_ahead_point.T - vehicle_position)
        print("look_ahead_distance:", look_ahead_distance)

        return look_ahead_distance

    def get_angle_between_vectors(self, v1, v2):
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        cos_alpha = dot_product / (norm_v1 * norm_v2)
        angle_in_rad = (np.arccos(np.clip(cos_alpha, -1.0, 1.0)) - np.pi/2)/2
        angle_in_degree = np.degrees(angle_in_rad)
        print("alpha in °: ", angle_in_degree)
        return angle_in_rad[1]