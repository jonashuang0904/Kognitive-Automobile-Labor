import numpy as np

class PurePursuitController:
    def __init__(self) -> None:
        self._look_ahead_distance = 1.0
        self._wheel_base = 0.3

    def update(self, rotation_matrix: np.array, translation: np.array, path_points: np.array) -> float:

        translation = translation.reshape(2, 1)
        
        local_points = np.linalg.inv(rotation_matrix) @ (path_points - translation)        
        local_points = local_points[:, local_points[0] > 0]
    
        if local_points.shape[1] == 0:
            raise ValueError("No pose infront of vehicle")
        
        distances = np.linalg.norm(local_points, axis=0)
        distances[distances < self._look_ahead_distance] = np.inf

        look_ahead_point = local_points[:, np.argmin(distances)]
        alpha = np.arctan2(look_ahead_point[1], look_ahead_point[0])
        
        delta = - np.arctan2(2.0 * self._wheel_base * np.sin(alpha), self._look_ahead_distance)
    
        return delta

