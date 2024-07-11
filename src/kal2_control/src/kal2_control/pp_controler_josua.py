import numpy as np

class PurePursuitController:
    def __init__(self) -> None:
        self._look_ahead_distance = 1.0
        self._wheel_base = 0.3

    def update(self, rotation_matrix: np.array, translation: np.array, path_points: np.array) -> float:

        print("rotation matrix:", rotation_matrix.shape)
        print("path points shape:", path_points.shape)

        #translation = np.array([3,0])
        translation = translation.reshape(2, 1)
        print("translation vector:", translation)

        #path_points = np.array([translation[0]+2, translation[1]])
        print("path points shape:", path_points.shape)
        #print("path point:", path_points)
        
        local_points = np.linalg.inv(rotation_matrix) @ (path_points - translation)
        #print("local points:", local_points)
        
        local_points = local_points[:, local_points[0] > 0]
        print("local points:", local_points.shape[1])

        if local_points.shape[1] == 0:
            raise ValueError("No pose infront of vehicle")
        
        distances = np.linalg.norm(local_points, axis=0)
        distances[distances < self._look_ahead_distance] = np.inf
        look_ahead_point = local_points[:, np.argmin(distances)]
        print("look_ahead_point:", look_ahead_point)

        alpha = np.arctan2(look_ahead_point[1], look_ahead_point[0])
        print("alpha:", alpha)
        delta = - np.arctan2(2.0 * self._wheel_base * np.sin(alpha), self._look_ahead_distance)
        print ("delta:", delta)

        return delta

