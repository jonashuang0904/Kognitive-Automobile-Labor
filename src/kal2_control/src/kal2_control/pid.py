import numpy as np

class PID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_cte = 0
        self.int_cte = 0
        self.segment_start = None
        self.segment_end = None

    def control(self, cte, dt):
        p_term = self.Kp * cte
        self.int_cte += cte * dt
        i_term = self.Ki * self.int_cte
        d_term = self.Kd * (cte - self.prev_cte) / dt
        self.prev_cte = cte

        control_output = p_term + i_term + d_term
        return control_output

def distance_point_to_line(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray):
    if isinstance(point, tuple):
        point = np.array(point)

    if isinstance(line_start, tuple):
        line_start = np.array(line_start)

    if isinstance(line_end, tuple):
        line_end = np.array(line_end)

    # Richtungsvektoren berechnen
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len = np.linalg.norm(line_vec)

    if  np.all(point_vec != 0):
        projection = np.dot(point_vec, line_vec) / line_len
        closest_point = line_start + projection * (line_vec / line_len)

        distance = np.linalg.norm(point - closest_point)

        # Bestimmung der Seite des Punktes bezüglich der Linie
        side = np.sign(np.cross(point_vec, line_vec))
        return side * distance

    else:
        return 0.0

def k_smallest_indices(arr: np.ndarray, k: int):
    if k <= 0:
        return np.array([], dtype=int)
    if k >= arr.size:
        return np.argsort(arr)

    indices = np.argpartition(arr, k)[:k]
    return indices[np.argsort(arr[indices])]

def calculate_cte(vehicle_position: np.array, path: np.array) -> float:
    if vehicle_position.shape != (2,):
        raise ValueError(f"Expected vehicle_position.shape to be (2,), got {vehicle_position.shape}.")

    if path.ndim != 2 or path.shape[0] != 2:
        raise ValueError(f"Expected path.shape to be (2, N), got {path.shape}.")

    if path.shape[1] < 2:
        raise ValueError(f"Minimum of 2 points required, got {path.shape[1]}.")

    distances = np.linalg.norm(path.T - vehicle_position, axis=1)
    min_indices = k_smallest_indices(distances, 2)

    #print("Distances:", distances)
    #print("Min indices:", min_indices)

    segment_start = path[:, min(min_indices)]
    #print("kleinsterIndex: ", min(min_indices))
    #print("Segment start:", segment_start)

    segment_end = path[:, max(min_indices)]
    #print("größterIndex: ", max(min_indices))
    #print("Segment end:", segment_end)

    cte = distance_point_to_line(vehicle_position, segment_start, segment_end)
    #print("cte: ", cte)

    return cte