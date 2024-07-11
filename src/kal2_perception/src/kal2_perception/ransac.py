import numpy as np

from numba import njit, prange

NUMBA_USE_CACHE=True


@njit(cache=NUMBA_USE_CACHE)
def norm(array: np.ndarray):
    for i in range(array.shape[1]):
        nrm = np.linalg.norm(array[:, i])
        array[:, i] /= nrm
    return array

@njit(cache=NUMBA_USE_CACHE)
def fit_polynomial_ridge(x: np.ndarray, y: np.ndarray, alpha: float, degree: int = 3):
    X = np.vander(x, degree + 1)

    A = np.eye(degree + 1, dtype=np.float64) * alpha
    A[-1, -1] = 0

    return np.linalg.inv(np.dot(X.T, X) + A).dot(X.T).dot(y)


@njit(cache=NUMBA_USE_CACHE)
def evaluate_cubic_polynomial(coeffs: np.ndarray, xs: np.ndarray):
    return coeffs[0] * xs**3 + coeffs[1] * xs**2 + coeffs[2] * xs + coeffs[3]

@njit(cache=NUMBA_USE_CACHE)
def count_inliers(points: np.ndarray, coeffs: np.ndarray, lane_width: float, max_distance: float):
    y_center = evaluate_cubic_polynomial(coeffs, points[0, :])
    y_left = y_center + lane_width / 2
    y_right = y_center - lane_width / 2

    def is_inlier(y_pred):
        return np.abs(points[1, :] - y_pred) < max_distance
    
    return np.sum(is_inlier(y_center)) + np.sum(is_inlier(y_left)) + np.sum(is_inlier(y_right))


@njit(parallel=True, cache=NUMBA_USE_CACHE)
def find_lanes_ransac_impl(
    points: np.ndarray,
    scores: np.ndarray,
    coefficients: np.ndarray,
    indices: np.ndarray,
    lane_width: float,
    max_distance: float,
    num_iterations: int,
    alpha: float
):
    for iteration in prange(num_iterations):
        idx = indices[iteration]
        x, y = points[:, idx]

        coeffs_center = fit_polynomial_ridge(x, y, alpha=alpha)
        score_center = count_inliers(points, coeffs_center, lane_width, max_distance)

        coeffs_left = coeffs_center + np.array([0, 0, 0, lane_width / 2])
        score_left = count_inliers(points, coeffs_left, lane_width, max_distance)

        coeffs_right = coeffs_center - np.array([0, 0, 0, lane_width / 2])
        score_right = count_inliers(points, coeffs_left, lane_width, max_distance)

        local_scores = np.array((score_center, score_left, score_right), dtype=np.float64)
        local_coeffs = np.vstack((coeffs_center, coeffs_left, coeffs_right))

        if score_center >= score_left and score_center >= score_right:
            best_index = 0
        elif score_left >= score_center and score_left >= score_right:
            best_index = 1
        else:
            best_index = 2

        scores[iteration] = local_scores[best_index]
        coefficients[iteration] = local_coeffs[best_index]

@njit(cache=NUMBA_USE_CACHE)
def fit_polynomial_normal_ridge(features: np.ndarray, alpha: float, degree: int = 3):
    features = features[:, np.abs(features[3]) > np.sqrt(2)/2]

    num_features = features.shape[1]

    X_points = np.vander(features[0], degree + 1)
    X_normals = np.vander(features[0], degree)
    X_normals[:, 0] *= 3
    X_normals[:, 0] *= 2

    X = np.zeros(shape=(num_features * 2, degree + 1))
    X[:num_features] = X_points
    X[num_features:, :degree] = X_normals

    dx_dy = features[2] / features[3]
    y = np.hstack((features[1], -dx_dy))

    A = np.eye(degree + 1, dtype=np.float64) * alpha
    A[-1, -1] = 0

    return np.linalg.inv(np.dot(X.T, X) + A).dot(X.T).dot(y)

@njit(parallel=True, cache=NUMBA_USE_CACHE)
def find_lanes_ransac_normal_impl(
    features: np.ndarray,
    scores: np.ndarray,
    coefficients: np.ndarray,
    indices: np.ndarray,
    lane_width: float,
    max_distance: float,
    num_iterations: int,
    alpha: float
):
    for iteration in prange(num_iterations):
        idx = indices[iteration]
        x, y = features[:2, idx]
        nx, ny = features[2:, idx]

        try:
            coeffs_center = fit_polynomial_normal_ridge(features[:, idx], alpha=alpha)
        except Exception: #np.linalg.LinAlgError:
            continue

        score_center = count_inliers(features, coeffs_center, lane_width, max_distance)

        coeffs_left = coeffs_center + np.array([0, 0, 0, lane_width / 2])
        score_left = count_inliers(features, coeffs_left, lane_width, max_distance)

        coeffs_right = coeffs_center - np.array([0, 0, 0, lane_width / 2])
        score_right = count_inliers(features, coeffs_left, lane_width, max_distance)

        local_scores = np.array((score_center, score_left, score_right), dtype=np.float64)
        local_coeffs = np.vstack((coeffs_center, coeffs_left, coeffs_right))

        if score_center >= score_left and score_center >= score_right:
            best_index = 0
        elif score_left >= score_center and score_left >= score_right:
            best_index = 1
        else:
            best_index = 2

        scores[iteration] = local_scores[best_index]
        coefficients[iteration] = local_coeffs[best_index]



class RansacLaneFinder:
    def __init__(self, lane_width: float, alpha: float, num_iterations: int = 1000, seed: int = 42) -> None:
        self._lane_width = lane_width
        self._alpha = alpha
        self._max_distance = 0.15
        self._num_iterations = num_iterations
        self._seed = seed

    def find_lanes(self, points: np.ndarray):
        if points.ndim != 2 or points.shape[0] != 2:
            raise ValueError(f"Expected points to have shape (2, N), got {points.shape}.")
        
        n_points = points.shape[1]
        num_iterations = self._num_iterations
        rng = np.random.default_rng(seed=self._seed)

        scores = np.zeros(num_iterations, dtype=np.float64)
        coefficients = np.zeros((num_iterations, 4), dtype=np.float64)
        indices = rng.choice(n_points, (num_iterations, 4))

        find_lanes_ransac_impl(
            points,
            scores,
            coefficients,
            indices,
            lane_width=self._lane_width,
            max_distance=self._max_distance,
            num_iterations=num_iterations, alpha=self._alpha #  0.1
        )

        best_index = np.argmax(scores)
        best_coeffs = coefficients[best_index]

        return best_coeffs
    
    def find_lanes_with_normals(self, features: np.ndarray):
        if features.ndim != 2 or features.shape[0] != 4:
            raise ValueError(f"Expected points to have shape (4, N), got {features.shape}.")
        
        n_points = features.shape[1]
        num_iterations = self._num_iterations
        rng = np.random.default_rng(seed=self._seed)

        scores = np.zeros(num_iterations, dtype=np.float64)
        coefficients = np.zeros((num_iterations, 4), dtype=np.float64)
        indices = rng.choice(n_points, (num_iterations, 4))

        find_lanes_ransac_normal_impl(
            features,
            scores,
            coefficients,
            indices,
            lane_width=self._lane_width,
            max_distance=self._max_distance,
            num_iterations=num_iterations,
            alpha=self._alpha # 1.0
        )

        best_index = np.argmax(scores)
        best_coeffs = coefficients[best_index]

        return best_coeffs
