from typing import Tuple, NamedTuple, List

import numpy as np
import rospy
import time
import cv2 as cv

from scipy.spatial import KDTree
from numba import njit
from itertools import product
from functools import cached_property

from kal2_perception.ransac import RansacLaneFinder


class LaneSegment:
    def __init__(self, pose: np.ndarray, lane_width: float, segment_length: float) -> None:
        self._pose = pose
        self._lane_width = lane_width
        self._segment_length = segment_length
        self._window_width = 0.15

        self._features = []
        self._projected_observations = []
        self.predicted_path = None

    def predict(self, pose: np.ndarray):
        pass

    def _filter_features_by_bounding_box(self, features: np.ndarray):
        if features.shape[1] == 0:
            return features

        rel_centers = np.linalg.inv(self.rotation_matrix) @ (features[:2] - self.position)

        mask = np.logical_and(
            np.abs(rel_centers[0]) < self._segment_length,
            np.abs(rel_centers[1]) < self._lane_width,
        )

        return features[:, mask]

    def _compute_projected_observations(self, features: np.ndarray):
        if features.ndim != 2 or features.shape[0] != 4:
            raise ValueError(f"Features must be a 4xN array, got {features.shape}.")

        centers, normals = features[:2], features[2:]

        mean_normal = np.mean(normals, axis=1).reshape(2, -1)
        scalar_error = (centers - self.position).T @ mean_normal

        error_left = scalar_error + (self._lane_width / 2)
        error_right = scalar_error - (self._lane_width / 2)

        errors = np.array([scalar_error, error_right, error_left])
        min_abs_indices = np.argmin(np.abs(errors), axis=0)
        scalar_error = np.choose(min_abs_indices, errors)

        return self.position + scalar_error[:].T * mean_normal, mean_normal

    @property
    def observations(self):
        if len(self._projected_observations) == 0:
            return np.empty((2, 0))

        return np.concatenate(self._projected_observations, axis=1)

    def predict(self, position):
        self._pose[:2] = 0.9 * self._pose[:2] + 0.1 * position

    def observe(self, features: np.ndarray):
        if features.ndim != 2 or features.shape[0] != 4:
            raise ValueError(f"Features must be a 4xN array, got {features.shape}.")

        filtered_features = self._filter_features_by_bounding_box(features)

        if filtered_features.shape[1] == 0:
            return

        proj_centers, mean_normal = self._compute_projected_observations(filtered_features)

        if proj_centers.ndim != 2 or proj_centers.shape[0] != 2:
            raise ValueError(f"Projected centers must be a 2xN array, got {proj_centers.shape}.")

        self._projected_observations.append(proj_centers)

        # for i in range(proj_centers.shape[1]):
        #     self._projected_observations.append(proj_centers[:, i])

        from scipy.spatial.distance import cdist

        weights = cdist(self.position.T, proj_centers.T, metric="mahalanobis", VI=np.eye(2))
        # print(weights.shape, weights.T.reshape(-1).shape, proj_centers.shape)

        # position = np.mean(proj_centers, axis=1)
        position = np.average(proj_centers, axis=1, weights=weights.T.reshape(-1))
        theta = -np.arctan2(mean_normal[0], mean_normal[1])
        self._pose[2] = 0.90 * self._pose[2] + 0.1 * theta
        self._pose[:2] = 0.90 * self._pose[:2] + 0.1 * position

    @property
    def position(self):
        return self._pose[:2].reshape(2, -1)

    @property
    def orientation(self):
        return self._pose[2]

    @property
    def rotation_matrix(self):
        theta = self.orientation
        return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    @property
    def bounding_box(self):
        half_w, half_h = self._lane_width / 2, self._segment_length / 2
        corners = np.array(
            [
                [half_h, -half_w],  # Bottom-left
                [-half_h, -half_w],  # Bottom-right
                [-half_h, half_w],  # Top-right
                [half_h, half_w],  # Top-left
            ]
        ).T

        return (self.rotation_matrix @ corners) + self.position


def fit_polynomial_normals_only_ridge(features: np.ndarray, alpha: float, degree: int = 3):
    features = features[:, np.abs(features[3]) > 0.5]

    X = np.vander(features[0], degree)
    X[:, 0] *= 3
    X[:, 0] *= 2

    dx_dy = features[2] / features[3]
    y = -dx_dy

    A = np.eye(degree, dtype=np.float64) * alpha
    A[-1, -1] = 0

    return np.linalg.inv(np.dot(X.T, X) + A).dot(X.T).dot(y)


def detect_lane(points: np.ndarray, lane_width: float = 0.9, radius: float = 0.5):
    from kal2_perception.birds_eye_view import rasterize_points, VoxelConfig

    resolution = 25
    config = VoxelConfig(resolution=resolution, x_min=0, x_max=radius, y_min=-radius, y_max=radius)
    points3d = np.vstack((points, np.zeros(points.shape[1]))).T
    grid = rasterize_points(points3d, config)
    grid = (grid > 0).astype(np.uint8) * 255

    color_grid = cv.cvtColor(grid, cv.COLOR_GRAY2BGR)

    # print(grid.shape)
    # y0 = grid.shape[1] // 2

    # a = np.linspace(-1, 1, 11) / (resolution * 1.0)**2
    # b = np.linspace(-1, 1, 11) / (resolution * 1.0)
    # c = np.linspace(-1, 1, 11)
    # d = y0 + np.linspace(-10, 10, 11)

    # coefficients = search_coefficients_nb(grid, SearchSpace(a, b, c, d), lane_width=resolution, max_distance=2)

    # x = np.linspace(0, grid.shape[0]-1, grid.shape[0]).astype(int)
    # y = (coefficients[0] * x**3 + coefficients[1] * x**2 + coefficients[2] * x + coefficients[3]).astype(int)

    # dy_dx = 3 * coefficients[0] * x**2 + 2 * coefficients[1] * x + coefficients[2]
    # normals = np.vstack((-dy_dx, np.ones_like(dy_dx)))
    # normals = normals / np.linalg.norm(normals, axis=0)

    # half_lane_wdith = (0.9 * resolution) // 2

    # y_pred_left = y - half_lane_wdith * normals[1]
    # x_pred_left = x - half_lane_wdith * normals[0]
    # y_pred_right = y + half_lane_wdith * normals[1]
    # x_pred_right = x + half_lane_wdith * normals[0]

    # cv.polylines(grid, [np.vstack((y, x)).T], isClosed=False, color=255, thickness=2)
    # cv.polylines(grid, [np.vstack((y_pred_left, x_pred_left)).astype(int).T], isClosed=False, color=255, thickness=2)
    # cv.polylines(grid, [np.vstack((y_pred_right, x_pred_right)).astype(int).T], isClosed=False, color=255, thickness=2)
    # coefficients = None
    from kal2_perception.ransac import RansacLaneFinder

    ransac = RansacLaneFinder(lane_width=0.9, alpha=1, num_iterations=500)
    coeffs = ransac.find_lanes(points)

    x = np.linspace(points[0].min(), points[0].max(), 20)
    y = np.polyval(coeffs, x)

    lane_points = np.vstack((x, y, np.zeros_like(x)))
    lane = rasterize_points(lane_points.T, config)
    color_grid[lane != 0] = (127, 255, 255)

    from sklearn.cluster import DBSCAN

    db = DBSCAN(eps=0.15, min_samples=20).fit(points.T)
    labels = db.labels_

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    for label in np.unique(labels)[1:6]:
        cluster = rasterize_points(points3d[labels == label], config)
        color_grid[cluster != 0] = colors[label]

    coefficients = coeffs
    return coefficients, color_grid


class Observation:
    def __init__(self, id: int, pose: np.ndarray, features: np.ndarray):
        self._pose = pose
        self._features = features
        self._curve = None
        self._id = id
        self._predicted_centers = []

    def update_curve(self, curve: np.ndarray):
        self._curve = curve

    def predict(self, positions):
        R_inv = np.linalg.inv(self.pose[:2, :2])
        t = self.position.reshape(2, 1)

        relative_position = R_inv @ (positions - t)

        x_min, x_max, y_min, y_max = self.get_bounding_box()
        mask = (
            (relative_position[0] > x_min)
            & (relative_position[0] < x_max)
            & (relative_position[1] > y_min)
            & (relative_position[1] < y_max)
        )

        coeffs = self._curve.copy()
        coeffs[-1] = 0

        ys = np.polyval(coeffs, relative_position[0])

        # offset = R_inv @ (self.predicted_center - t)
        relative_predictions = np.vstack((relative_position[0], ys))

        predictions = (self.pose[:2, :2] @ relative_predictions) + t
        predictions[:, ~mask] = np.nan
        return predictions
    
    def predict_all(self):
        x_min, x_max, _, _ = self.get_bounding_box()

        xs = np.linspace(x_min, x_max, 20)
        coeffs = self._curve.copy()
        coeffs[-1] = 0

        ys = np.polyval(coeffs, xs)

        # offset = R_inv @ (self.predicted_center - t)
        relative_predictions = np.vstack((xs, ys))

        return (self.pose[:2, :2] @ relative_predictions) + self.position
    
    def observe(self, positions):
        if np.any(np.isnan(positions)):
            return
        
        self._predicted_centers.append(positions)

    @property
    def predicted_center(self):
        if len(self._predicted_centers) == 0:
            return self.position
        predictions = np.array(self._predicted_centers).T
        median = np.median(predictions, axis=1)
        return median.reshape(2, 1)

    @property
    def pose(self):
        return self._pose

    @property
    def position(self):
        return self._pose[:2, 3].reshape(2, 1)

    @property
    def features(self):
        return self._features

    @cached_property
    def relative_features(self):
        R_inv = np.linalg.inv(self._pose[:2, :2])
        t = self._pose[:2, 3].reshape(2, 1)

        points = R_inv @ (self._features[:2] - t)
        normals = R_inv @ self._features[2:]

        return np.vstack((points, normals))

    def get_bounding_box(self, relative: bool = True):
        features = self.relative_features if relative else self.features
        return features[0].min(), features[0].max(), features[1].min(), features[1].max()


class ObservationAccumulator:
    def __init__(self):
        self._observations: List[Observation] = []

    def save(self):
        import pickle

        with open("/home/josua/kal2_ws/src/kal2_perception/notebooks/observations.pickle", "wb") as fp:
            pickle.dump(self._observations, fp)

    def add_observation(self, pose: np.ndarray, features: np.ndarray):
        if pose.shape != (4, 4):
            raise ValueError(f"Expected pose to be a (4,4) matrix, got {pose.shape}.")

        if features.ndim != 2 or features.shape[0] != 4:
            raise ValueError(f"Expected features to have shape (4, N), got {features.shape}.")

        self._observations.append(Observation(id=len(self._observations), pose=pose, features=features))

        # if len(self._observations) == 300:
        #     self.save()
        #     rospy.logwarn("Saving observations...")

    def get_nearby_points(self, pose: np.ndarray, radius: float, transform_to_local: bool = True):
        if pose.shape != (4, 4):
            raise ValueError(f"Expected pose to be a (4,4) matrix, got {pose.shape}.")
        
        observations = self.get_recent_nodes()

        features = np.concatenate([observation.features for observation in observations], axis=1)
        points = features[:2, :]
        normals = features[2:, :]

        R_inv = np.linalg.inv(pose[:2, :2])
        t = pose[:2, 3].reshape(2, 1)

        kdtree = KDTree(points.T)
        indices = kdtree.query_ball_point(x=pose[:2, 3], r=radius)

        if transform_to_local:
            points = R_inv @ (points[:, indices] - t)
            normals = R_inv @ normals[:, indices]
        else:
            points = points[:, indices]
            normals = normals[:, indices]

        return points, normals
    
    def get_recent_nodes(self, max_distance: float = 2):
        n_nodes = len(self._observations)

        if n_nodes == 0:
            return []

        position = self._observations[-1].pose[:2, 3]
        recent_nodes = []

        for index in range(1, n_nodes+1):
            node = self._observations[-index]
            dist = np.linalg.norm(node.pose[:2, 3] - position)

            if dist > max_distance:
                break

            recent_nodes.append(node)

        return recent_nodes



    def estimate_lane(self, index: int = -1, radius: float = 3.0, lane_width: float = 0.9):
        node = self._observations[index]

        points, normals = self.get_nearby_points(node._pose, radius=radius)

        ransac = RansacLaneFinder(lane_width=lane_width, alpha=1, num_iterations=500)
        coeffs = ransac.find_lanes_with_normals(np.vstack((points, normals)))

        node.update_curve(coeffs)

    def predict_center_line(self, n: int = 20):
        observations = self.get_recent_nodes()#self._observations[-n:]

        positions = np.concatenate([node.position for node in observations], axis=1)

        for i, node in enumerate(observations):
            predictions = node.predict(positions[:, i:])

            for other_node, prediction in zip(observations[i:], predictions.T):
                other_node.observe(prediction)

        current_pose = self._observations[-1].pose
        center_points = np.hstack([observation.predict_all() for  observation in observations])
        relative_points = np.linalg.inv(current_pose[:2, :2]) @ (center_points - current_pose[:2, 3].reshape(2, 1))

        path = np.polyfit(relative_points[0], relative_points[1], deg=3)
        path[-1] = 0

        xs = np.linspace(-2, 2, 20)
        ys = np.polyval(path, xs)
        predicted_points = np.vstack((xs, ys))
        predicted_points = current_pose[:2, :2] @ predicted_points + current_pose[:2, 3].reshape(2, 1)

        # return np.hstack([observation.predicted_center for observation in observations])
        return predicted_points
    
    def get_features(self, n_nodes: int):
        return np.concatenate([observation.features for observation in self._observations[-n_nodes:]], axis=1)


class LaneMap:
    def __init__(
        self, initial_pose, lane_width: float = 0.9, segment_length: float = 0.25, look_ahead_distance: float = 3.0
    ):
        self._lane_width = lane_width
        self._segment_length = segment_length
        self._look_ahead_distance = look_ahead_distance
        self._lane_segments: List[LaneSegment] = []
        self._features = []
        self._observations = ObservationAccumulator()

        self._coefficients = None
        self._center_line_points = []

        self._add_lane_segment(np.array(initial_pose))

    @property
    def features(self):
        return self._observations.get_features(20)

    def _add_lane_segment(self, pose: Tuple[float, float, float]):
        self._lane_segments.append(LaneSegment(pose, self._lane_width, self._segment_length))

    def update(self, features: np.ndarray, vehicle_pose: np.ndarray):
        t0 = time.time()

        self._observations.add_observation(pose=vehicle_pose, features=features)
        # points, normals = self._observations.get_nearby_points(vehicle_pose, radius=3.0)
        self._observations.estimate_lane()
        center_points = self._observations.predict_center_line()

        # coeffs, grid = detect_lane(points, lane_width=self._lane_width, radius=2)
        return center_points

        coeffs_normals = fit_polynomial_normals_only_ridge(acc_features, alpha=1)
        coeffs_normals = np.hstack((coeffs_normals, 0))
        # coeffs=np.mean(np.vstack((coeffs, coeffs_normals)), axis=0)
        coeffs = coeffs_normals
        t2 = time.time()
        print(f"Detect lane took {t2 - t1:.3f}s")

        last_segment = self._lane_segments[-1]
        current_position = (vehicle_pose @ np.array([[0], [0], [0], [1]]))[:2]

        distance_to_last_segment = np.linalg.norm(current_position - last_segment.position)

        if distance_to_last_segment < self._look_ahead_distance:
            if len(self._lane_segments) > 1:
                second_to_last_segment = self._lane_segments[-2]
                theta = last_segment.orientation - (last_segment.orientation - second_to_last_segment.orientation)
                R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                position = (last_segment.position + R @ np.array([[self._segment_length], [0]])).squeeze()
                print(f"delta theta: {last_segment.orientation - second_to_last_segment.orientation}")
            else:
                R = last_segment.rotation_matrix
                position = (last_segment.position + R @ np.array([[self._segment_length], [0]])).squeeze()
                theta = last_segment.orientation

            rospy.loginfo(f"adding new segment at {position}")
            self._add_lane_segment(np.array([position[0], position[1], theta]))

        if len(features) == 0:
            return

        # if np.abs(coeffs[3] - 0.25) > 0.3:
        #     coeffs[3] -= np.sign(coeffs[3]) * self._lane_width // 2

        print(coeffs[3])

        coeffs[3] = 0.25

        x_min = nearby_points[0].min()
        x_max = nearby_points[0].max()
        x_pred = np.linspace(x_min, x_max, 20)
        y_pred = np.polyval(coeffs, x_pred)
        pred = vehicle_pose[:2, :2] @ np.vstack((x_pred, y_pred)) + current_position.reshape(2, -1)
        print(pred.shape)
        self._center_line_points.append(pred.T)

        # for segment in self._lane_segments:
        #     relative_position = np.linalg.inv(vehicle_pose[:2, :2]) @ (segment.position - current_position.reshape(2, -1))
        #     if x_min <= relative_position[0] <= x_max:
        #         y_pred = np.polyval(coeffs, relative_position[0])
        #         rel_predicted_position = np.array((relative_position[0], y_pred)).reshape(2, -1)
        #         predicted_position = vehicle_pose[:2, :2] @ rel_predicted_position + current_position.reshape(2, -1)
        #         print(predicted_position.shape)
        #         segment.predict(predicted_position.T)

        n = min(10, len(self._lane_segments))

        for segment in self._lane_segments[-n:]:
            segment.observe(features)

        # observations = [segment.observations for segment in self._lane_segments[-n:]]

        # for index in range(n):
        #     current_segement = self._lane_segments[-n+index]
        #     visible_observations = np.concatenate(observations[index:], axis=1)

        #     if visible_observations.shape[1] < 50:
        #         continue

        #     rel_visible_observations = np.linalg.inv(current_segement.rotation_matrix) @ (
        #         visible_observations - current_segement.position
        #     )
        #     try:
        #         poly = np.polynomial.Polynomial.fit(rel_visible_observations[0], rel_visible_observations[1], 3)
        #     except np.linalg.LinAlgError:
        #         continue

        #     # print(f"x min: {rel_visible_observations[0].min()}, max: {rel_visible_observations[0].max()}")

        #     # print(rel_visible_observations[0].max())
        #     if rel_visible_observations[0].max() < 1.0:
        #         continue

        #     x = np.linspace(0, 1.0, 10)
        #     y = poly(x)
        #     predicted_path = current_segement.rotation_matrix @ np.vstack((x, y)) + current_segement.position
        #     current_segement.predicted_path = predicted_path

        return grid
