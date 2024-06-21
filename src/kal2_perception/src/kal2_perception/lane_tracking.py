from typing import Tuple, NamedTuple, List

import numpy as np
import rospy
import time
import cv2 as cv

from numba import njit
from itertools import product


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
    

def search_coefficients(image: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray):
    acc_shape = (len(a), len(b), len(c), len(d))
    accumulator = np.zeros(acc_shape, dtype=int)

    resolution = 50
    half_lane_wdith = (0.9 * resolution) // 2


    x = np.linspace(0, image.shape[0]-2, image.shape[0])
    
    for ai, bi, ci, di in product(range(len(a)), range(len(b)), range(len(c)), range(len(d))):
        y_pred = (a[ai] * x**3 + b[bi] * x**2 + c[ci] * x + d[di])

        dy_dx = 3 * a[ai] * x**2 + 2 * b[bi] * x + c[ci]
        normals = np.vstack((-dy_dx, np.ones_like(dy_dx)))
        normals = normals / np.linalg.norm(normals, axis=0)
        
        y_pred_left = y_pred - half_lane_wdith * normals[1]
        x_pred_left = x - half_lane_wdith * normals[0]
        y_pred_right = y_pred + half_lane_wdith * normals[1]
        x_pred_right = x + half_lane_wdith * normals[0]

        template = np.zeros_like(image)

        def add_to_template(x, y_pred):
            points = np.vstack((y_pred, x)).astype(int).T
            cv.polylines(template, [points], isClosed=False, color=255, thickness=2)

        add_to_template(x, y_pred)
        add_to_template(x_pred_left, y_pred_left)
        add_to_template(x_pred_right, y_pred_right)

        accumulator[ai, bi, ci, di] = np.sum(cv.bitwise_and(image, template))

    max_indices = np.unravel_index(np.argmax(accumulator), acc_shape)
    
    best_a = a[max_indices[0]]
    best_b = b[max_indices[1]]
    best_c = c[max_indices[2]]
    best_d = d[max_indices[3]]
    
    return best_a, best_b, best_c, best_d

from typing import NamedTuple
from numba import prange, njit

class SearchSpace(NamedTuple):
    a: float
    b: float
    c: float
    d: float

    @property
    def shape(self):
        return (len(self.a), len(self.b), len(self.c), len(self.d))

@njit(parallel=True)
def search_coefficients_impl(image: np.ndarray, accumulator: np.ndarray, search_space: SearchSpace, lane_width: int, max_distance: int):
    points = np.where(image == 255)

    half_lane_width = lane_width // 2

    xs = points[0]
    ys = points[1]

    for ai in prange(len(search_space.a)):
        for bi in range(len(search_space.b)):
            for ci in range(len(search_space.c)):
                for di in range(len(search_space.d)):
                    a, b, c, d = search_space.a[ai], search_space.b[bi], search_space.c[ci], search_space.d[di]

                    y_pred_center = a * xs**3 + b * xs**2 + c * xs + d
                    y_pred_left = a * xs**3 + b * xs**2 + c * xs + d + half_lane_width
                    y_pred_right = a * xs**3 + b * xs**2 + c * xs + d - half_lane_width

                    diff_center = np.abs(ys - y_pred_center)
                    diff_left = np.abs(ys - y_pred_left)
                    diff_right = np.abs(ys - y_pred_right)
                    score = np.sum(diff_center < max_distance) + np.sum(diff_left < max_distance) + np.sum(diff_right < max_distance)
 
                    accumulator[ai, bi, ci, di] = score



def search_coefficients_nb(image: np.ndarray, search_space: SearchSpace, lane_width: int = 100, max_distance: int = 5):
    accumulator = np.zeros(search_space.shape, dtype=np.int64)

    search_coefficients_impl(image, accumulator, search_space, lane_width, max_distance)
    
    max_indices = np.unravel_index(np.argmax(accumulator), search_space.shape)
    
    best_a = search_space.a[max_indices[0]]
    best_b = search_space.b[max_indices[1]]
    best_c = search_space.c[max_indices[2]]
    best_d = search_space.d[max_indices[3]]
    
    return best_a, best_b, best_c, best_d
    

def detect_lane(points: np.ndarray, lane_width: float = 0.9, radius: float = 0.5):
    from kal2_perception.birds_eye_view import rasterize_points, VoxelConfig

    resolution=25
    config = VoxelConfig(resolution=resolution, x_min=0, x_max=radius, y_min=-radius, y_max=radius)
    points3d = np.vstack((points, np.zeros(points.shape[1]))).T
    print(points3d.shape)
    grid = rasterize_points(points3d, config)
    grid = (grid > 0).astype(np.uint8) * 255

    print(grid.shape)
    y0 = grid.shape[1] // 2

    a = np.linspace(-1, 1, 11) / (resolution * 1.0)**2
    b = np.linspace(-1, 1, 11) / (resolution * 1.0)
    c = np.linspace(-1, 1, 11)
    d = y0 + np.linspace(-10, 10, 11)

    coefficients = search_coefficients_nb(grid, SearchSpace(a, b, c, d), lane_width=resolution, max_distance=2)

    x = np.linspace(0, grid.shape[0]-1, grid.shape[0]).astype(int)
    y = (coefficients[0] * x**3 + coefficients[1] * x**2 + coefficients[2] * x + coefficients[3]).astype(int)

    dy_dx = 3 * coefficients[0] * x**2 + 2 * coefficients[1] * x + coefficients[2]
    normals = np.vstack((-dy_dx, np.ones_like(dy_dx)))
    normals = normals / np.linalg.norm(normals, axis=0)

    half_lane_wdith = (0.9 * resolution) // 2
    
    y_pred_left = y - half_lane_wdith * normals[1]
    x_pred_left = x - half_lane_wdith * normals[0]
    y_pred_right = y + half_lane_wdith * normals[1]
    x_pred_right = x + half_lane_wdith * normals[0]

    cv.polylines(grid, [np.vstack((y, x)).T], isClosed=False, color=255, thickness=2)
    cv.polylines(grid, [np.vstack((y_pred_left, x_pred_left)).astype(int).T], isClosed=False, color=255, thickness=2)
    cv.polylines(grid, [np.vstack((y_pred_right, x_pred_right)).astype(int).T], isClosed=False, color=255, thickness=2)
    coefficients = None

    return coefficients, grid




class LaneMap:
    def __init__(
        self, initial_pose, lane_width: float = 0.9, segment_length: float = 0.25, look_ahead_distance: float = 3.0
    ):
        self._lane_width = lane_width
        self._segment_length = segment_length
        self._look_ahead_distance = look_ahead_distance
        self._lane_segments: List[LaneSegment] = []
        self._features = []

        self._coefficients = None

        self._add_lane_segment(np.array(initial_pose))

    def _add_lane_segment(self, pose: Tuple[float, float, float]):
        self._lane_segments.append(LaneSegment(pose, self._lane_width, self._segment_length))

    def update(self, features: np.ndarray, vehicle_pose: np.ndarray):
        t0 = time.time()
        self._features.append(features[:2])
        points = np.concatenate(self._features, axis=1).T

        current_position = vehicle_pose[:2, 3]

        from scipy.spatial import KDTree
        kdtree = KDTree(points)

        indices = kdtree.query_ball_point(x=current_position, r=3)
        nearby_points = np.linalg.inv(vehicle_pose[:2, :2]) @ (points[indices].T - current_position.reshape(2, -1))
        t1 = time.time()
        print(f"KDTree took {t1 - t0:.3f}s\t n: {len(nearby_points)}/{len(points)}")


        coeffs, grid = detect_lane(nearby_points, lane_width=self._lane_width, radius=2)
        t2 = time.time()
        print(f"Detect lane took {t2 - t1:.3f}s")


        return grid

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

        n = min(10, len(self._lane_segments))

        for segment in self._lane_segments[-n:]:
            segment.observe(features)

        observations = [segment.observations for segment in self._lane_segments[-n:]]

        for index in range(n):
            current_segement = self._lane_segments[-n+index]
            visible_observations = np.concatenate(observations[index:], axis=1)

            if visible_observations.shape[1] < 50:
                continue

            rel_visible_observations = np.linalg.inv(current_segement.rotation_matrix) @ (
                visible_observations - current_segement.position
            )
            try:
                poly = np.polynomial.Polynomial.fit(rel_visible_observations[0], rel_visible_observations[1], 3)
            except np.linalg.LinAlgError:
                continue

            # print(f"x min: {rel_visible_observations[0].min()}, max: {rel_visible_observations[0].max()}")

            # print(rel_visible_observations[0].max())
            if rel_visible_observations[0].max() < 1.0:
                continue

            x = np.linspace(0, 1.0, 10)
            y = poly(x)
            predicted_path = current_segement.rotation_matrix @ np.vstack((x, y)) + current_segement.position
            current_segement.predicted_path = predicted_path
