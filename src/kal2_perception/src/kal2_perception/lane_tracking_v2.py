import time

from typing import Tuple, List, Optional
from itertools import product

import numpy as np
import numba as nb
from scipy.optimize import least_squares
from numba import njit
import rospy
import logging
import warnings
import pickle

from sklearnex import patch_sklearn
patch_sklearn()

logger = logging.getLogger('sklearnex')
logger.setLevel(logging.ERROR)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import DBSCAN
from sklearn.linear_model import Ridge

from kal2_perception.lane_model import PolynomialFilterV4
from kal2_perception.ransac import RansacLaneFinder

NUMBA_USE_CACHE=True

class Observation:
    def __init__(self, pose: np.ndarray, features: np.ndarray):
        self._pose = pose
        self._features = features

    @property
    def pose(self):
        return self._pose
    
    @property
    def features(self):
        return self._features
    
    @property
    def points(self):
        return self._features[:2]
    
    @property
    def normals(self):
        return self._features[2:]
    
    @property
    def rotation_matrix(self):
        return self._pose[:2, :2]
    
    @property
    def translation(self):
        return self._pose[:2, 3].reshape(2, 1)

# @njit((nb.float64[:, :], nb.int64[:], nb.int64[:], nb.int64), cache=NUMBA_USE_CACHE)
@njit(cache=NUMBA_USE_CACHE)
def _make_features_individual_coeffs(clustered_points: np.ndarray, valid_labels: np.ndarray, unique_labels: np.ndarray, degree: int = 3):
    n_coeff = degree + 1
    n_samples = clustered_points.shape[1]

    X = np.zeros(shape=(n_samples, n_coeff * len(unique_labels)), dtype=np.float64)

    for index, label in enumerate(unique_labels):
        start, stop = n_coeff * index, n_coeff * (index + 1)
        mask = label == valid_labels

        X[mask, start:stop] = np.vander(clustered_points[0, mask], n_coeff)

    return X, clustered_points[1]

# @njit(nb.float64[:, :](nb.int64, nb.int64), cache=NUMBA_USE_CACHE)
@njit(cache=NUMBA_USE_CACHE)
def _make_penalty_matrix(n_clusters: int, degree: int) -> np.ndarray:
    n_coeff = degree + 1
    penalty_matrix = np.zeros((n_clusters * (n_clusters - 1) // 2 * degree, n_clusters * n_coeff), dtype=np.float64)
    row = 0
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            for k in range(0, n_coeff - 1):  # Exclude intercept (k=3)
                penalty_matrix[row, i * n_coeff + k] = 1
                penalty_matrix[row, j * n_coeff + k] = -1
                row += 1

    return penalty_matrix


class LeastSquaresLaneFinder:
    def __init__(self, lane_width: float, alpha: float = 0.1) -> None:
        self._alpha = alpha
        self._beta = 1.5
        self._epsilon = 0.2
        self._degree = 3

    def _fit_polynomial(self, points: np.ndarray, labels: np.ndarray, unique_labels: np.ndarray):
        n_clusters = len(unique_labels)

        if n_clusters == 0:
            raise ValueError(f"Need at least one cluster: {n_clusters}")
        
        if points.ndim != 2 or points.shape[0] != 2 or points.shape[1] == 0:
            raise ValueError(f"Point should be a (2,N) array, got {points.shape}")

        X_data, y_data = _make_features_individual_coeffs(points,labels, unique_labels, self._degree)

        X_penalty = np.sqrt(self._beta) * _make_penalty_matrix(n_clusters, self._degree)
        y_penalty = np.zeros(X_penalty.shape[0])

        X_combined = np.vstack([X_data, X_penalty])
        y_combined = np.concatenate([y_data, y_penalty])

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=UserWarning)
            reg = Ridge(alpha=self._alpha).fit(X_combined, y_combined)
        
        return reg.coef_.reshape((n_clusters, self._degree + 1))

    def find_lanes(self, points: np.ndarray, normals: np.ndarray, return_debug: bool = False) -> np.ndarray:
        features = np.vstack((points, normals))
        poly_features = PolynomialFeatures(degree=2).fit_transform(features.T)

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=UserWarning)
            clustering = DBSCAN(eps=self._epsilon, min_samples=10).fit(poly_features)

        labels = clustering.labels_
        unique_labels, counts = np.unique(labels[labels>=0], return_counts=True)
        unique_labels =  unique_labels[np.argsort(-counts)][:5]

        mask = np.logical_or.reduce([labels == label for label in unique_labels])
        valid_labels = labels[mask]
        clustered_points = points[:, mask]

        skip = max(len(valid_labels) // 200, 1)
        unique_labels = np.unique(valid_labels[::skip])

        coeffs = self._fit_polynomial(clustered_points[:, ::skip], valid_labels[::skip], unique_labels)

        if return_debug:
            return np.mean(coeffs, axis=0), clustered_points[:, ::skip], valid_labels[::skip], coeffs
        
        return np.mean(coeffs, axis=0)

class LaneMap:
    def __init__(self, initial_pose, lane_width: float = 0.9, look_ahead_distance: float = 3.0):
        self._lane_width: float = lane_width
        self._look_ahead_distance: float = look_ahead_distance
        self._observations: List[Observation] = []
        self._filter = PolynomialFilterV4(initial_pose=None)
        self._center_line = []

        self._coefficients = None
        self._center_line_points = []

    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump(obj=self._observations, file=file)

    def get_features(self, n_nodes: int):
        return np.concatenate([o.features for o in self._observations[-n_nodes:]], axis=1)
    
    def get_recent_observations(self, max_distance: float = 2) -> List[Observation]:
        n_nodes = len(self._observations)

        if n_nodes == 0:
            return []

        position = self._observations[-1].translation
        recent_nodes = []

        for index in range(1, n_nodes+1):
            node = self._observations[-index]
            dist = np.linalg.norm(node.translation - position)

            if dist > max_distance:
                break

            recent_nodes.append(node)

        return recent_nodes
    
    def _make_center_line(self):
        xs = np.linspace(start=-0.15, stop=2.0, num=10)
        center_points = self._filter.eval(xs)

        return center_points

    def update(self, features: np.ndarray, vehicle_pose: np.ndarray):
        self._observations.append(Observation(pose=vehicle_pose, features=features))

        x, y = vehicle_pose[:2, 3]
        theta = np.arctan2(vehicle_pose[1, 0], vehicle_pose[0, 0])
        self._filter.predict(new_pose=np.array([x, y, theta]))

        recents_observations = self.get_recent_observations()
        recent_features = np.concatenate([o.features for o in recents_observations], axis=1)

        R = vehicle_pose[:2, :2]
        t = vehicle_pose[:2, 3].reshape(2, 1)
        local_points = np.linalg.inv(R) @ (recent_features[:2] -t)

        mask  = (local_points[0] > -1.0) & (local_points[0] < 1.0) & (local_points[1] > -1.5) & (local_points[1] < 1.5)
        local_normals =  np.linalg.inv(R) @ (recent_features[2:, mask])

        if np.sum(mask) == 0 or local_points.shape[1] < 100:
            return self._make_center_line()

        lane_finder = LeastSquaresLaneFinder(lane_width=self._lane_width)

        try:
            coeffs = lane_finder.find_lanes(local_points[:, mask], local_normals)
        except ValueError as e:
            rospy.logerr(e)
            return self._make_center_line()
        
        a, b, c, d = tuple(coeffs)
        self._filter.update(np.array([d, c, b, a]))
        self._center_line.append(self._filter.state.translation.T)

        return self._make_center_line()
