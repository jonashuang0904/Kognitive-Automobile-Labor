from dataclasses import dataclass
from typing import Any

import numpy as np
import sympy as sp


class ApproximateClothoid:
    def __init__(self, k0: float, k1: float) -> None:
        self.k0 = k0
        self.k1 = k1

    def y(self, arc_length):
        return (self.k0 / 2.0) * arc_length**2 + (self.k1 / 6.0) * arc_length**3

    def azimuth(self, arc_length):
        return self.k0 * arc_length + 0.5 * self.k1 * arc_length**2

    def curvature(self, arc_length):
        return self.k0 + self.k1 * arc_length


@dataclass
class ClothoidState:
    x: float
    y: float
    theta: float
    k0: float
    k1: float

    @property
    def roation_matrix(self) -> np.ndarray:
        return np.array([[np.cos(self.theta), -np.sin(self.theta)], [np.sin(self.theta), np.cos(self.theta)]])

    @property
    def translation(self) -> np.ndarray:
        return np.array([self.x, self.y]).reshape(2, 1)

    @property
    def coeffs(self):
        return np.array([self.k1 / 6, self.k0 / 2, 0, 0])


def wrap(alpha):
    return (alpha + np.pi) % (2.0 * np.pi) - np.pi


class PolynomialFilter:
    def __init__(self, initial_pose: np.ndarray) -> None:
        self._symbolic_state = sp.symbols("x y theta k0 k1")
        self._symbolic_input = sp.symbols("px py")
        self._numeric_state = np.array([initial_pose[0], initial_pose[1], initial_pose[2], 0.0, 0.0])
        self._state_covariance = np.eye(5) * 0.1

        x, y, theta, k0, k1 = self._symbolic_state
        px, py = self._symbolic_input

        dlong = (px - x) * sp.cos(theta) + (py - y) * sp.sin(theta)
        dlat = (k1 / 6) * dlong**3 + (k0 / 2) * dlong
        s_appox = sp.sqrt((px - x) ** 2 + (py - y) ** 2)

        x_next = px
        y_next = dlong * sp.sin(theta) + dlat * sp.cos(theta) + y
        theta_next = theta + sp.atan(k0 * s_appox + 0.5 * k1 * s_appox**2)
        k0_next = k0 + k1 * s_appox
        k1_next = k1

        self._state_transition = sp.Matrix([x_next, y_next, theta_next, k0_next, k1_next])
        self._state_transition_function = sp.lambdify([x, y, theta, k0, k1, px, py], self._state_transition)
        self._state_transition_jacobian = sp.derive_by_array(self._state_transition, [x, y, theta, k0, k1])
        self._state_transition_jacobian_function = sp.lambdify(
            [x, y, theta, k0, k1, px, py], self._state_transition_jacobian
        )

    def _compute_state_transition(self, px, py):
        x, y, theta, k0, k1 = tuple(self._numeric_state)
        return self._state_transition_function(x, y, theta, k0, k1, px, py)

    def _compute_state_transition_jacobian(self, px, py):
        x, y, theta, k0, k1 = tuple(self._numeric_state)
        F = self._state_transition_jacobian_function(x, y, theta, k0, k1, px, py)
        return np.asarray(F).squeeze()

    @property
    def state(self):
        x, y, theta, k0, k1 = tuple(self._numeric_state)
        return ClothoidState(x, y, theta, k0, k1)

    @property
    def covariance(self):
        return self._state_covariance

    def eval(self, xs: np.ndarray) -> np.ndarray:
        ys = np.polyval(self.state.coeffs, xs)
        return self.state.roation_matrix @ np.vstack(tup=(xs, ys)) + self.state.translation

    def predict(self, new_pose: np.ndarray):
        px, py = new_pose[0], new_pose[1]
        x_next = self._compute_state_transition(px, py).flatten()
        x_next[2] = wrap(x_next[2])
        self._numeric_state = x_next

        F = self._compute_state_transition_jacobian(px, py)
        P = self._state_covariance
        self._state_covariance = F @ P @ F.T + np.eye(5) * 0.1

    def update(self, new_coeffs: np.ndarray):
        x, y, theta = tuple(self._numeric_state[:3])

        a, b, c, d = tuple(new_coeffs)
        z_x = x - np.sin(theta) * d
        z_y = y + np.cos(theta) * d
        z_theta = theta + np.arctan2(c, 1)
        k0 = 2 * b
        k1 = 6 * a

        P = self._state_covariance
        H = np.eye(5)
        R = np.eye(5) * 0.1
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        y = np.array([z_x, z_y, z_theta, k0, k1]) - self._numeric_state
        y[2] = wrap(y[2])

        self._numeric_state += K @ y
        self._numeric_state[2] = wrap(self._numeric_state[2])
        self._state_covariance = (np.eye(5) - K @ H) @ self._state_covariance


class PolynomialFilterV3:
    def __init__(self, initial_pose) -> None:
        self._state = ClothoidState(x=initial_pose[0], y=initial_pose[1], theta=initial_pose[2], k0=0, k1=0)
        self._current_vehicle_pose = initial_pose

    def eval(self, xs: np.ndarray) -> np.ndarray:
        return np.polyval(self._state.coeffs, xs)

    def eval_world(self, xs: np.ndarray) -> np.ndarray:
        ys = np.polyval(self._state.coeffs, xs)
        return self._state.roation_matrix @ np.vstack(tup=(xs, ys)) + np.vstack((self._state.x, self._state.y))

    def to_lane_coordinates(self, points):
        return np.linalg.inv(self._state.roation_matrix) @ (points - self._state.translation)

    def to_world_coordinates(self, points):
        return self._state.roation_matrix @ points + self._state.translation

    @property
    def state(self):
        return self._state

    def predict(self, new_pose: np.ndarray) -> None:
        rel_translation = self.to_lane_coordinates(new_pose[:2, np.newaxis]).reshape(-1)
        d_long, d_lat = rel_translation[0], rel_translation[1]

        # x_center_line, y_center_line = closest_point_on_polynomial(coeffs=self._state.coeffs, x0=d_long, y0=d_lat, initial_guess=d_long)
        x_center_line, y_center_line = d_long, self.eval(d_long)
        new_origin = self.to_world_coordinates(points=np.vstack((x_center_line, y_center_line))).reshape(-1)
        self._state.x = new_origin[0]
        self._state.y = new_origin[1]

        clothoid = ApproximateClothoid(self._state.k0, self._state.k1)

        arc_length = np.sqrt(x_center_line**2 + y_center_line**2)
        self._state.k0 = clothoid.curvature(arc_length=arc_length)
        self._state.theta += np.arctan(clothoid.azimuth(arc_length=arc_length))

        self._current_pose = new_pose

    def update(self, coeffs):
        a, b, c, d = coeffs

        k0 = 2 * b
        k1 = 6 * a

        alpha = 0.90
        self._state.k0 = alpha * self._state.k0 + (1 - alpha) * k0
        self._state.k1 = alpha * self._state.k1 + (1 - alpha) * k1
