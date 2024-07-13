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


def _wrap_angle(alpha):
    return (alpha + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class Pose2D:
    x: float
    y: float
    theta: float

    @property
    def rotation_matrix(self):
        return np.array([[np.cos(self.theta), -np.sin(self.theta)], [np.sin(self.theta), np.cos(self.theta)]])

    @property
    def translation(self):
        return np.array([[self.x], [self.y]])

    def as_array(self):
        return np.array([self.x, self.y, self.theta])

    @staticmethod
    def from_array(pose: np.ndarray):
        return Pose2D(*tuple(pose))


class PolynomialFilterV4:
    def __init__(self, initial_pose: np.ndarray) -> None:
        dlat, tan_theta, k0, k1 = sp.symbols("dlat tan_theta k0 k1")
        delta_x, delta_y, delta_theta = sp.symbols("Delta_x Delta_y Delta_theta")
        a, b, c, d = sp.symbols("a, b, c, d")

        s = delta_x
        dlat_next = dlat + tan_theta * s + (k0 / 2) * s**2 + (k1 / 6) * s**3 - delta_y

        theta_next = tan_theta + k0 * s + 0.5 * k1 * s**2 - sp.tan(delta_theta)
        k0_next = k0 + k1 * s
        k1_next = k1

        self._x = sp.Matrix([dlat, tan_theta, k0, k1])
        self._u = sp.Matrix([delta_x, delta_y, delta_theta])
        self._z = sp.Matrix([d, c, b, a])

        self._state_transition_eq = sp.Matrix([dlat_next, theta_next, k0_next, k1_next])
        self._state_transition = sp.lambdify([self._x, self._u], self._state_transition_eq)
        self._Fx = sp.lambdify([self._x, self._u], self._state_transition_eq.jacobian(self._x))
        self._Fu = sp.lambdify([self._x, self._u], self._state_transition_eq.jacobian(self._u))

        self._measurement_eq = sp.Matrix([dlat, tan_theta, k0/2, k1/6])
        self._predict_measurement = sp.lambdify([self._x], self._measurement_eq)
        self._Hx = sp.lambdify([self._x, self._z], self._measurement_eq.jacobian(self._x))

        self._state = np.zeros(4)
        self._state_covariance = np.eye(4)
        self._Q = np.diag([1, 1, 1])
        self._R = np.diag([0.002, 0.001, 0.0001, 0.00001])

        self._last_pose = Pose2D.from_array(initial_pose) if initial_pose is not None else None

    def display_equations(self):
        from IPython.display import Latex, display
        display(Latex(rf"""$$f(x, u) = {sp.latex(self._state_transition_eq)},
                       \quad \frac{{\partial f(x,u)}}{{\partial x}} = {sp.latex(self._state_transition_eq.jacobian(self._x))},
                       \quad \frac{{\partial f(x,u)}}{{\partial u}} = {sp.latex(self._state_transition_eq.jacobian(self._u))}$$"""))
        display(Latex(rf"""$$h(x) = {sp.latex(self._measurement_eq)},
                       \quad H = {sp.latex(self._measurement_eq.jacobian(self._x))}$$"""))

    @property
    def state(self):
        y, theta, k0, k1 = tuple(self._state)
        return ClothoidState(0, y, theta, k0, k1)

    @property
    def covariance(self):
        return self._state_covariance

    def eval(self, xs: np.ndarray) -> np.ndarray:
        ys = np.polyval(self.state.coeffs, xs)
        return self._last_pose.rotation_matrix @ np.vstack(tup=(xs, ys)) + self._last_pose.translation

    def predict(self, new_pose: np.ndarray):
        if self._last_pose is None:
            self._last_pose = Pose2D.from_array(new_pose)
            return

        R = self._last_pose.rotation_matrix

        u = new_pose - self._last_pose.as_array() #TODO: check if angle needs to be wrapped.
        u[:2] = (R.T @ u[:2].reshape(2, 1)).flatten()

        x_next = self._state_transition(self._state, u).flatten()
        self._state = x_next

        Fx = self._Fx(self._state, u)
        Fu = self._Fu(self._state, u)

        P = self._state_covariance
        self._state_covariance = Fx @ P @ Fx.T + Fu @ self._Q @ Fu.T
        self._last_pose = Pose2D.from_array(new_pose)

    def update(self, z: np.ndarray):
        Hx = self._Hx(self._state, z)

        z_pred = self._predict_measurement(self._state).flatten()

        P = self._state_covariance
        K = P @ Hx.T @ np.linalg.inv(Hx @ P @ Hx.T + self._R)
        y = z - z_pred

        self._state += K @  y
        self._state_covariance = (np.eye(4) - K @ Hx) @ self._state_covariance
