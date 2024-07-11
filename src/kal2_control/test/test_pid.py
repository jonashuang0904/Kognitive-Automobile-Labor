import pytest
import numpy as np

from kal2_control.pid import distance_point_to_line, calculate_cte


def test_distance_point_to_line():
    line_start = np.array([0, 0])
    line_end = np.array([1, 0])
    point = np.array([0.5, 0.5])

    distance = distance_point_to_line(point, line_start, line_end)
    assert np.abs(distance - 0.5) < 0.001


def test_distance_point_to_line_with_tuple():
    line_start = (0, 0)
    line_end = np.array([1, 0])
    point = np.array([0.5, 0.5])

    distance = distance_point_to_line(point, line_start, line_end)
    assert distance == 0.5


def test_calculate_cte():
    position = np.array([0, 0])
    path = np.array([[2, 0], [1, 0], [0, 0]]).T

    calculate_cte(position, path)

    assert False