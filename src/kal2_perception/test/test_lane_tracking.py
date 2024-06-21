import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from pathlib import Path
import rospkg

from matplotlib import pyplot as plt
import cv2 as cv

from kal2_perception.lane_tracking import flip_normals


def test_flip_normals_single_feature():
    normal = np.array([[0], [1]])
    normal_flipped = -normal

    result = flip_normals(normal, direction=np.array([[0], [1]]))
    assert_array_equal(result, normal)

    result = flip_normals(normal_flipped, direction=np.array([[0], [1]]))
    assert_array_equal(result, normal)

def test_flip_normals_multiple_features():
    normal = np.array([[0, 0], [1, -1]])

    result = flip_normals(normal, direction=np.array([[0], [1]]))
    expected = np.array([[0, 0], [1, 1]])
    assert_array_equal(result, expected)