import pytest
import numpy as np
from numpy.testing import assert_allclose

from kal2_perception.lane_tracker import LaneSegment


def test_lane_segment_bounding_box():
    lane_segment = LaneSegment(pose=[0, 0, 0], lane_width=2.0, segment_length=1.0)
    bbox = lane_segment.bounding_box

    assert bbox.shape == (2, 4)
    assert_allclose(bbox[:, 0], (0.5, -1.0)), "Bottom left corner is not correct."
    assert_allclose(bbox[:, 1], (-0.5, -1.0)), "Bottom right corner is not correct."
    assert_allclose(bbox[:, 2], (-0.5, 1.0)), "Top right corner is not correct."
    assert_allclose(bbox[:, 3], (0.5, 1.0)), "Top left corner is not correct."


def test_lane_segment_bounding_box_rotated():
    lane_segment = LaneSegment(pose=[0, 0, np.pi/2], lane_width=2.0, segment_length=1.0)
    bbox = lane_segment.bounding_box

    assert bbox.shape == (2, 4)
    assert_allclose(bbox[:, 0], (1.0, 0.5)), "Bottom left corner is not correct."
    assert_allclose(bbox[:, 1], (1.0, -0.5)), "Bottom right corner is not correct."
    assert_allclose(bbox[:, 2], (-1.0, -0.5)), "Top right corner is not correct."
    assert_allclose(bbox[:, 3], (-1.0, 0.5)), "Top left corner is not correct."


def test_lane_segment_bounding_box_translated():
    lane_segment = LaneSegment(pose=[1, 0, 0], lane_width=2.0, segment_length=1.0)
    bbox = lane_segment.bounding_box

    assert bbox.shape == (2, 4)
    assert_allclose(bbox[:, 0], (1.5, -1.0)), "Bottom left corner is not correct."
    assert_allclose(bbox[:, 1], (0.5, -1.0)), "Bottom right corner is not correct."
    assert_allclose(bbox[:, 2], (0.5, 1.0)), "Top right corner is not correct."
    assert_allclose(bbox[:, 3], (1.5, 1.0)), "Top left corner is not correct."