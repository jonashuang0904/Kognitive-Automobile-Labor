import numpy as np
import cv2 as cv

import open3d as o3d

from filterpy.kalman import KalmanFilter
import filterpy.kalman as kf

kf.predict

from dataclasses import dataclass
from numba import njit

from time import time
from typing import List, Tuple

from scipy.spatial import KDTree

from itertools import tee, islice

def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def n_wise(iterable, n=2):
    if n < 1:
        raise ValueError("n must be at least 1")
    
    iterables = tee(iterable, n)
    for i, it in enumerate(iterables):
        next(islice(it, i, i), None)
    
    return zip(*iterables)


@dataclass
class VoxelConfig:
    resolution: int = 50
    x_min: float = 0
    x_max: float = 3.5
    y_min: float = -1.5
    y_max: float = 1.5
    max_height: float = 0.2

    @property
    def grid_size_x(self):
        return int((self.x_max - self.x_min) * self.resolution) + 1

    @property
    def grid_size_y(self):
        return int((self.y_max - self.y_min) * self.resolution) + 1


@njit
def populate_grid(grid: np.ndarray, indices: np.ndarray):
    grid_size_x, grid_size_y = grid.shape

    for i in range(len(indices)):
        x = indices[i, 0]
        y = indices[i, 1]

        if 0 <= x < grid_size_x or 0 <= y < grid_size_y:
            grid[x, y] += 1

    return grid


def voxelize_point_cloud(point_cloud: o3d.geometry.PointCloud, config: VoxelConfig = VoxelConfig()) -> np.ndarray:
    points = np.asarray(point_cloud.points)
    mask = (
        (points[:, 0] > config.x_min)
        & (points[:, 0] < config.x_max)
        & (points[:, 1] > config.y_min)
        & (points[:, 1] < config.y_max)
        & (points[:, 2] < config.max_height)
    )
    filtered_points = points[mask] - np.array([config.x_min, config.y_min, 0]).T

    indices = (filtered_points * config.resolution).astype(int)
    grid = np.zeros((config.grid_size_x, config.grid_size_y), dtype=int)

    return populate_grid(grid, indices)


class SlidingWindow:
    def __init__(self, x0: int, y0: int, window_height: int):
        self._window_height = window_height
        self._window_width = 40
        self._lane_width = 90

        # self._kf = KalmanFilter(dim_x=1, dim_z=1)
        self._x0 = x0
        self._y0 = y0

        self._x = x0
        self._P = 10

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y0

    def get_bounding_box(self, which: str = "left"):
        half_height = self._window_height // 2
        half_width = self._window_width // 2

        if which == "left":
            offset = self._lane_width // 2
        elif which == "right":
            offset = -self._lane_width // 2
        else:
            offset = 0

        x = int(self.x) + offset
        y = int(self.y)

        return ((x - half_width, y - half_height), (x + half_width, y + half_height))

    def predict(self, R, t):
        x = R[0, 0] * self._x + R[0, 1] * self._y0 + t[0]
        self._x, self._P = kf.predict(x=x, P=self._P, Q=10)

    def update(self, image):
        self._window_width = min(int(self._P), 40)

        def find_lane(which: str):
            (x0, y0), (x1, y1) = self.get_bounding_box(which=which)

            window = image[y0:y1, x0:x1]

            indices = np.argwhere(window > 0)

            if len(indices) < 10:
                return None, None

            return indices.mean(axis=0)[1], indices.var(axis=0)[1]

        mean_left, var_left = find_lane("left")
        mean_center, var_center = find_lane("center")
        mean_right, var_right = find_lane("right")

        means = np.array([mean_left, mean_center, mean_right])
        variances = np.array([var_left, var_center, var_right])
        offsets = np.array([-self._lane_width // 2, 0, self._lane_width // 2])

        detections = np.argwhere(means != None)

        if len(detections) == 0:
            return None

        if mean_left is not None:
            mean = means[0] + offsets[0]
            variance = variances[0]
        else:
            return None

        mean = np.sum(means[detections]) / len(detections)
        variance = np.sum(variances[detections])

        # if len(detections) == 1:
        #     mean = means[detections[0]]
        #     variance = variances[detections[0]]
        # elif len(detections) == 2:
        #     mean = (means[detections[0]] + means[detections[1]]) / 2
        #     variance = variances[detections[0]] + variances[detections[1]]
        # else:
        #     print("more than 2 detections")
        #     return None

        z = self._x + mean - self._window_width // 2
        self._x, self._P = kf.update(x=self._x, P=self._P, z=z, R=variance)

        return z


def create_sliding_windows(grid_size_x: int, grid_size_y: int, lane_width: int, number_of_windows: int):
    half_window_height = grid_size_y // (2 * number_of_windows - 2)
    window_height = 2 * half_window_height + 1
    x0 = 150  # (grid_size_x // 2) + (lane_width // 2)

    print(f"window hiehgt: {grid_size_y / number_of_windows} vs {window_height}")

    return [
        SlidingWindow(x0=x0, y0=half_window_height + i * window_height, window_height=window_height)
        for i in range(number_of_windows)
    ]


def draw_sliding_windows(image, sliding_windows, color=(0, 255, 0), which="left"):
    for window in sliding_windows:
        (x0, y0), (x1, y1) = window.get_bounding_box(which=which)

        if (
            0 <= x0 < image.shape[1]
            and 0 <= y0 < image.shape[0]
            and 0 <= x1 < image.shape[1]
            and 0 <= y1 < image.shape[0]
        ):
            cv.rectangle(img=image, pt1=(x0, y0), pt2=(x1, y1), color=color, thickness=1)
            try:
                image[int(window.y), int(window.x)] = color
            except IndexError:
                pass

    return image


def draw_lane_features(features, color_image: np.ndarray, intrinsic_matrix: np.ndarray, transform_world_to_camera: np.ndarray):
    for feature in features:
        center, normal, length = feature

        camera_coords = transform_world_to_camera @ np.vstack([center, -0.23, 1])
        image_coords_homogeneous = intrinsic_matrix @ camera_coords[:3]
        image_coords = image_coords_homogeneous[:2, :] / image_coords_homogeneous[2, :]
        image_coords = image_coords.T.astype(int)

        if np.any(image_coords_homogeneous[2] < 0):
            continue

        cv.circle(color_image, tuple(image_coords[0]), 3, (0, 0, 255), -1)

        tangent = np.array([normal[1], -normal[0]])
        start_point = center # - 0.5 * length * normal
        end_point = center + 0.1 * normal

        line = np.hstack([start_point, end_point])
        camera_coords = transform_world_to_camera @ np.vstack([line, -0.23 * np.ones((1, 2)), np.ones((1, 2))])
        image_coords_homogeneous = intrinsic_matrix @ camera_coords[:3, :]
        image_coords = image_coords_homogeneous[:2, :] / image_coords_homogeneous[2, :]
        image_coords = image_coords.T.astype(int)

        if np.any(image_coords_homogeneous[2] < 0):
            continue

        cv.line(color_image, tuple(image_coords[0]), tuple(image_coords[1]), (0, 0, 255), 2)

    return color_image


def draw_lane_segment(lane_segment, color_image: np.ndarray, intrinsic_matrix: np.ndarray, transform_world_to_camera: np.ndarray):
    overlay = color_image.copy()

    def to_homogeneous_coords(coords):
        return np.vstack([coords, -0.23 * np.ones((1, coords.shape[1]), dtype=float), np.ones((1, coords.shape[1]), dtype=float)])

    def to_image_coords(points):
        camera_coords = transform_world_to_camera @ points
        image_coords_homogeneous = intrinsic_matrix @ camera_coords[:3, :]
        image_coords = image_coords_homogeneous[:2, :] / image_coords_homogeneous[2, :]
        image_coords = image_coords.T.astype(int)

        return image_coords if np.all(image_coords_homogeneous[2, :] > 0) else None

    corners = to_image_coords(to_homogeneous_coords(lane_segment.bounding_box))

    if corners is not None:
        BRIGHT_GREEN = (170, 255, 0)
        cv.line(overlay, tuple(corners[0]), tuple(corners[1]), color=BRIGHT_GREEN, thickness=2)
        cv.line(overlay, tuple(corners[1]), tuple(corners[2]), color=BRIGHT_GREEN, thickness=2)
        cv.line(overlay, tuple(corners[2]), tuple(corners[3]), color=BRIGHT_GREEN, thickness=2)
        cv.line(overlay, tuple(corners[3]), tuple(corners[0]), color=BRIGHT_GREEN, thickness=2)
        cv.fillPoly(overlay, [corners], color=BRIGHT_GREEN)


    def draw_lane_features(lane_features, color):
        lane_points = np.array(lane_features).reshape(-1, 2).T
        lane_image_coords = to_image_coords(to_homogeneous_coords(lane_points))

        if lane_image_coords is not None:
            for i in range(1, len(lane_image_coords)):
                cv.circle(overlay, tuple(lane_image_coords[i]), 10, color, -1)

    BRIGHT_RED = (255, 87, 51)
    draw_lane_features(lane_segment._center_lane_features, color=BRIGHT_RED)
    draw_lane_features(lane_segment._left_lane_features, color=BRIGHT_RED)
    draw_lane_features(lane_segment._right_lane_features, color=BRIGHT_RED)

    def draw_search_area(lane: str, color):
        search_area = lane_segment.get_search_area(lane)
        corners = to_image_coords(to_homogeneous_coords(search_area))

        if corners is not None:
            cv.fillPoly(overlay, [corners], color=color)

    # draw_search_area("center", (255, 0, 0))
    # draw_search_area("right", (255, 0, 0))
    # draw_search_area("left", (255, 0, 0))

    return cv.addWeighted(overlay, 0.5, color_image, 0.5, 0)

def flip_normal(normal, direction):
    return -normal if np.dot(normal.T, direction) < 0 else normal


class LaneSegment:
    def __init__(self, pose: np.ndarray, lane_width: float, segment_length: float) -> None:
        self._pose = pose
        self._lane_width = lane_width
        self._segment_length = segment_length
        self._window_width = 0.15
        self._age = 0

        self._left_lane_features = []
        self._center_lane_features = []
        self._center_lane_normals = []
        self._right_lane_features = []
        self._features = []

        self._right_lane_estimate = np.array([[0], [0]])

    def save(self, filename):
        print(f"saving lane segment to {filename}")
        np.savez(filename, pose=self._pose, features=self._features)

    def predict(self, pose: np.ndarray):
        if self._age < 5:
            alpha = 0.01
        else:
            alpha = 0.95
        self._pose = alpha * self._pose + (1-alpha) * pose

    def observe(self, features):
        position = self.position
        R = self.rotation_matrix
        R_inv = np.linalg.inv(R)

        new_feature_count = 0

        for center, normal, _ in features:
            relative_difference = R_inv @ (center - position)

            if np.abs(relative_difference[0]) > self._segment_length or np.abs(relative_difference[1]) > self._lane_width:
                continue

            new_feature_count += 1

            y_axis = R @ np.array([[0], [1]])
            normal = flip_normal(normal, y_axis)

            self._features.append(np.array([center, normal]).reshape(-1))

            # from skspatial.objects import Vector
            # delta_theta = Vector(ey.squeeze()).angle_signed(Vector(normal.squeeze()))
            # self._pose[2] += delta_theta * 0.1

            # self._center_lane_features.append(center)

            if np.abs(relative_difference[1]) < self._window_width: #self._lane_width / 6:
                # self._pose[:2] = relative_feature[:2]
                # self._pose[1] -= relative_feature[1] * 0.1
                # theta = Vector(np.array([0, 1])).angle_signed(Vector(normal.squeeze()))
                # theta = np.arctan2(normal[0], normal[1])
                # self._pose[2] = theta
                # print(f"orientation: {self.orientation} - {theta} = {self.orientation - theta}")
                self._center_lane_features.append(center)
                self._center_lane_normals.append(normal)
                # print("center")

            #if self._lane_width / 4 < relative_difference[1] < 3 * self._lane_width / 4:
            if np.abs(relative_difference[1] - self._lane_width / 2) < self._window_width:
                # self._pose[1] -= relative_feature[1] * 0.1
                # self._pose[2] += delta_theta * 0.01
                # print("right")
                self._left_lane_features.append(center)

            # elif -3 * self._lane_width / 4 < relative_difference[1] < -self._lane_width / 4:
            elif np.abs(relative_difference[1] + self._lane_width / 2) < self._window_width:
                # self._pose[2] += delta_theta * 0.01
                # print("left")
                self._right_lane_features.append(center)

        if new_feature_count == 0:
            return
        
        self._age += 1

        if len(self._features) > 5:
            features = np.array(self._features)
            error = features[:, :2] - self.position.squeeze()
            scalar_error = np.dot(error, y_axis)
            error_center = scalar_error
            error_right = scalar_error - self._lane_width / 2
            error_left = scalar_error + self._lane_width / 2

            errors = np.array([error_center, error_right, error_left])
            min_abs_indices = np.argmin(np.abs(errors), axis=0)
            scalar_error = np.choose(min_abs_indices, errors)

            projected_error = np.median(scalar_error) * y_axis
            self._pose[:2] += 0.05 * projected_error.squeeze()

            mean_normal = np.median(features[:, 2:], axis=0)
            theta = np.arctan2(mean_normal[0], mean_normal[1])
            self._pose[2] = -theta


        # if len(self._center_lane_features) > 5:
        #     alpha = 0.5

        #     mean_position = np.median(np.array(self._center_lane_features), axis=0)
        #     error = mean_position - self.position
        #     projected_error = np.dot(error.T, y_axis) * y_axis
        #     self._pose[:2] += projected_error.squeeze()

        #     mean_normal = np.median(np.array(self._center_lane_normals), axis=0)
        #     theta = np.arctan2(mean_normal[0], mean_normal[1])
        #     self._pose[2] = -theta #alpha * self._pose[2] - (1-alpha) * theta

        # if len(self._right_lane_features) > 5:
        #     alpha = 0.5

        #     mean_position = np.mean(np.array(self._right_lane_features), axis=0)
        #     error = mean_position - self.position
        #     projected_error = (np.dot(error.T, y_axis) + self._lane_width / 2) * y_axis

        #     self._right_lane_estimate = self.position + projected_error
        #     # self._pose[:2] += projected_error.squeeze()
        #     # print(f"error: {np.dot(error.T, y_axis)}")


    @property
    def position(self):
        return np.array([[self._pose[0]], [self._pose[1]]])

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
        )

        position = self.position
        theta = self.orientation

        R = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])
        return (R @ corners.T) + position
    
    def get_search_area(self, lane: str):
        window_width = self._window_width
        half_lane_width = self._lane_width / 2

        if lane == "center":
            y_min = -window_width
            y_max = window_width
        elif lane == "right":
            y_min = half_lane_width - window_width
            y_max = half_lane_width + window_width
        elif lane == "left":
            y_min = -half_lane_width - window_width
            y_max = -half_lane_width + window_width
        else:
            raise ValueError(f"Unknown lane: {lane}")
        
        half_length = self._segment_length / 2
        corners = np.array(
            [
                [half_length, y_min],  # Bottom-left
                [-half_length, y_min],  # Bottom-right
                [-half_length, y_max],  # Top-right
                [half_length, y_max],  # Top-left
            ]
        )

        return (self.rotation_matrix @ corners.T) + self.position

class LaneMap:
    def __init__(self, initial_pose, lane_width: float = 0.9, segment_length: float = 0.25, look_ahead_distance: float = 2.0):
        self._lane_width = lane_width
        self._segment_length = segment_length
        self._look_ahead_distance = look_ahead_distance
        self._lane_segments: List[LaneSegment] = []

        self._add_lane_segment(np.array(initial_pose))

    def _add_lane_segment(self, pose: Tuple[float, float, float]):
        self._lane_segments.append(LaneSegment(pose, self._lane_width, self._segment_length))

    def update(self, vehicle_pose: np.ndarray, features):
        last_segment = self._lane_segments[-1]
        current_position = (np.linalg.inv(vehicle_pose) @ np.array([[0], [0], [0], [1]]))[:2]

        distance_to_last_segment = np.linalg.norm(current_position - last_segment.position)

        # print(f"distance to last segment: {distance_to_last_segment:.2f}")
        # print(f"current position: {current_position}")
        # print(f"last segment position: {last_segment.position.squeeze()}")

        if distance_to_last_segment < self._look_ahead_distance:
            theta = last_segment.orientation
            R = np.array([[np.cos(theta), -np.sin(theta)],  [np.sin(theta), np.cos(theta)]])
            position = (last_segment.position + R @ np.array([[self._segment_length], [0]])).squeeze()
            print(f"adding new segment at {position}")
            self._add_lane_segment(np.array([position[0], position[1], theta]))

        if len(features) == 0:
            return
        
        n = min(10, len(self._lane_segments))
        
        for current_seg, next_seg in pairwise(self._lane_segments[-n:]):
            R = current_seg.rotation_matrix
            position = (current_seg.position + R @ np.array([[current_seg._segment_length], [0]])).squeeze()
            next_seg.predict(np.array([position[0], position[1], current_seg.orientation]))
        
        for segment in self._lane_segments[-n:]:
            segment.observe(features)

        # if len(self._lane_segments) > 40:
        #     for index, segment in enumerate(self._lane_segments):
        #         segment.save(f"/home/josua/kal2_ws/segments/lane_segment_{index:02d}.npz")
        #     exit(0)
        


    def draw(self, color_image: np.ndarray, intrinsic_matrix: np.ndarray, transform_world_to_camera: np.ndarray):
        n = min(10, len(self._lane_segments))

        for lane_segment in self._lane_segments[-n:]:
            color_image = draw_lane_segment(lane_segment, color_image, intrinsic_matrix, transform_world_to_camera)

        return color_image


def compute_features_hough(point_cloud, transform_vehicle_to_world, color_image=None):
    config = VoxelConfig(resolution=100)
    voxel_grid = voxelize_point_cloud(point_cloud, config=config)
    voxel_grid[voxel_grid > 0] = 255

    lines = cv.HoughLinesP(voxel_grid.astype(np.uint8), 1, np.pi / 180, threshold=10, minLineLength=10, maxLineGap=10)

    voxel_grid_color = cv.cvtColor(voxel_grid.astype(np.uint8), cv.COLOR_GRAY2BGR) 

    features = []
    
    if lines is not None:
        for line in lines:
            y1, x1, y2, x2 = line[0]
            cv.line(voxel_grid_color, (int(y1), int(x1)), (int(y2), int(x2)), (0, 0, 255), 1)

            x1 = x1 / config.resolution + config.x_min
            x2 = x2 / config.resolution + config.x_min
            y1 = y1 / config.resolution + config.y_min
            y2 = y2 / config.resolution + config.y_min

            dx = x2 - x1
            dy = y2 - y1
            normal = np.array([[dy], [-dx]], dtype=float)

            # normal = np.array([[y2 - y1], [x1 - x2]], dtype=float)
            normal /= np.linalg.norm(normal)
            normal = (transform_vehicle_to_world[:3, :3] @ np.vstack([normal, 0]))[:2]
            # print(transform_vehicle_to_world @ np.vstack([normal, 0, 1]))

            center = np.array([[(x1 + x2) / 2], [(y1 + y2) / 2]], dtype=float)
            center = (transform_vehicle_to_world @ np.vstack([center, 0, 1]))[:2]

            length = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))

            features.append((center, normal, length))


    # cv.imshow("voxel_grid", voxel_grid_color)
    # cv.waitKey(1)

    return features

def compute_features_hough_image(binary_image, transform_vehicle_to_world):
    from kal2_perception.birds_eye_view import BevTransformer, BevRoi
    from skimage.morphology import skeletonize, remove_small_objects

    t0 = time()

    extrinsic_matrix = np.array(
            [
                [-0.00471339, -0.99997761, -0.00474976, 0.00151022],
                [-0.16772034, 0.00547306, -0.98581942, 0.158225],
                [0.98582335, -0.00384992, -0.16774238, -0.21260152],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    intrinsic_matrix = np.array([[913.75, 0.0, 655.42], [0.0, 911.92, 350.49], [0.0, 0.0, 1.0]])
    region_of_interest = BevRoi(min_distance=0.5, max_distance=2.5, width=1.5)
    scale = 240
    bev_transformer = BevTransformer.from_roi(
            region_of_interest, intrinsic_matrix, extrinsic_matrix, scale=scale
    )

    bev_image = bev_transformer.transform(binary_image)
    bev_image = skeletonize(bev_image).astype(np.uint8) * 255

    lines = cv.HoughLinesP(bev_image, 1, np.pi / 180, threshold=10, minLineLength=25, maxLineGap=15)
    bev_image = cv.cvtColor(bev_image, cv.COLOR_GRAY2BGR)

    features = []

    def to_vehicle_coords(u: int, v: int):
        distance = (region_of_interest.max_distance - region_of_interest.min_distance)
        target_height = int(distance * scale)
        target_width = int(region_of_interest.width * scale)

        x = (v / target_height) * distance + region_of_interest.min_distance
        y = (u / target_width) * region_of_interest.width - region_of_interest.width / 2

        return transform_vehicle_to_world @ np.array([[x], [y], [0], [1]])
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(bev_image, (x1, y1), (x2, y2), (0, 0, 255), 1)

            center = to_vehicle_coords(u=(x1 + x2) / 2, v=(y1 + y2) / 2)[:2]

            normal = np.array([[y2 - y1], [x1 - x2]], dtype=float)
            normal /= np.linalg.norm(normal)
            normal = (transform_vehicle_to_world[:3, :3] @ np.vstack([normal, 0]))[:2]

            features.append((center, normal, 0))

            # H_inv = np.linalg.inv(bev_transformer.homography)
            # p0 = H_inv @ np.array([x1, y1, 1]).reshape(-1, 1)
            # p1 = H_inv @ np.array([x2, y2, 1]).reshape(-1, 1)

            # p0 /= p0[2]
            # p1 /= p1[2]

            # print(p0, p1)

            # x1 = x1 / config.resolution + config.x_min
            # x2 = x2 / config.resolution + config.x_min
            # y1 = y1 / config.resolution + config.y_min
            # y2 = y2 / config.resolution + config.y_min

            # dx = x2 - x1
            # dy = y2 - y1
            # normal = np.array([[dy], [-dx]], dtype=float)

            # # normal = np.array([[y2 - y1], [x1 - x2]], dtype=float)
            # normal /= np.linalg.norm(normal)
            # normal = (transform_vehicle_to_world[:3, :3] @ np.vstack([normal, 0]))[:2]
            # # print(transform_vehicle_to_world @ np.vstack([normal, 0, 1]))

            # center = np.array([[(x1 + x2) / 2], [(y1 + y2) / 2]], dtype=float)
            # center = (transform_vehicle_to_world @ np.vstack([center, 0, 1]))[:2]

            # length = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))

            # features.append((center, normal, length))

    t1 = time()
    print(f"query time: {t1 - t0:.2f}")

    # cv.imshow("voxel_grid", bev_image)
    # cv.waitKey(1)

    return features

def compute_features_local(point_cloud, transform_vehicle_to_world):
    t0 = time()
    sampled = point_cloud.voxel_down_sample(voxel_size=0.05)
    points = np.asarray(sampled.points)[:,:2]
    
    kdtree = KDTree(points)
    res = kdtree.query_ball_point(points, r=0.1)

    for indices in res:
        neighbors = points[indices]
        center = np.mean(neighbors, axis=0)
        cov = np.cov(neighbors.T)

        # eigvals, eigvecs = np.linalg.eig(cov)
        # idx = np.argsort(eigvals)
        # eigvals, eigvecs= eigvals[idx], eigvecs[:, idx]

        print(cov)
        

    print(f"query time: {time() - t0:.2f}")


    return None


class LaneTracker:
    def __init__(self, initial_pose: np.ndarray, lane_width_meters: float = 0.9):
        self._voxel_config = VoxelConfig(resolution=100)
        self._lane_width_pixels = int(lane_width_meters * self._voxel_config.resolution)

        self._sliding_windows_left = create_sliding_windows(
            grid_size_x=self._voxel_config.grid_size_x,
            grid_size_y=self._voxel_config.grid_size_y,
            lane_width=self._lane_width_pixels,
            number_of_windows=15,
        )

        print("grid_size_x", self._voxel_config.grid_size_x)
        print("grid_size_y", self._voxel_config.grid_size_y)
        print("lane_width_pixels", self._lane_width_pixels)
        print("window height", self._sliding_windows_left[0]._window_height)

        self.transform_vehicle_to_camera = None
        self.camera_intrinsics = None

        self._last_vehicle_pose = None
        self._lane_map = LaneMap(initial_pose=(3, 0.5, 0))

    def initialize(self, initial_pose, transform_vehicle_to_camera, camera_intrinsics):
        self.transform_vehicle_to_camera = transform_vehicle_to_camera
        self.camera_intrinsics = camera_intrinsics
        self._last_vehicle_pose = initial_pose

        # self._add_new_landmark(vehicle_pose=initial_pose, distance=0)

        # for _ in range(4):
        #     self._add_new_landmark(vehicle_pose=initial_pose, distance=0.5)

    def is_initialized(self):
        return self.transform_vehicle_to_camera is not None and self.camera_intrinsics is not None
    
    def update_with_image(self, color_image, binary_image, transform_world_to_vehicle, transform_world_to_camera) -> np.ndarray:
        if not self.is_initialized():
            return color_image
        
        # compute_features_local(point_cloud, transform_vehicle_to_world=np.linalg.inv(transform_world_to_vehicle))
        features = compute_features_hough_image(binary_image, transform_vehicle_to_world=np.linalg.inv(transform_world_to_vehicle))
        color_image = draw_lane_features(features, color_image, self.camera_intrinsics, transform_world_to_camera)

        self._lane_map.update(transform_world_to_vehicle, features=features)

        # transform_world_to_camera =  self.transform_vehicle_to_camera @ vehicle_pose
        return self._lane_map.draw(color_image, self.camera_intrinsics, transform_world_to_camera)

    def update(self, color_image, point_cloud: o3d.geometry.PointCloud, transform_world_to_vehicle, transform_world_to_camera) -> np.ndarray:
        if not self.is_initialized():
            return color_image
        
        # compute_features_local(point_cloud, transform_vehicle_to_world=np.linalg.inv(transform_world_to_vehicle))
        features = compute_features_hough(point_cloud, transform_vehicle_to_world=np.linalg.inv(transform_world_to_vehicle))
        color_image = draw_lane_features(features, color_image, self.camera_intrinsics, transform_world_to_camera)

        self._lane_map.update(transform_world_to_vehicle, features=features)

        # transform_world_to_camera =  self.transform_vehicle_to_camera @ vehicle_pose
        return self._lane_map.draw(color_image, self.camera_intrinsics, transform_world_to_camera)
        return #draw_lane_segment(segement, color_image, self.camera_intrinsics, transform_world_to_camera)

        voxel_grid = voxelize_point_cloud(point_cloud, config=VoxelConfig(resolution=50))  # ~1-2ms
        voxel_grid[voxel_grid > 0] = 255

        diff = vehicle_pose @ np.linalg.inv(self._last_vehicle_pose)
        R = diff[:2, :2]
        t = diff[:2, 3]

        voxel_grid_color = cv.cvtColor(voxel_grid.astype(np.uint8), cv.COLOR_GRAY2BGR)
        voxel_grid_color = draw_sliding_windows(voxel_grid_color, self._sliding_windows_left, which="center")
        voxel_grid_color = draw_sliding_windows(voxel_grid_color, self._sliding_windows_left, which="right")
        voxel_grid_color = draw_sliding_windows(voxel_grid_color, self._sliding_windows_left, which="left")

        lines = cv.HoughLinesP(voxel_grid.astype(np.uint8), 1, np.pi / 180, threshold=10, minLineLength=10, maxLineGap=10)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv.line(voxel_grid_color, (x1, y1), (x2, y2), 255, 1)

        # line = np.array([xs, v_k]).T.astype(np.int32)
        # print(line)

        # cv.polylines(voxel_grid_color, [line], isClosed=False, color=(255, 255, 0), thickness=1)

        # cv.polylines(color_image, [x_k.T.astype(np.int32)], isClosed=False, color=(0, 255, 0), thickness=1)

        for window in self._sliding_windows_left:
            window.predict(np.linalg.inv(R), t)
            z = window.update(voxel_grid)

            if z is not None and 0 < z < voxel_grid_color.shape[1]:
                print(f"z: {z}")
                cv.circle(voxel_grid_color, (int(z), int(window.y)), 3, (0, 0, 255), -1)

        # for u, v in zip(x_k[0], y):
        #     u = int(u)
        #     v = int(v)

        #     if 0 <= v < voxel_grid.shape[0] and 0 <= u < voxel_grid.shape[1]:
        #         voxel_grid[v, u] = (255, 0, 0)

        self._last_vehicle_pose = vehicle_pose

        return voxel_grid_color
