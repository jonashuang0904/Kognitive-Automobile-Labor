import numpy as np
import cv2 as cv
import open3d as o3d
from open3d.geometry import Image, PointCloud, RGBDImage

from dataclasses import dataclass
from typing import Tuple
from numba import njit

from kal2_perception.camera import CameraInfo

@dataclass
class BevRoi:
    min_distance: float
    max_distance: float
    width: float
    z_offset: float = 0

    def as_vehicle_coordinates(self) -> np.ndarray:
        half_width = self.width / 2
        return np.array([[self.min_distance, self.max_distance, self.max_distance, self.min_distance], [half_width, half_width, -half_width, -half_width], 4*[self.z_offset]])
    
    def as_camera_coordinates(self, extrinsic_matrix: np.ndarray) -> np.ndarray:
        points = np.vstack([self.as_vehicle_coordinates(), np.ones((1, 4))])
        return (extrinsic_matrix @ points)[:3]
    
    def as_pixel_coordinates(self, intrinsic_matrix: np.ndarray, extrinsic_matrix: np.ndarray) -> np.ndarray:
        camera_coords = self.as_camera_coordinates(extrinsic_matrix)
        pixels = intrinsic_matrix @ camera_coords
        pixels /= pixels[2, :]
        return pixels[:2]


class PerspectiveBevTransformer:
    def __init__(self, homography: np.ndarray, target_size: Tuple[int, int]) -> None:
        self._homography = homography
        self._target_size = target_size

    def transform(self, image: np.ndarray, border_mode = cv.BORDER_CONSTANT) -> np.ndarray:
        return cv.warpPerspective(image, self.homography, self.target_size, borderMode=border_mode)

    @property
    def homography(self):
        return self._homography

    @property
    def target_size(self):
        return self._target_size
    
    @staticmethod
    def from_roi(roi: BevRoi, intrinsic_matrix: np.ndarray, extrinsic_matrix: np.ndarray, scale: int, flip_vertical: bool = False) -> np.ndarray:
        if intrinsic_matrix.shape != (3, 3):
            raise ValueError("Intrinsic matrix must be 3x3 matrix.")

        if extrinsic_matrix.shape != (3, 4) and extrinsic_matrix.shape != (4, 4):
            raise ValueError("Extrinsic matrix must be 3x4 or 4x4 matrix.")

        src_points = roi.as_pixel_coordinates(intrinsic_matrix, extrinsic_matrix).T.astype(np.float32)
        
        target_height = int((roi.max_distance - roi.min_distance) * scale)
        target_width = int(roi.width * scale)

        if flip_vertical:
            dst_points = np.array([[0, target_height], [0, 0], [target_width, 0], [target_width, target_height]], dtype=np.float32)
        else:
            dst_points = np.array([[target_width, 0], [target_width, target_height], [0, target_height], [0, 0]], dtype=np.float32)

        homography = cv.getPerspectiveTransform(src=src_points, dst=dst_points)
        return PerspectiveBevTransformer(homography=homography, target_size=(target_width, target_height))
    
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
def _populate_grid(grid: np.ndarray, indices: np.ndarray):
    grid_size_x, grid_size_y = grid.shape

    for i in range(len(indices)):
        x = indices[i, 0]
        y = indices[i, 1]

        if 0 <= x < grid_size_x or 0 <= y < grid_size_y:
            grid[x, y] += 1

    return grid

def rasterize_points(points: np.ndarray, config: VoxelConfig) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected (N,3), got {points.shape}.")

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

    return _populate_grid(grid, indices)

def camera_info_to_o3d_intrinsics(camera_info: CameraInfo) -> o3d.camera.PinholeCameraIntrinsic:
    height, width = camera_info.image_size
    fx = camera_info.intrinsic_matrix[0, 0]
    fy = camera_info.intrinsic_matrix[1, 1]
    cx = camera_info.intrinsic_matrix[0, 2]
    cy = camera_info.intrinsic_matrix[1, 2]
    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

class PointcloudBevTransformer:
    def __init__(self, camera_info: CameraInfo, extrinsic_matrix: np.ndarray) -> None:
        self._config = VoxelConfig(resolution=100, x_max=2.0)
        self._depth_scale = 1000.0
        self._max_depth = 3.5
        self._intrinsics = camera_info_to_o3d_intrinsics(camera_info)
        self._extrinsic_matrix = extrinsic_matrix

    @property
    def intrinsic_matrix(self):
        f = self._config.resolution
        x_min, y_min = self._config.x_min, self._config.y_min
        return np.array([[f, 0, -f*x_min], [0, f, -f*y_min], [0, 0, 1]])


    def _rasterize_point_cloud(self, point_cloud: o3d.geometry.PointCloud) -> np.ndarray:
        return rasterize_points(np.asarray(point_cloud.points), self._config)
    
    def transform(self, color_image: np.ndarray, depth_image: np.ndarray) -> np.ndarray:
        color, depth = Image(color_image), Image(depth_image)

        rgbd_image = RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=self._depth_scale, depth_trunc=self._max_depth
        )

        point_cloud = PointCloud.create_from_rgbd_image(rgbd_image, self._intrinsics)
        point_cloud.transform(self._extrinsic_matrix)

        raster = self._rasterize_point_cloud(point_cloud)
        raster[raster > 0] = 255

        return raster.astype(np.uint8)
