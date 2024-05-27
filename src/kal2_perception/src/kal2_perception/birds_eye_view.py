import numpy as np
import open3d as o3d
import cv2 as cv

from dataclasses import dataclass
from typing import Tuple


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


class BevTransformer:
    def __init__(self, homography: np.ndarray, target_size: Tuple[int, int]) -> None:
        self._homography = homography
        self._target_size = target_size

    def transform(self, image: np.ndarray) -> np.ndarray:
        return cv.warpPerspective(image, self.homography, self.target_size)

    @property
    def homography(self):
        return self._homography

    @property
    def target_size(self):
        return self._target_size
    
    @staticmethod
    def from_roi(roi: BevRoi, intrinsic_matrix: np.ndarray, extrinsic_matrix: np.ndarray, scale: int) -> np.ndarray:
        if intrinsic_matrix.shape != (3, 3):
            raise ValueError("Intrinsic matrix must be 3x3 matrix.")

        if extrinsic_matrix.shape != (3, 4) and extrinsic_matrix.shape != (4, 4):
            raise ValueError("Extrinsic matrix must be 3x4 or 4x4 matrix.")

        src_points = roi.as_pixel_coordinates(intrinsic_matrix, extrinsic_matrix).T.astype(np.float32)
        
        target_height = int((roi.max_distance - roi.min_distance) * scale)
        target_width = int(roi.width * scale)
        dst_points = np.array([[0, target_height], [0, 0], [target_width, 0], [target_width, target_height]], dtype=np.float32)

        homography = cv.getPerspectiveTransform(src=src_points, dst=dst_points)
        return BevTransformer(homography=homography, target_size=(target_width, target_height))