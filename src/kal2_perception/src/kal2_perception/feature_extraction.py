from abc import ABC, abstractmethod

import numpy as np
import cv2 as cv


class FeatureExtractor(ABC):
    def __init__(self) -> None:
        pass

    def extract(self, image: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8:
            raise ValueError(f"Image must be of type uint8, got {image.dtype}")

        return self._do_extract(image)

    @abstractmethod
    def _do_extract(self, image: np.ndarray) -> np.ndarray:
        pass


def flip_normals(normals: np.ndarray, direction: np.ndarray):
    dot_products = np.sum(normals * direction, axis=0)
    flipped_normals = np.where(dot_products < 0, -normals, normals)
    return flipped_normals


def transform_features(features: np.ndarray, intrinsic_matrix: np.ndarray, extrinsic_matrix: np.ndarray) -> np.ndarray:
    if features.ndim != 2 or features.shape[0] != 4:
        raise ValueError(f"Features must be a 4xN matrix, got {features.shape}")

    if intrinsic_matrix.shape != (3, 3):
        raise ValueError(f"Intrinsic matrix must be a 3x3 matrix, got {intrinsic_matrix.shape}")

    if extrinsic_matrix.shape != (4, 4):
        raise ValueError(f"Extrinsic matrix must be a 4x4 matrix, got {extrinsic_matrix.shape}")
    
    n_features = features.shape[1]
    
    def convert_image_to_camera_coordinates(image_points: np.ndarray) -> np.ndarray:
        """
        Convert image points to camera coordinates using the intrinsic matrix (of the BevTransformer). Note that the image
        coordinate system is rotated by 90 degrees compared to the camera coordinate system due to the BevTransformer.
        """
        image_points = np.vstack((image_points[1], image_points[0], np.ones((1, n_features))))
        return np.linalg.inv(intrinsic_matrix) @ image_points

    start_points = convert_image_to_camera_coordinates(features[:2])
    end_points = convert_image_to_camera_coordinates(features[2:])    

    centers = (start_points + end_points) / 2
    centers = (extrinsic_matrix @ np.vstack([centers, np.ones((1, n_features))]))

    diff = end_points - start_points
    normals = np.array([diff[1], -diff[0]], dtype=float) / np.linalg.norm(diff, axis=0)
    normals = flip_normals(normals, direction=np.array([[0], [1]]))
    normals = extrinsic_matrix[:3, :3] @ np.vstack((normals, np.zeros((1, n_features))))

    return np.vstack((centers[:2], normals[:2]))


class HoughFeatureExtractor(FeatureExtractor):
    def __init__(self) -> None:
        super().__init__()

        self._hough_params = {"rho": 1, "theta": np.pi / 180, "threshold": 10, "minLineLength": 10, "maxLineGap": 10}

    def _do_extract(self, image: np.ndarray) -> np.ndarray:
        lines = cv.HoughLinesP(image=image, **self._hough_params)
        lines = lines if lines is not None else []

        if len(lines) > 1:
            return np.array(lines).squeeze().T
        
        return np.array(lines).reshape(4, -1)