import cv2 as cv
import numpy as np

from abc import ABC, abstractmethod


class ImageOverlay:
    def __init__(self, intrinsic_matrix, extrinsic_matrix):
        self._intrinsic_matrix = intrinsic_matrix
        self._extrinsic_matrix = extrinsic_matrix

    def project_to_image(self, points: np.ndarray):
        if points.ndim != 2 or points.shape[0] != 3:
            raise ValueError(f"Points must be a 3xN matrix, got {points.shape}")
        
        tf_world_to_camera = self._extrinsic_matrix
        
        n_points = points.shape[1]
        camera_coords = tf_world_to_camera @ np.vstack([points, np.ones((1, n_points))])
        image_coords = self._intrinsic_matrix @ camera_coords[:3]
        pixel_coords  = (image_coords[:2, :] / image_coords[2, :]).astype(int)
        
        # return only points in front of the camera:
        return pixel_coords[:2, image_coords[2, :] > 0]
    
    def draw(self, image: np.ndarray, *args, **kwargs):
        if image.dtype != np.uint8:
            raise ValueError(f"Image must be of type uint8, got {image.dtype}")
        
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Image must be a 3-channel image, got {image.shape}")

        return self._do_draw(image, *args, **kwargs)

    @abstractmethod
    def _do_draw(self, image: np.ndarray):
        pass
    

class LaneFeatureOverlay(ImageOverlay):
    def __init__(self, intrinsic_matrix: np.ndarray, extrinsic_matrix: np.ndarray):
        super().__init__(intrinsic_matrix, extrinsic_matrix)

    def _do_draw(self, image: np.ndarray, features: np.ndarray) -> np.ndarray:
        n_features = features.shape[1]

        centers = np.vstack((features[:2], -0.23 * np.ones((1, n_features))))
        normals = np.vstack((features[2:], np.zeros((1, n_features))))

        centers_image = self.project_to_image(centers)
        normals_image = self.project_to_image(centers + 0.1 * normals)

        for center, normal in zip(centers_image.T, normals_image.T):
            cv.circle(image, tuple(center), 3, (0, 0, 255), -1)
            cv.line(image, tuple(center), tuple(normal), (0, 0, 255), 2)

        return image
    
class LaneSegmentOverlay(ImageOverlay):
    def __init__(self, intrinsic_matrix: np.ndarray, extrinsic_matrix: np.ndarray):
        super().__init__(intrinsic_matrix, extrinsic_matrix)

    def _do_draw(self, image: np.ndarray, lane_segments) -> np.ndarray:
        overlay = image.copy()

        for lane_segment in lane_segments:
            
            corners = np.vstack((lane_segment.bounding_box, -0.23 * np.ones((1, 4))))
            corners = self.project_to_image(corners).T # (n, 2)

            if corners.shape == (4, 2):
                BRIGHT_GREEN = (170, 255, 0)
                cv.line(overlay, tuple(corners[0]), tuple(corners[1]), color=BRIGHT_GREEN, thickness=2)
                cv.line(overlay, tuple(corners[1]), tuple(corners[2]), color=BRIGHT_GREEN, thickness=2)
                cv.line(overlay, tuple(corners[2]), tuple(corners[3]), color=BRIGHT_GREEN, thickness=2)
                cv.line(overlay, tuple(corners[3]), tuple(corners[0]), color=BRIGHT_GREEN, thickness=2)
                cv.fillPoly(overlay, [corners], color=BRIGHT_GREEN)
            else:
                continue

            if len(lane_segment.observations) == 0:
                continue

            features = np.array(lane_segment.observations)
            n_features = features.shape[1]

            centers = np.vstack((features[:2], -0.23 * np.ones((1, n_features))))
            centers_image = self.project_to_image(centers)

            for center in centers_image.T:
                try:
                    cv.circle(image, tuple(center), 3, (0, 255, 255), -1)
                except cv.error as e:
                    print(f"Error: {e}, center: {center}")

            predicted_path = lane_segment.predicted_path
            if predicted_path is not None:
                path =  np.vstack((predicted_path, -0.23 * np.ones((1, predicted_path.shape[1]))))
                path = self.project_to_image(path).astype(np.int32).T

                if path.shape[0] != 10:
                    continue

                cv.polylines(image, [path], isClosed=False, color=(255, 0, 0), thickness=2)

        if len(lane_segments) > 1:
            last_segment = lane_segments[-1]
            second_last_segment = lane_segments[-2]

            predicted_position = last_segment.position + last_segment.rotation_matrix @ np.array([[last_segment._segment_length], [0.0]])
            predicted_position = self.project_to_image(np.vstack((predicted_position, -0.23 * np.ones((1, 1))))).reshape(-1)

            if predicted_position.shape == (2,):
                cv.circle(image, (predicted_position[0], predicted_position[1]), 3, (255, 0, 0), -1)

            theta = last_segment.orientation - (last_segment.orientation - second_last_segment.orientation)
            R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

            predicted_position = (last_segment.position + R @ np.array([[last_segment._segment_length], [0]]))
            predicted_position = self.project_to_image(np.vstack((predicted_position, -0.23 * np.ones((1, 1))))).reshape(-1)

            if predicted_position.shape == (2,):
                cv.circle(image, (predicted_position[0], predicted_position[1]), 3, (255, 255, 0), -1)


        return cv.addWeighted(overlay, 0.5, image, 0.5, 0)
    
class LaneMapOverlay(ImageOverlay):
    def __init__(self, intrinsic_matrix: np.ndarray, extrinsic_matrix: np.ndarray):
        super().__init__(intrinsic_matrix, extrinsic_matrix)

    def _do_draw(self, image: np.ndarray, lane_map) -> np.ndarray:
        overlay = image.copy()

        if lane_map._center_line_points is not None:
            points = np.concatenate(lane_map._center_line_points).T

            path =  np.vstack((points, -0.23 * np.ones((1, points.shape[1]))))
            path = self.project_to_image(path).astype(np.int32)

            from kal2_perception.birds_eye_view import rasterize_points, VoxelConfig
            points = np.vstack((points, np.zeros((1, points.shape[1]))))
            grid = rasterize_points(points.T, VoxelConfig(x_min=0, x_max=15, y_min=-10, y_max=10))
            grid[grid != 0] = 255

            # for point in path:
            #     cv.circle(image, tuple(point), 3, (0, 0, 255), -1)

            # cv.polylines(image, [path], isClosed=False, color=(255, 0, 0), thickness=2)
        else:
            grid = np.zeros((10, 10), dtype=np.uint8)



        return grid.astype(np.uint8) #cv.addWeighted(overlay, 0.5, image, 0.5, 0)

    

class ProjectedErrorsOverlay(ImageOverlay):
    def __init__(self, intrinsic_matrix: np.ndarray, extrinsic_matrix: np.ndarray):
        super().__init__(intrinsic_matrix, extrinsic_matrix)

    def _do_draw(self, image: np.ndarray, lane_segments) -> np.ndarray:
        for lane_segment in lane_segments:
            if len(lane_segment._projected_observations) == 0:
                continue

            features = np.concatenate(lane_segment._projected_observations)
            
            print(features.shape)
            n_features = features.shape[1]

            centers = np.vstack((features[:2], -0.23 * np.ones((1, n_features))))
            centers_image = self.project_to_image(centers)

            for center in centers_image.T:
                cv.circle(image, tuple(center), 3, (0, 255, 255), -1)

        return image