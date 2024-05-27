import numpy as np
import cv2 as cv
import open3d as o3d
from dataclasses import dataclass

import seaborn as sns
import matplotlib.pyplot as plt

import rospy
from sensor_msgs.msg import Image

import message_filters
from cv_bridge import CvBridge

from kal2_perception.node_base import NodeBase


def line_kernel(width: int, rows: int):
    kernel = np.ones((rows, width * 3)) * 2
    kernel[:, :width] = -1
    kernel[:, -width:] = -1
    kernel /= np.sum(np.abs(kernel))

    return kernel


def create_homography_from_points(
    points: np.ndarray, intrinsic_matrix: np.ndarray, extrinsic_matrix: np.ndarray, scale: int
) -> np.ndarray:
    if points.shape != (3, 4):
        raise ValueError("Points must be 3x4 matrix.")

    if intrinsic_matrix.shape != (3, 3):
        raise ValueError("Intrinsic matrix must be 3x3 matrix.")

    if extrinsic_matrix.shape != (3, 4) and extrinsic_matrix.shape != (4, 4):
        raise ValueError("Extrinsic matrix must be 3x4 or 4x4 matrix.")

    camera_coords = extrinsic_matrix[:3] @ np.vstack([points, np.ones((1, points.shape[1]))])
    pixel_coords = intrinsic_matrix @ camera_coords[:, :]
    pixel_coords /= pixel_coords[2, :]

    target_height = 2000  # pixels
    target_width = 1500  # pixels
    scale = 500  # pixels per meter
    center = np.array([[target_width / 2, target_height / 2]], dtype=np.float32).T

    src_points = pixel_coords[:2, :].T.astype(np.float32)
    dst_points = center + points[:2] * scale
    dst_points = dst_points.T.astype(np.float32)

    print(dst_points)

    return cv.getPerspectiveTransform(src=src_points, dst=dst_points)

class GroundPlaneEstimator:
    def __init__(self, intrinsic_matrix: np.ndarray, extrinsic_matrix: np.ndarray) -> None:
        pass

    def estimate(self, *, point_cloud: o3d.geometry.PointCloud) -> np.ndarray:
        plane_model, _ = point_cloud.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        return plane_model


@dataclass
class BevRoi:
    min_distance: float
    max_distance: float
    width: float

    @property
    def points(self) -> np.ndarray:
        return np.array(
            [
                [-self.width / 2, 0, self.min_distance],
                [-self.width / 2, 0, self.max_distance],
                [self.width / 2, 0, self.max_distance],
                [self.width / 2, 0, self.min_distance],
            ]
        )
    
    @property
    def homogenous_coords(self) -> np.ndarray:
        return np.array(
            [
                [self.min_distance, -self.width / 2, 0, 1],
                [self.max_distance, -self.width / 2, 0, 1],
                [self.max_distance, self.width / 2, 0, 1],
                [self.min_distance,self.width / 2, 0, 1],
            ]
        )


class BevTransformer:
    def __init__(self, homography, target_size) -> None:
        self._H = homography
        self._target_size = target_size

    def transform(self, image: np.ndarray) -> np.ndarray:
        return cv.warpPerspective(image, self._H, self._target_size)
    
    @staticmethod
    def from_point_cloud(point_cloud: o3d.geometry.PointCloud, intrinsic_matrix: np.ndarray, roi: BevRoi, scale: int) -> np.ndarray:
        (a, b, c, d), _ = point_cloud.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)

        points = roi.points
        points[:, 1] = -(a * points[:, 0] + c * points[:, 2] + d) / b

        pixels = intrinsic_matrix @ points.T
        pixels = np.divide(pixels, pixels[2, :], out=pixels, where=pixels[2, :] != 0)
        src_points = pixels[:2].T.astype(np.float32)
        print(f"src_points: {src_points}")

        target_height = int((roi.max_distance - roi.min_distance) * scale)
        target_width = int(roi.width * scale)
        dst_points = np.array([[0, target_height], [0, 0], [target_width, 0], [target_width, target_height]], dtype=np.float32)
        
        H = cv.getPerspectiveTransform(src=src_points, dst=dst_points)
        return BevTransformer(H, target_size=(target_width, target_height))
    
    
    @staticmethod
    def from_roi(roi: BevRoi, intrinsic_matrix: np.ndarray, extrinsic_matrix: np.ndarray, scale: int) -> np.ndarray:
        if intrinsic_matrix.shape != (3, 3):
            raise ValueError("Intrinsic matrix must be 3x3 matrix.")

        if extrinsic_matrix.shape != (3, 4) and extrinsic_matrix.shape != (4, 4):
            raise ValueError("Extrinsic matrix must be 3x4 or 4x4 matrix.")
                
        points = roi.homogenous_coords.T
        print(f"points: {points}")
        print(f"extrinsic_matrix: {extrinsic_matrix}")

        camera_coords = np.linalg.inv(extrinsic_matrix) @ points #np.vstack([roi.points, np.ones((1, roi.points.shape[1]))])
        print(f"camera_coords: {camera_coords}")
        pixels = intrinsic_matrix @ camera_coords[:3]
        #pixels =  np.divide(pixels, pixels[2, :], out=pixels, where=pixels[2, :] != 0)
        src_points = pixels[:2].T.astype(np.float32)
        print(f"src_points: {src_points}")

        def as_cv_coords(points):
            return points[:2].T.astype(np.float32)[:, ::-1]
        
        #src_points = as_cv_coords(pixel_coords)

        target_height = int((roi.max_distance - roi.min_distance) * scale)
        target_width = int(roi.width * scale)
        dst_points = np.array([[0, target_height], [0, 0], [target_width, 0], [target_width, target_height]], dtype=np.float32)

        H = cv.getPerspectiveTransform(src=src_points, dst=dst_points)
        return BevTransformer(H, target_size=(target_width, target_height))
        


    @staticmethod
    def from_points(
        points: np.ndarray, intrinsic_matrix: np.ndarray, extrinsic_matrix: np.ndarray, scale: int
    ) -> np.ndarray:
        if points.shape != (3, 4):
            raise ValueError("Points must be 3x4 matrix.")

        if intrinsic_matrix.shape != (3, 3):
            raise ValueError("Intrinsic matrix must be 3x3 matrix.")

        if extrinsic_matrix.shape != (3, 4) and extrinsic_matrix.shape != (4, 4):
            raise ValueError("Extrinsic matrix must be 3x4 or 4x4 matrix.")

        camera_coords = extrinsic_matrix[:3] @ np.vstack([points, np.ones((1, points.shape[1]))])
        pixel_coords = intrinsic_matrix @ camera_coords[:, :]
        pixel_coords /= pixel_coords[2, :]

        target_height = 2000  # pixels
        target_width = 1500  # pixels
        scale = 500  # pixels per meter
        center = np.array([[target_width / 2, target_height / 2]], dtype=np.float32).T

        def as_cv_coords(points):
            return points[:2].T.astype(np.float32)[:, ::-1]
        
        src_points = as_cv_coords(pixel_coords)
        dst_points = as_cv_coords(center + points[:2] * scale)

        p0 = (28, 500)
        p3 = (756, 500)
        p2 = (697, 238)
        p1 = (564, 238)
        src_points = np.array([p0, p1, p2, p3], dtype=np.float32)
        dst_points = np.array([[target_width/3, target_height - 125], [target_width/3, 500], [target_width / 2, 500], [target_width / 2, target_height - 125]], dtype=np.float32)

        # src_points = pixel_coords[:2, :].T.astype(np.float32)
        # dst_points = center + points[:2] * scale
        # dst_points = dst_points.T.astype(np.float32)

        H = cv.getPerspectiveTransform(src=src_points, dst=dst_points)
        return BevTransformer(H)


class LaneDector:
    def __init__(self):
        self._intrinsic_matrix = np.array([[913.75, 0.0, 655.42], [0.0, 911.92, 350.49], [0.0, 0.0, 1.0]])
        self._extrinsic_matrix = np.array(
            [
                [-1.0, 0.005, -0.004, -0.0],
                [0.005, 0.168, -0.986, -0.236],
                [-0.005, -0.986, -0.168, -0.062],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self._region_of_interest = np.array([[0.0, 0.0, 4.0, 4.0], [-2.0, 2.0, 2.0, -2.0], [0.0, 0.0, 0.0, 0.0]])

        # self._bev_transformer = BevTransformer.from_points(
        #     self._region_of_interest, self._intrinsic_matrix, self._extrinsic_matrix, 500
        # )
        self._bev_transformer = None

        self._cv_bridge = CvBridge()
        self._publisher = rospy.Publisher("/debug/bev", Image, queue_size=10)
        self._publisher2 = rospy.Publisher("/debug/bev2", Image, queue_size=10)

    def detect(self, color_image: np.ndarray, depth_image: np.ndarray, timestamp: rospy.Time) -> np.ndarray:
        # rospy.loginfo("Detecting lanes...")
        # Implement lane detection here
        # bev_image = self._bev_transformer.transform(color_image)
                
        color = o3d.geometry.Image(color_image)
        depth = o3d.geometry.Image(depth_image)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=True)
        intrinsics = o3d.camera.PinholeCameraIntrinsic(1280, 720, fx=913.75, fy=911.92, cx=655.42, cy=350.49)
        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics, project_valid_depth_only=True)

        if not self._bev_transformer:
            self._bev_transformer = BevTransformer.from_point_cloud(point_cloud, intrinsics.intrinsic_matrix, roi=BevRoi(0.5, 3, 3), scale=240)
        # bev_transformer = BevTransformer.from_roi(roi=BevRoi(0.3, 3, 3), intrinsic_matrix=intrinsics.intrinsic_matrix, extrinsic_matrix=self._extrinsic_matrix, scale=250)

        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)

        # gray = gray.astype(np.float32)
        # gray = cv.filter2D(gray, -1, line_kernel(30, 3))
        # gray[gray < 0.0] = 0
        # gray = ((gray - gray.min()) / (gray.max() - gray.min())) * 255
        # gray = gray.astype(np.uint8)

        # bev_image = self._bev_transformer.transform(color_image)
        bev_image = cv.warpPerspective(gray, self._bev_transformer._H, self._bev_transformer._target_size, flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE, borderValue=gray.mean())
        
        bev_image = bev_image.astype(np.float32)
        filtered = cv.filter2D(bev_image, -1, line_kernel(5, 3)) * 2.0
        filtered[filtered < 0.0] = 0
        #filtered = ((filtered - filtered.min()) / (filtered.max() - filtered.min())) * 255

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        for i in range(6):
            cols = np.sum(filtered[100 * i: 100 * (i+1)], axis=0)
            ax.plot(cols)

        import io
        with io.BytesIO() as io_buf:
            fig.savefig(io_buf, format='raw')
            io_buf.seek(0)
            img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                                newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        plt.close()



        # color = o3d.geometry.Image(color_image)
        # depth = o3d.geometry.Image(depth_image)
        # rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_scale=1000.0, depth_trunc=100, convert_rgb_to_intensity=False)
        # intrinsics = o3d.camera.PinholeCameraIntrinsic(1280, 720, fx=913.75, fy=911.92, cx=655.42, cy=350.49)
        # point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics, project_valid_depth_only=False)
        # point_cloud.transform(self._extrinsic_matrix)

        # # ax = sns.scatterplot(x=np.asarray(point_cloud.points)[:,0], y=np.asarray(point_cloud.points)[:,1], s=1)
        # pcd = o3d.t.geometry.PointCloud.from_legacy(point_cloud)
        # intrinsic = o3d.core.Tensor(intrinsics.intrinsic_matrix)
        # extrinsics = o3d.core.Tensor(np.array([[1, 0, 0, 0],
        #                                         [0, -1, 0, 0],
        #                                         [0, 0, -1, 10],
        #                                         [0, 0, 0, 1]]))
        # rgbd_reproj = pcd.project_to_rgbd_image(1280, 720, intrinsic, extrinsics, depth_scale=500.0, depth_max=20.0)
        # img = (np.asarray(rgbd_reproj.color) * 255).astype(np.uint8)
        #img = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)
        #print(img.min(), img.max())

        self._publisher.publish(self._cv_bridge.cv2_to_imgmsg(filtered.astype(np.uint8), encoding="mono8"))
        self._publisher2.publish(self._cv_bridge.cv2_to_imgmsg(img_arr[:, :, :3], encoding="bgr8"))

        return None

class LaneDetectorNode(NodeBase):
    def __init__(self) -> None:
        super().__init__(name="lane_detector_node")

        self._color_image_subscriber = message_filters.Subscriber(self.params.color_image_topic, Image)
        self._depth_image_subscriber = message_filters.Subscriber(self.params.depth_image_topic, Image)

        self._image_subscriber = message_filters.TimeSynchronizer(
            [self._color_image_subscriber, self._depth_image_subscriber], queue_size=10
        )
        self._image_subscriber.registerCallback(self._image_callback)

        self._cv_bridge = CvBridge()
        self._lane_detector = LaneDector()

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def _image_callback(self, color_image_msg: Image, depth_image_msg: Image) -> None:
        color_image = self._cv_bridge.imgmsg_to_cv2(color_image_msg, desired_encoding="bgr8")
        depth_image = self._cv_bridge.imgmsg_to_cv2(depth_image_msg, desired_encoding="16UC1")

        path = self._lane_detector.detect(color_image, depth_image, color_image_msg.header.stamp)
