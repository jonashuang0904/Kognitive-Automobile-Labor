from typing import Callable

import numpy as np

import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge

import ros_numpy


def convert_images_to_arrays(*encodings: str):
    def decorator(func: Callable[..., None]) -> Callable[..., None]:
        cv_bridge = CvBridge()
        
        def wrapper(self, *image_msgs: Image):
            if len(encodings) != len(image_msgs):
                raise ValueError("Number of encodings must match the number of image messages")
            
            images = [cv_bridge.imgmsg_to_cv2(msg, encoding) for msg, encoding in zip(image_msgs, encodings)]
            return func(self, *images, image_msgs[0].header.stamp)

        return wrapper

    return decorator


def create_point_cloud_msg(points: np.ndarray, normals: np.ndarray):
    if points.shape != normals.shape:
        raise ValueError(f"Expected points and normals to have the same shape: {points.shape} != {normals.shape}.")

    if points.ndim != 2 or points.shape[0] != 2:
        raise ValueError(f"Expected input shape (2,N), got {points.shape}.")

    data = np.zeros(
        points.shape[1],
        dtype=[
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("normal_x", np.float32),
            ("normal_y", np.float32),
            ("normal_z", np.float32),
        ],
    )
    data['x'] = points[0]
    data['y'] = points[1]
    data['normal_x'] = normals[0]
    data['normal_y'] = normals[1]


    return ros_numpy.msgify(PointCloud2, data)
