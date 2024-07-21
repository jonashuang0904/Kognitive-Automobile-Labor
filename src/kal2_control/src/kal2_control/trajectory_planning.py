import time
from pathlib import Path
from enum import Enum
import yaml

import numpy as np

from scipy.interpolate import splprep, splev
from scipy.signal import savgol_filter

import rospy
from nav_msgs.msg import Path as NavMsgPath
from std_msgs.msg import Header
from geometry_msgs.msg import Quaternion, PoseStamped, Pose, Point

def filter_duplicates(points: np.ndarray):
    filtered_points = []
    for index in range(points.shape[1]-1):
        current_point = points[:, index + 1]
        last_point = points[:, index]

        if np.all(current_point == last_point):
            continue

        filtered_points.append(last_point)

    return np.vstack((filtered_points)).T


def calculate_trajectory(points, rotation, translation, n_samples: int = 100, v_min = 1.0, v_max = 2.0):
    points = filter_duplicates(np.array(points))

    tck, u = splprep(x=points, s=0.0068, k=3)
    spline_samples = splev(np.linspace(0, 1, num=n_samples), tck)

    df = splev(np.linspace(0, 1, n_samples), tck, der=1)
    ddf = splev(np.linspace(0, 1, n_samples), tck, der=2)

    dx = df[0]
    dy = df[1]
    ddx = ddf[0]
    ddy = ddf[1]
    angle = np.arctan2(df[1], df[0])

    n_overlap = 20
    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**(3/2)
    curvature = np.hstack((curvature[-n_overlap:], curvature, curvature[:n_overlap]))

    smooth_curvature = savgol_filter(curvature, 15, 3)
    smooth_curvature /= smooth_curvature.max()
    smooth_radius = 1 / (np.abs(smooth_curvature) + 10e-1)

    speed = v_min + (v_max - v_min) * (smooth_radius - smooth_radius.min()) / (smooth_radius.max() - smooth_radius.min())
    speed = savgol_filter(speed, 10, 2)
    normals = np.array([np.ones_like(dx), -dx/dy]) * np.sign(angle)
    normals /= np.linalg.norm(x=normals, axis=0)
    normals = rotation @ normals

    points = rotation @ np.array(spline_samples) + translation

    return points, normals, angle, speed[n_overlap:-n_overlap]


def create_nav_path(points, normals, angle, speed, offset = 0.25):
    path = NavMsgPath()
    stamp = rospy.Time.from_sec(time.time())
    header = Header(frame_id="stargazer", stamp=stamp)
    path.header = header

    for index in range(points.shape[1]):
        point = points[:, index] + normals[:, index] * offset
        rotation = Quaternion(x=0, y=0, z=np.sin(angle[index] / 2), w=np.cos(angle[index] / 2))

        point_msg = Point(x=point[0], y=point[1], z=speed[index])
        pose_stamped_msg = PoseStamped(header=header, pose=Pose(
            position=point_msg, orientation=rotation))
        path.poses.append(pose_stamped_msg)

    return path


def load_recorded_path(file_path: str) -> NavMsgPath:
    with Path(file_path).open() as file:
        path_dict = yaml.safe_load(file)

    path_msg = NavMsgPath()
    path_msg.header.seq = path_dict['header']['seq']
    path_msg.header.stamp.secs = path_dict['header']['stamp']['secs']
    path_msg.header.stamp.nsecs = path_dict['header']['stamp']['nsecs']
    path_msg.header.frame_id = path_dict['header']['frame_id']

    for pose_dict in path_dict['poses']:
        pose_stamped = PoseStamped()
        pose_stamped.header.seq = pose_dict['header']['seq']
        pose_stamped.header.stamp.secs = pose_dict['header']['stamp']['secs']
        pose_stamped.header.stamp.nsecs = pose_dict['header']['stamp']['nsecs']
        pose_stamped.header.frame_id = pose_dict['header']['frame_id']
        
        pose_stamped.pose.position = Point(
            pose_dict['pose']['position']['x'],
            pose_dict['pose']['position']['y'],
            pose_dict['pose']['position']['z']
        )
        
        pose_stamped.pose.orientation = Quaternion(
            pose_dict['pose']['orientation']['x'],
            pose_dict['pose']['orientation']['y'],
            pose_dict['pose']['orientation']['z'],
            pose_dict['pose']['orientation']['w']
        )
        
        path_msg.poses.append(pose_stamped)

    return path_msg


class LaneId(Enum):
    LEFT = 0.2
    RIGHT = -0.2
    CENTER = 0.0

def create_trajectory_from_path(path: NavMsgPath, lane_offset, theta: float = 0.00, map_offset: np.ndarray = np.zeros((2,1))):
    points = np.array([(pose.pose.position.x, pose.pose.position.y) for pose in path.poses]).T

    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    points, normals, angle, speed = calculate_trajectory(points, rotation=rotation, translation=map_offset)
    nav_path = create_nav_path(points, normals, angle, speed, offset=lane_offset)

    return nav_path