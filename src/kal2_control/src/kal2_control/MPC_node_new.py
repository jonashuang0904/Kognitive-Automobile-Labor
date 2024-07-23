from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
from kal2_control.MPCController import MPCController
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import TransformStamped
import rospy
import yaml
import numpy as np
import matplotlib.pyplot as plt
import tf2_ros


class MpcControllerNode:
    def __init__(self) -> None:
        rospy.init_node("mpc_controller_node")

        # Load the path from the yaml file
        file_path = rospy.get_param("~path_file", "/home/zikuang0904/kal2_new/kal2_ws/src/kal2_control/record/interpolation.yaml") 
        self._trajectory = self.load_path(file_path)
        rospy.loginfo(f"Loaded path from {file_path}")

        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)

        ##### Subscribers #####
        self._path_subscriber = rospy.Subscriber("/path", Path, self._trajectory_callback, queue_size=10)
        
        ##### Publishers #####
        self._speed_publisher = rospy.Publisher("/anicar/mux/ackermann_cmd_mux/input/path_follow", AckermannDriveStamped, queue_size=10)
        self._steering_angle_publisher = rospy.Publisher("/anicar/mux/ackermann_cmd_mux/input/path_follow", AckermannDriveStamped, queue_size=10)
        self._trajectory_publisher = rospy.Publisher("/predicted_trajectory", Path, queue_size=10)
        self._acc_publisher = rospy.Publisher("/anicar/mux/ackermann_cmd_mux/input/path_follow", AckermannDriveStamped, queue_size=10)
        
        # Timer callback for the control loop
        self._timer = rospy.Timer(rospy.Duration(0.1), self._timer_callback)

        # Initialize the controller
        self._mpc_controller = MPCController()
        self.current_index = 0
        

        # Initialize the trajectory and current pose
        self._trajectory = None
        self._current_pose = None
        rospy.loginfo("MpcControllerNode initialized with _trajectory and _current_pose set to None")

        # Actual position data
        self.time_data = []
        self.x_actual = []
        self.y_actual = []
        self.theta_actual = []
        self.v_actual = []
        
        # reference position data
        self.x_ref = []
        self.y_ref = []
        self.theta_ref = []

        # MPC predicted position data
        self.x_predicted = []
        self.y_predicted = []
        self.theta_predicted = []
        self.v_predicted = []
        self.delta_predicted = []
        self.acc_predicted = []

        self.start_time = rospy.Time.now().to_sec() # Get the current time in seconds
        rospy.on_shutdown(self.plot_results)
         
    # Load the path from the yaml file
    def load_path(self, file_path):       
        with open(file_path, 'r') as file:
            path_data = yaml.safe_load(file)

        return path_data
        
    # Callback function for the path subscriber
    def _trajectory_callback(self, path_msg):
        path_data = path_msg.poses
        self._trajectory = []
        for pose in path_data:
            x = pose.pose.position.x
            y = pose.pose.position.y
            orientation = pose.pose.orientation
            quaternion = [orientation.w, orientation.x, orientation.y, orientation.z]
            euler = euler_from_quaternion(quaternion)
            theta = euler[2]
            v = 0.5
            self._trajectory.append([x, y, theta, v])
        rospy.loginfo_once(f"Received trajectory data: {self._trajectory}")
        
    # Get the transform between stargazer and vehicle_rear_axle
    def get_transform(self, target_frame, source_frame, timeout: rospy.Duration = rospy.Duration(1.0)):

        transform = self.buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), timeout=timeout)
        return transform
        

    def find_closest_point(self):
        current_pose = np.array(self._current_pose[:2])
        trajectory_points = np.array([point[:2] for point in self._trajectory])
        
        distances = np.linalg.norm(trajectory_points - current_pose, axis=1)
        closest_index = np.argmin(distances)

        return closest_index, distances[closest_index]


    def _timer_callback(self, event):

        transform = self.get_transform("stargazer", "vehicle_rear_axle")

        if transform is not None:
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            quaternion = [transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]     
        
            _, _, theta = euler_from_quaternion(quaternion)
            self._current_pose = [x, y, theta]
        else:
            rospy.loginfo_once("Transform is None")
            return

        rospy.loginfo_once("Timer callback triggered")
        if self._trajectory is None:
            rospy.loginfo_once("Trajectory is None")
            return
        if self._current_pose is None:
            rospy.loginfo_once("Current pose is None")
            return
        if self._trajectory is None or self._current_pose is None:
            rospy.loginfo_once("Trajectory or current pose is None")
            return
        
        # Prepare the reference trajectory
        prediction_horizon = self._mpc_controller.prediction_horizon

        # Ensure that the indices are within the bounds of the trajectory
        self.current_index = (self.current_index + 1) % len(self._trajectory)
        indices = [(self.current_index + i) % len(self._trajectory) for i in range(prediction_horizon)]

        opt_states_now = np.array([self._trajectory[index] for index in indices])
        
        # Instead of calculating the yaw based on adjacent points, use a lookahead approach
        yaw = np.zeros(len(opt_states_now))
        for i in range(len(opt_states_now)):
            j = min(i + 3, len(opt_states_now) - 1)
            dx = opt_states_now[j, 0] - opt_states_now[i, 0]
            dy = opt_states_now[j, 1] - opt_states_now[i, 1]
            yaw = np.arctan2(dy, dx)
            opt_states_now[i, 2] = yaw
        opt_states_now[-1, 2] = opt_states_now[-2, 2]
        # Ensure that the yaw values are continuous
        opt_states_now[:, 2] = np.unwrap(opt_states_now[:, 2])
            

        # Compute the control inputs and state trajectory
        self._current_control_trajectory = self._mpc_controller.calculate_trajectory(opt_states_now)

        # Update the current index
        self.current_index += 1
        if self.current_index >= len(self._trajectory):
            self.current_index = 0

        # Publish the control message
        ackermann_msg = AckermannDriveStamped()
        ackermann_msg.header.stamp = rospy.Time.now()
        ackermann_msg.header.frame_id = "vehicle_rear_axle"
        ackermann_msg.drive.steering_angle = self._current_control_trajectory[0]
        ackermann_msg.drive.acceleration= self._current_control_trajectory[1]
        ackermann_msg.drive.speed = self._mpc_controller.next_states[1][3]
        self._steering_angle_publisher.publish(ackermann_msg)
        self._acc_publisher.publish(ackermann_msg)
        rospy.loginfo(f"Published control message: {ackermann_msg}")

        # Create and publish the predicted trajectory message
        predicted_trajectory_msg = Path()
        predicted_trajectory_msg.header.stamp = rospy.Time.now()
        predicted_trajectory_msg.header.frame_id = "stargazer"
        predicted_states = self._mpc_controller.next_states

        for state in predicted_states: 
            pose = PoseStamped()
            pose.pose.position.x = state[0]
            pose.pose.position.y = state[1]

            quaternion = quaternion_from_euler(0, 0, state[2])
            pose.pose.orientation.x = quaternion[0]
            pose.pose.orientation.y = quaternion[1]
            pose.pose.orientation.z = quaternion[2]
            pose.pose.orientation.w = quaternion[3]

            predicted_trajectory_msg.poses.append(pose)
        
        self._trajectory_publisher.publish(predicted_trajectory_msg)
        rospy.loginfo(f"Published predicted trajectory message: {predicted_trajectory_msg}")
        
        # Save the actual and predicted position data
        current_time = rospy.Time.now().to_sec() - self.start_time
        self.time_data.append(current_time)

        # Actual position
        self.x_actual.append(self._current_pose[0])
        self.y_actual.append(self._current_pose[1])
        self.theta_actual.append(self._current_pose[2])

        # reference position
        self.x_ref.append(opt_states_now[0])
        self.y_ref.append(opt_states_now[1])
        self.theta_ref.append(opt_states_now[2])

        # Predicted position
        mpc_predicted_states = self._mpc_controller.next_states[1] # predicted states for the first time step
        
        self.x_predicted.append(mpc_predicted_states[0])
        self.y_predicted.append(mpc_predicted_states[1])
        self.theta_predicted.append(mpc_predicted_states[2])
        self.v_predicted.append(mpc_predicted_states[3])

        self.delta_predicted.append(self._current_control_trajectory[0])
        self.acc_predicted.append(self._current_control_trajectory[1])
        print(f"self._mpc_controller.next_states: {self._mpc_controller.next_states}")

    def plot_results(self):

        fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True) 

        axs[0].plot(self.time_data, self.x_actual, label='Actual x')
        axs[0].plot(self.time_data, self.x_predicted, label='Predicted x')
        axs[0].plot(self.time_data, self.x_ref, label='Reference x')
        axs[0].set_ylabel('X position (m)')
        axs[0].legend()

        axs[1].plot(self.time_data, self.y_actual, label='Actual y')
        axs[1].plot(self.time_data, self.y_predicted, label='Predicted y')
        axs[1].plot(self.time_data, self.y_ref, label='Reference y')
        axs[1].set_ylabel('Y position (m)')
        axs[1].legend()

        axs[2].plot(self.time_data, self.theta_actual, label='Actual theta')
        axs[2].plot(self.time_data, self.theta_predicted, label='Predicted theta')
        axs[2].plot(self.time_data, self.theta_ref, label='Reference theta')
        axs[2].set_ylabel('Yaw (rad)')
        axs[2].legend()

        axs[3].plot(self.time_data, self.v_actual, label='Actual v')
        axs[3].plot(self.time_data, self.v_predicted, label='Predicted v')
        axs[3].plot(self.time_data, self.delta_predicted, label='predicted delta')
        axs[3].set_ylabel('Velocity (m/s)')
        axs[3].legend()

        plt.xlabel('Time (s)')
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    mpc_controller_node = MpcControllerNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    mpc_controller_node.plot_results()