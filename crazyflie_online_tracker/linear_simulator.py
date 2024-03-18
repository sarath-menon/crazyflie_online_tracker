#!/usr/bin/env python3
# import rospy
import rclpy
from rclpy.node import Node

import struct
import numpy as np
import yaml
import os
from scipy.spatial.transform import Rotation
from scipy.linalg import solve_discrete_are
from crazyflie_online_tracker_interfaces.msg import CommandOuter, ControllerState, CrazyflieState
from .controller import ControllerStates
from .actuator import Actuator
from scipy.linalg import inv


# Load data from the YAML file
yaml_path = os.path.join(os.path.dirname(__file__), '../param/data.yaml')

with open(yaml_path, 'r') as file:
    yaml_data = yaml.safe_load(file)

# Extract the required variables
g = yaml_data['g']
m = yaml_data['m']
f = yaml_data['f']
T = yaml_data['T']

delta_t = float(1/f)


# Used for Filtering noisy state
filtering = yaml_data['filtering']

process_noise_var = yaml_data['process_noise_var']
measurement_noise_var = yaml_data['measurement_noise_var']
noise_std = np.sqrt(measurement_noise_var)

A_outer = np.array([[0, 0, 0,   1, 0, 0,    0, 0, 0],
                    [0, 0, 0,   0, 1, 0,    0, 0, 0],
                    [0, 0, 0,   0, 0, 1,    0, 0, 0],

                    [0, 0, 0,   0, 0, 0,    0, g, 0],
                    [0, 0, 0,   0, 0, 0,   -g, 0, 0],
                    [0, 0, 0,   0, 0, 0,    0, 0, 0],

                    [0, 0, 0,   0, 0, 0,    0, 0, 0],
                    [0, 0, 0,   0, 0, 0,    0, 0, 0],
                    [0, 0, 0,   0, 0, 0,    0, 0, 0]])
A = np.eye(9) + delta_t*A_outer
B_outer = np.array([[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],

                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                  [1/m, 0, 0, 0],
                
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
B = delta_t*B_outer

if filtering:
    H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0],

                  [0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0],

                  [0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1]])

    R = measurement_noise_var * np.eye(9)
    Q = process_noise_var * np.eye(9)

    # Check positive definiteness
    print("Q positive definite?", np.all(np.linalg.eigvals(Q) > 0))
    print("R positive definite?", np.all(np.linalg.eigvals(R) > 0))

    # Attempt to solve the DARE
    try:
        P_inf = solve_discrete_are(A.T, H.T, Q, R)
        print("DARE solution found.")
        # Calculate the SS-Kalman gain
        K_inf = P_inf @ H.T @ inv(H @ P_inf @ H.T + R)
    except np.linalg.LinAlgError as e:
        print(f"Error: {e}")


class CrazyflieActuator(Actuator):
    def __init__(self):
        super().__init__()
        self.curr_state = np.zeros((9, 1))

        # set initial state of the simulation
        self.curr_state[:3] = np.array([0.6, 0.0, 0.4]).reshape((3, 1))
        self.z_ss = np.array([0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape((9, 1))
        self.last_command_received = False

        # ros2 config
        node = rclpy.create_node('asdf')
        self.setpoint_sub = node.create_subscription(CommandOuter, "controllerCommand", self.callback_command, 10)
        self.controller_state_pub = node.create_publisher(ControllerState, 'controllerState', 10)
        self.drone_state_pub = node.create_publisher(CrazyflieState, 'crazyflieState', 10)
        self.filtered_drone_state_pub = node.create_publisher(CrazyflieState, 'FilteredCrazyflieState', 10)
        self.system_output_pub = node.create_publisher(CrazyflieState, 'SystemOutput', 10)


        controller_state = ControllerState()
        controller_state.state = ControllerStates.normal
        self.controller_state_pub.publish(controller_state)
        state_msg = self.state_vec_to_msg(self.curr_state)
        self.drone_state_pub.publish(state_msg)
        if filtering:
            # Initialize the estimated state and the action
            self.estimated_curr_state = self.curr_state
            action_init = np.zeros((4, 1))
            # Set up the measurement for the current state
            measurement = H @ self.curr_state
            # Add Gaussian noise to the measurement
            measurement += np.random.normal(0, noise_std, size=measurement.shape)
            # Publish the noisy measurement
            measurement_msg = self.state_vec_to_msg(measurement)
            self.system_output_pub.publish(measurement_msg)
            # Filter and publish filtered state
            filtered_curr_state = self.kalman_filtering(measurement, action_init, K_inf)
            filtered_curr_state_msg = self.state_vec_to_msg(filtered_curr_state)
            self.filtered_drone_state_pub.publish(filtered_curr_state_msg)

    def kalman_filtering(self, measurement, action, kalman_gain):
        # Prediction step
        predicted_next_state = A @ (self.estimated_curr_state - self.z_ss)+ B @ action
        # Update step
        estimated_next_state = predicted_next_state + kalman_gain @ (measurement - self.z_ss - H @ predicted_next_state)
        filtered_next_state = estimated_next_state + self.z_ss

        # Update estimated state for the next iteration
        self.estimated_curr_state = filtered_next_state

        return filtered_next_state

    def callback_command(self, data):
        self.get_logger().info('setpoint received')
        # # don't send any more command if the crazyflie has already landed.
        if data.is_last_command or self.last_command_received:
            print('last command received')
            controller_state = ControllerState()
            controller_state.state = ControllerStates.stop
            self.controller_state_pub.publish(controller_state)
            self.last_command_received = True
            return
        action = np.array([data.thrust - m*g, 
                 data.omega.y, 
                 data.omega.x, 
                 data.omega.z]).reshape((4, 1))
        next_state = A @ self.curr_state + B @ action
        next_state_msg = self.state_vec_to_msg(next_state)
        self.drone_state_pub.publish(next_state_msg)
        self.curr_state = next_state

        if filtering:
            # Set up the measurement
            measurement = H @ next_state
            # Add Gaussian noise to the measurement
            measurement += np.random.normal(0, noise_std, size=measurement.shape)
            # Publish the noisy measurement
            measurement_msg = self.state_vec_to_msg(measurement)
            self.system_output_pub.publish(measurement_msg)
            # Filter and publish filtered state
            filtered_next_state = self.kalman_filtering(measurement, action, K_inf)
            filtered_next_state_msg = self.state_vec_to_msg(filtered_next_state)
            self.filtered_drone_state_pub.publish(filtered_next_state_msg)

    def state_vec_to_msg(self, state_vector):
        state = CrazyflieState()
        state.pose.position.x = state_vector[0]
        state.pose.position.y = state_vector[1]
        state.pose.position.z = state_vector[2]
        r = Rotation.from_euler('ZYX', [state_vector[8][0], state_vector[7][0], state_vector[6][0]])
        quaternion = r.as_quat()
        state.pose.orientation.x = quaternion[0]
        state.pose.orientation.y = quaternion[1]
        state.pose.orientation.z = quaternion[2]
        state.pose.orientation.w = quaternion[3]
        state.velocity.linear.x = state_vector[3]
        state.velocity.linear.y = state_vector[4]
        state.velocity.linear.z = state_vector[5]
        return state


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = CrazyflieActuator()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()
    

if __name__ == '__main__':
    main()
