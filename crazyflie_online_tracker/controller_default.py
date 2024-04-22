#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
import time
import yaml
import os
from scipy import linalg
from crazyflie_online_tracker_interfaces.msg import CommandOuter
from .controller import Controller, ControllerStates
from datetime import datetime
from crazyflie_online_tracker_interfaces.msg import ControllerState, CommandOuter, CrazyflieState, TargetState

import time
# Load data from the YAML file
yaml_path = os.path.join(os.path.dirname(__file__), '../param/data.yaml')

with open(yaml_path, 'r') as file:
    data = yaml.safe_load(file)

# Extract the required variableshover
m = data['m']
g = data['g']
f = data['f']
T = data['T']
target = data['target']
mode = data['mode']
delta_t = float(1/f)
filtering = data['filtering']


class DefaultController(Controller):
    '''
    Naive LQR (or simply LQR) controller
    '''
    def __init__(self):
        super().__init__()

        rclpy.init()
        self.node = rclpy.create_node("DefaultController")

        self.m = m

        self.controller_state_sub = self.node.create_subscription(ControllerState, 'controllerState', self.callback_controller_state, 10)
        self.controller_command_pub = self.node.create_publisher(CommandOuter, 'controllerCommand', 10)
        self.controller_state_pub = self.node.create_publisher(ControllerState, 'controllerStateKF', 10)

        self.drone_state_sub = self.node.create_subscription(CrazyflieState, 'crazyflieState', self.callback_state_drone, 10)
        self.target_state_sub = self.node.create_subscription(TargetState, 'targetState', self.callback_state_target, 10)

         # declare params
        self.node.declare_parameter('filename', 'Filename')

        # get params
        self.filename = self.node.get_parameter('filename')

         # timer calbacks
        timer_period = 0.5  # seconds
        self.timer = self.node.create_timer(timer_period, self.timer_callback)

        # Set to True to save data for post-processing
        self.save_log = True

        # Set to True to generate a plot immediately
        self.plot = False
        
        rclpy.spin(self.node)

    def timer_callback(self):

        if self.controller_state != ControllerStates.stop:
            ready = False
            if len(self.drone_state_raw_log) > 0 or len(self.filtered_drone_state_raw_log)>0:
                if len(self.target_state_raw_log) > 0:
                    ready = True
                if self.t <= T:
                    if ready:
                        t0 = time.time()
                        self.publish_setpoint()
                        t1 = time.time()
                        Time+= (t1-t0)
                        Time_T+= 1
                        self.node.get_logger().info('default controller published the setpoint')


                        self.t += self.delta_t
                        # if wait_for_simulator_initialization:
                        #     rospy.sleep(4)
                        #     count -= 1
                        #     if count < 0:
                        #         wait_for_simulator_initialization = False
                    else:
                        self.node.get_logger().info('No drone or target state estimation is available. Skipping.')
                else:
                    if self.controller_state == ControllerStates.normal:
                        self.publish_setpoint(is_last_command=True)
                        self.node.get_logger().info('Simulation finished.')

                    else:
                        self.publish_setpoint()


        if self.controller_state == ControllerStates.stop:
            self.node.get_logger().info('controller state is set to STOP. Terminating.')

            if self.save_log: # the simulation had started and has now been terminated

                additional_info = f"_{target}_T{T}_f{f}_mode{mode}"
                if filtering:
                    additional_info = f"_{target}_T{T}_f{f}_mode{mode}_Filtered"
                new_filename = self.filename.value + additional_info
                self.save_data(new_filename) # save log data to file for evaluation
                time.sleep(2)
                if self.plot:
                    self.node.get_logger().info('Printing the figures')
                    self.node.get_logger().info('Time: ' + str(Time/Time_T))
                    os.system("rosrun crazyflie_online_tracker plot.py")

    def compute_setpoint(self):
        drone_state = self.drone_state_raw_log[-1]
        target_state = self.target_state_raw_log[-1]

        if self.controller_state == ControllerStates.normal:
            # self.node.get_logger().info("controller state: normal")
            self.drone_state_log.append(drone_state)
            self.target_state_log.append(target_state)


            # self.node.get_logger().info("observe target[default]:" + str(target_state))
            # self.node.get_logger().info("current state[default]: " + str(drone_state))

            # option 1: rotated error + smoothed setpoint: NOT USED
            if self.last_setpoint is None:
                self.last_setpoint = drone_state[:3]
            desired_pos_limited = self.limit_pos_change(self.last_setpoint, target_state[:3])
            error_limited = drone_state - target_state
            error_limited[0] = drone_state[0] - desired_pos_limited[0]
            error_limited[1] = drone_state[1] - desired_pos_limited[1]
            error_limited[2] = drone_state[2] - desired_pos_limited[2]
            self.last_setpoint = desired_pos_limited
            # self.node.get_logger().info("setpoint limited:" + str(desired_pos_limited))
            [thrust, roll_rate, pitch_rate, yaw_rate] = self.compute_setpoint_viaLQR_controller(self.K_star, error_limited, drone_state[8])
            action_rotated_limited = np.array([thrust, pitch_rate, roll_rate, yaw_rate])

            # option 2: rotated error: USED
            error = drone_state - target_state
            [thrust, roll_rate, pitch_rate, yaw_rate] = self.compute_setpoint_viaLQR(self.K_star, error, drone_state[8])
            action_rotated = np.array([thrust, pitch_rate, roll_rate, yaw_rate])

            # option 3: naive LQR: NOT USED
            error = drone_state - target_state
            action_naive = -self.K_star@error
            action_naive[0] = action_naive[0] + self.m*self.g
            roll_rate = action_naive[1].copy()
            pitch_rate = action_naive[2].copy()
            action_naive[1] = pitch_rate
            action_naive[2] = roll_rate

            action = action_rotated # option 2
            self.action_log.append(action)
            setpoint = CommandOuter()
            setpoint.thrust = action[0]
            setpoint.omega.x = action[1] # pitch rate
            setpoint.omega.y = action[2] # roll rate
            setpoint.omega.z = action[3] # yaw rate
            # self.node.get_logger().info('error:'+str(action))

            disturbance_feedback = np.zeros((4,1))
            self.action_DF_log.append(disturbance_feedback)


        elif self.controller_state == ControllerStates.takeoff:
            self.node.get_logger().info("controller state: takeoff")
            setpoint = self.takeoff(drone_state)

            # self.node.get_logger().info('takeoff action:' + str(setpoint))
        elif self.controller_state == ControllerStates.landing:
            self.node.get_logger().info("controller state: landing")
            setpoint = self.landing(drone_state)
        self.setpoint = setpoint

        # compute disturbance w_t according to the latest drone and target state estimation
        if len(self.target_state_log) < 2:
            return
        else:
            last_target = self.target_state_log[-2]  # r_t
            curr_target = self.target_state_log[-1]  # r_{t+1}
        # w_t = Ar_t - r_{t+1}
        disturbance = self.A @ last_target - curr_target
        self.disturbances.append(disturbance)
    
    def publish_setpoint(self, is_last_command=False):
        if is_last_command:
            self.setpoint = CommandOuter()
            self.setpoint.is_last_command = True
        else:
            self.compute_setpoint()
        self.controller_command_pub.publish(self.setpoint)


def main(args=None):

    default_controller = DefaultController()


    # Set to True to save data for post-processing
    save_log = True

    # Used for timing the algorithms
    Time = 0
    Time_T = 0

    # # Set to True to generate a plot immediately
    # plot = True

    # while not (rospy.is_shutdown() or default_controller.controller_state == ControllerStates.stop):
    #     ready = False
    #     if len(default_controller.drone_state_raw_log) > 0 or len(default_controller.filtered_drone_state_raw_log)>0:
    #         if len(default_controller.target_state_raw_log) > 0:
    #             ready = True
    #         if default_controller.t <= T:
    #             if ready:
    #                 t0 = time.time()
    #                 default_controller.publish_setpoint()
    #                 t1 = time.time()
    #                 Time+= (t1-t0)
    #                 Time_T+= 1
    #                 # self.node.get_logger().info('default controller published the setpoint')
    #                 default_controller.t += default_controller.delta_t
    #                 if wait_for_simulator_initialization:
    #                     rospy.sleep(4)
    #                     count -= 1
    #                     if count < 0:
    #                         wait_for_simulator_initialization = False
    #             else:
    #                 self.node.get_logger().info('No drone or target state estimation is available. Skipping.')
    #         else:
    #             if default_controller.controller_state == ControllerStates.normal:
    #                 default_controller.publish_setpoint(is_last_command=True)
    #                 self.node.get_logger().info('Simulation finished.')
    #             else:
    #                 default_controller.publish_setpoint()
    #         rate.sleep()

    # if default_controller.controller_state == ControllerStates.stop:
    #     self.node.get_logger().info('controller state is set to STOP. Terminating.')
    #     if save_log: # the simulation had started and has now been terminated
    #         filename = rospy.get_param('filename')
    #         additional_info = f"_{target}_T{T}_f{f}_mode{mode}"
    #         if filtering:
    #             additional_info = f"_{target}_T{T}_f{f}_mode{mode}_Filtered"
    #         new_filename = filename + additional_info
    #         default_controller.save_data(new_filename) # save log data to file for evaluation
    #         time.sleep(2)
    #         if plot:
    #             self.node.get_logger().info('Printing the figures')
    #             self.node.get_logger().info('Time: ' + str(Time/Time_T))
    #             os.system("rosrun crazyflie_online_tracker plot.py")


if __name__ == '__main__':
    main()


    