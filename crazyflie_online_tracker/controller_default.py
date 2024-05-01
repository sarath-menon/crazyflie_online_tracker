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
from geometry_msgs.msg import Twist
import signal

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

        # self.setpoint_publisher = self.node.create_publisher(Twist, '/cf231/cmd_vel_legacy', 10)

         # declare params
        self.node.declare_parameter('filename', 'Filename')

        # get params
        self.filename = self.node.get_parameter('filename')

         # timer calbacks
        self.timer = self.node.create_timer(self.delta_t, self.timer_callback)

        # Set to True to save data for post-processing
        self.save_log = True

        # Set to True to generate a plot immediately
        self.plot = True


        # for timing in algorithms
        self.Time = 0        # print(self.backend.time())
        self.Time_T = 0

        signal.signal(signal.SIGINT, self.exit_handler)

        # takeoff drone
        self.initial_position = np.array([0, 0, 0])
        self.hover_position = np.array([0, 0, 0.3])

        self.set_to_manual_mode()
        time.sleep(4)

        #self.takeoff_autonomous()
        self.takeoff_manual()

        self.drone_ready = False

        rclpy.spin(self.node)        # print(self.backend.time())

    
    def exit_handler(self, signum, frame):
        print("Sending land command")
        self.land()
        exit(1)

    def check_drone_at_position(self, pos=np.array([0, 0, 0])):
        drone_state = self.drone_state_raw_log[-1]
        x, y, z, vx, vy, vz, yaw, yaw_rate, thrust = drone_state

        tol_x = 0.1
        tol_y = 0.1
        tol_z = 0.1

        position_diff = np.abs([x - pos[0], y - pos[1], z - pos[2]])
        x_diff, y_diff, z_diff = position_diff

        self.node.get_logger().info(f"x: {x_diff}, y: {y_diff}, z: {z_diff}")

        if np.all(position_diff < np.array([tol_x, tol_y, tol_z])):
            return True
        else:
            return False

    def timer_callback(self):

        # self.node.get_logger().info(f"Length of drone state log: {len(self.drone_state_raw_log)}")
        # self.node.get_logger().info(f"Length of target state log: {len(self.target_state_raw_log)}") 

        self.node.get_logger().info(f"Controller state: {self.controller_state}")

        if len(self.drone_state_raw_log) == 0 or len(self.target_state_raw_log) == 0:
            self.node.get_logger().info('No drone or target state estimation is available. Skipping.')
            return

            # if self.save_log: # the simulation had started and has now been terminated

            #     additional_info = f"_{target}_T{T}_f{f}_mode{mode}"
            #     if filtering:
            #         additional_info = f"_{target}_T{T}_f{f}_mode{mode}_Filtered"
            #     new_filename = self.filename.value + additional_info
            #     self.save_data(new_filename) # save log data to file for evaluation
            #     time.sleep(2)
            #     if self.plot:
            #         self.node.get_logger().info('Printing the figures')
            #         os.system("python3 ../crazyflie_online_tracker/plot.py")

            # exit()


        if self.controller_state == ControllerStates.takeoff:
            self.setpoint = self.takeoff()
            self.controller_command_pub.publish(self.setpoint)

        elif self.controller_state == ControllerStates.idle:
                self.node.get_logger().info("Drone landed")
                exit()

        elif self.controller_state == ControllerStates.landing: 
            if self.check_drone_at_position(pos=self.hover_position) == False:
                    self.go_to_position(self.hover_position)
                    self.node.get_logger().info("Going to hover position before landing")

            else:
                # self.setpoint = self.landing()
                # self.controller_command_pub.publish(self.setpoint)
                self.land_autonomous()
                self.node.get_logger().info("Landing started")

            
        elif self.controller_state == ControllerStates.flight:
            if self.t >= T:
                self.node.get_logger().info('Simulation finished.')
                self.land()
                return

            if self.drone_ready == False:
                if self.check_drone_at_position(pos=self.hover_position ) == False:
                    self.go_to_position(self.hover_position )
                    self.node.get_logger().info("Going to initial position")
                else:
                    self.drone_ready = True

            else:
                t0 = time.time()
                self.publish_setpoint()
                t1 = time.time()
                self.Time+= (t1-t0)
                self.Time_T+= 1
            

        self.t += self.delta_t
        self.node.get_logger().info("Time: " + str(self.t))

    def go_to_position(self, desired_pos):
        self.compute_setpoint(desired_pos=desired_pos)
        self.publish_setpoint()

    def set_to_manual_mode(self):
        # Publish empty setpoints to ensure the drone remains stationary
        empty_setpoint = CommandOuter()
        empty_setpoint.thrust = 10000.0
        
        # Publish the empty setpoint multiple times
        for _ in range(10):
            self.controller_command_pub.publish(empty_setpoint)
        
        self.node.get_logger().info("Published empty setpoint for manual mode")
           

    def compute_setpoint(self, desired_pos=None):

        drone_state = self.drone_state_raw_log[-1]
        target_state = self.target_state_raw_log[-1]

        self.drone_state_log.append(drone_state)
        self.target_state_log.append(target_state)
        

        # self.node.get_logger().info("observe target[default]:" + str(target_state))
        # self.node.get_logger().info("current state[default]: " + str(drone_state))

        if desired_pos is not None:
            target_state[:3] = desired_pos.reshape(3,1)

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

        action = action_rotated # option 2msg.omega.x
        self.action_log.append(action)
        setpoint = CommandOuter()

        # setpoint.header.stamp.sec = self.t
        setpoint.thrust = float(action[0])
        setpoint.omega.x = float(action[1]) # pitch rate
        setpoint.omega.y = float(action[2]) # roll rate
        setpoint.omega.z = float(action[3]) # yaw rate
        # self.node.get_logger().info('error:'+str(action))

        # # Rescale thrust from (0,0.56) to (0,60000)
        thrust_motor = setpoint.thrust/ 4
        setpoint.thrust = self.thrust_newton_to_cmd(thrust_motor)

        disturbance_feedback = np.zeros((4,1))
        self.action_DF_log.append(disturbance_feedback)

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
    
    def publish_setpoint(self):
        self.compute_setpoint()
        self.controller_command_pub.publish(self.setpoint)

    def takeoff_autonomous(self):
        self.setpoint = CommandOuter()
        self.setpoint.is_takeoff = True
        self.controller_command_pub.publish(self.setpoint)
        self.controller_state = ControllerStates.takeoff
        time.sleep(2)
        
        self.node.get_logger().info("Switching from takeoff to flight state")
        self.controller_state = ControllerStates.flight

    def takeoff_manual(self):
        self.controller_state = ControllerStates.takeoff

    def land(self):
        self.controller_state = ControllerStates.landing

    def land_autonomous(self):
        self.setpoint = CommandOuter()
        self.setpoint.is_last_command = True
        self.controller_command_pub.publish(self.setpoint)
        self.controller_state = ControllerStates.idle

def main(args=None):

    default_controller = DefaultController()

if __name__ == '__main__':
    main()


    