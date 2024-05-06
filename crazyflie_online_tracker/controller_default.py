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
from crazyflie_interfaces.msg import FullState

# Load data from the YAML file
yaml_path = os.path.join(os.path.dirname(__file__), '../param/data.yaml')

with open(yaml_path, 'r') as file:
    data = yaml.safe_load(file)

# Extract the required variableshover
m = data['m']
g = data['g']
f = data['f']
T = data['T']
delta_t = float(1/f)
filtering = data['filtering']


class DefaultController(Controller):
    '''
    Naive LQR (or simply LQR) controller
    '''
    def __init__(self):
        super().__init__()

        # Set to True to save data for post-processing
        self.save_log = True

        # Set to True to generate a plot immediately
        self.plot = True

        # takeoff drone
        self.initial_position = np.array([0, 0, 0])
        self.hover_position = np.array([0, 0, 0.4])

        self.takeoff_manual()

        rclpy.spin(self)        # print(self.backend.time())

           

    def compute_setpoint(self):

        drone_state = self.drone_state_raw_log[-1]
        target_state = self.target_state_raw_log[-1]

        self.drone_state_log.append(drone_state)
        self.target_state_log.append(target_state)
        
        # option 2: rotated error: USED
        error = drone_state - target_state
        [thrust, roll_rate, pitch_rate, yaw_rate] = self.compute_setpoint_viaLQR(self.K_star, error, drone_state[8])
        action_rotated = np.array([thrust, pitch_rate, roll_rate, yaw_rate])

        action = action_rotated # option 2msg.omega.x
        self.action_log.append(action)

        setpoint = FullState()

        setpoint.acc.z = float(action[0])
        setpoint.twist.angular.x = float(action[1]) # pitch rate
        setpoint.twist.angular.y = float(action[2]) # roll rate
        setpoint.twist.angular.z = float(action[3]) # yaw rate


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
    
    

def main(args=None):

    default_controller = DefaultController()

if __name__ == '__main__':
    main()


    