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
from crazyflie_online_tracker_interfaces.srv import DroneStatus
import time
from crazyflie_interfaces.msg import FullState
from crazyflie_interfaces.srv import Land
from rosgraph_msgs.msg import Clock

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

        rclpy.init()
        self.node = rclpy.create_node("DefaultController")

        self.m = m

        self.controller_state_sub = self.node.create_subscription(ControllerState, 'controllerState', self.callback_controller_state, 10)
        self.controller_command_pub = self.node.create_publisher(FullState, '/cf231/cmd_vel', 10)
        self.controller_state_pub = self.node.create_publisher(ControllerState, 'controllerStateKF', 10)

        self.drone_state_sub = self.node.create_subscription(CrazyflieState, 'crazyflieState', self.callback_state_drone, 10)
        self.target_state_sub = self.node.create_subscription(TargetState, 'targetState', self.callback_state_target, 10)

        self.clock_sub = self.node.create_subscription(Clock, 'clock', self.timer_callback, 10)

        # service clients
        self.land_cli = self.node.create_client(Land, '/cf231/land')
        while not self.land_cli.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('service not available, waiting again...')
        self.req = Land.Request()


        # self.setpoint_publisher = self.node.create_publisher(Twist, '/cf231/cmd_vel_legacy', 10)

         # declare params
        self.node.declare_parameter('filename', 'Filename')
        self.node.declare_parameter('wait_for_drone_ready', False)

        # # services
        # self.srv = self.node.create_service(DroneStatus, 'drone_status', self.drone_status_callback)

        # get params
        self.filename = self.node.get_parameter('filename')

        #  # timer calbacks
        # self.timer = self.node.create_timer(self.delta_t, self.timer_callback)

        # Set to True to save data for post-processing
        self.save_log = True

        # Set to True to generate a plot immediately
        self.plot = True


        # for timing in algorithms
        self.Time = 0        # print(self.backend.time())
        self.Time_T = 0

        self.T_prev = 0.0

        signal.signal(signal.SIGINT, self.exit_handler)

        # takeoff drone
        self.initial_position = np.array([0, 0, 0])
        self.hover_position = np.array([0, 0, 0.4])

        self.set_to_manual_mode()
        time.sleep(2)

        #self.takeoff_autonomous()
        self.takeoff_manual()

        self.drone_ready = False

        rclpy.spin(self.node)        # print(self.backend.time())

    
    
    

           

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


    