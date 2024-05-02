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
        self.node.declare_parameter('wait_for_drone_ready', False)

        # # services
        # self.srv = self.node.create_service(DroneStatus, 'drone_status', self.drone_status_callback)

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
        exit()

    def drone_status_callback(self, request, response):
        response.is_drone_ready = self.drone_ready
        self.node.get_logger().info("is drone ready server: %d" % response.is_drone_ready)
        return response

    def check_drone_at_position(self, pos=np.array([0, 0, 0])):
        drone_state = self.drone_state_raw_log[-1]
        x, y, z, vx, vy, vz, yaw, yaw_rate, thrust = drone_state

        tol_x = 0.1
        tol_y = 0.1
        tol_z = 0.05

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

        if len(self.drone_state_raw_log) == 0:
            self.node.get_logger().info('No drone state estimate is available. Skipping.')
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

            
        if self.controller_state == ControllerStates.flight:
            if self.t >= T:
                self.node.get_logger().info('Simulation finished.')
                self.land()

            elif self.drone_ready == False:
                if self.check_drone_at_position(pos=self.hover_position ) == False:
                    self.go_to_position(self.hover_position )
                    self.node.get_logger().info("Going to initial position")
                else:
                    self.drone_ready = True
                    os.system("ros2 param set /state_estimator_target_virtual wait_for_drone_ready True")

            elif len(self.target_state_raw_log) == 0:
                self.node.get_logger().info('No target state estimate is available. Skipping.')

            else:
                t0 = time.time()
                self.track_setpoint()
                t1 = time.time()
                self.Time+= (t1-t0)
                self.Time_T+= 1

        elif self.controller_state == ControllerStates.takeoff:
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

            

        self.t += self.delta_t
        self.node.get_logger().info("Time: " + str(self.t))

    # def go_to_position(self, desired_pos):
    #     self.compute_setpoint(desired_pos=desired_pos)
    #     self.track_setpoint()

    def set_to_manual_mode(self):
        # Publish empty setpoints to ensure the drone remains stationary
        empty_setpoint = CommandOuter()
        empty_setpoint.thrust = 10000.0
        
        # Publish the empty setpoint multiple times
        for _ in range(10):
            self.controller_command_pub.publish(empty_setpoint)
            time.sleep(0.1)
        
        self.node.get_logger().info("Published empty setpoint for manual mode")

    def go_to_position(self, desired_pos, desired_velocity=np.array([0.0, 0.0, 0.0])):

        drone_state = self.drone_state_raw_log[-1]

        target_state = np.array([desired_pos[0], desired_pos[1], desired_pos[2], 
                                 desired_velocity[0], desired_velocity[1], desired_velocity[2], 
                                 0, 0, 0]).reshape(9, 1)

          # option 2: rotated error: USEDs
        error = drone_state - target_state
        [thrust, roll_rate, pitch_rate, yaw_rate] = self.compute_setpoint_viaLQR(self.K_star, error, drone_state[8])
        action_rotated = np.array([thrust, pitch_rate, roll_rate, yaw_rate])

        action = action_rotated # option 2msg.omega.x
        self.action_log.append(action)
        setpoint = CommandOuter()

        # setpoint.header.stamp.sec = self.t
        setpoint.thrust = float(action[0])
        setpoint.omega.x = float(action[1]) # pitch rate
        setpoint.omega.y = float(action[2]) # roll rate
        setpoint.omega.z = float(action[3]) # yaw rate'

        self.setpoint = setpoint

         # # Rescale thrust from (0,0.56) to (0,60000)
        thrust_motor = setpoint.thrust/ 4
        setpoint.thrust = self.thrust_newton_to_cmd(thrust_motor)

        self.controller_command_pub.publish(self.setpoint)

           

    def compute_setpoint(self, desired_pos=None):

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
    
    def track_setpoint(self):
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


    