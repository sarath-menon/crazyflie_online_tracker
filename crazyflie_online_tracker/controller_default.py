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

    def send_land_request(self, duration=6, height=0.15):
        self.req.group_mask = 0
        self.req.height = height
        self.req.duration.sec = duration
        self.req.duration.nanosec = 0

        self.future = self.land_cli.call_async(self.req)
        rclpy.spin_until_future_complete(self.node, self.future)
        return self.future.result()


    
    def exit_handler(self, signum, frame):
        print("Sending land command")
        self.land_autonomous()
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

        # self.node.get_logger().info(f"x: {x_diff}, y: {y_diff}, z: {z_diff}")

        if np.all(position_diff < np.array([tol_x, tol_y, tol_z])):
            return True
        else:
            return False

    def timer_callback(self, msg):

        self.t = msg.clock.sec  + msg.clock.nanosec / 1e9
        self.delta_t = self.t - self.T_prev
        self.T_prev = self.t


        # self.node.get_logger().info(f"Length of drone state log: {len(self.drone_state_raw_log)}")
        # self.node.get_logger().info(f"Length of target state log: {len(self.target_state_raw_log)}") 

        # self.node.get_logger().info(f"Controller state: {self.controller_state}")

           
        if len(self.drone_state_raw_log) == 0:
            self.node.get_logger().info('No drone state estimate is available. Skipping.')
            return
            
        if self.controller_state == ControllerStates.flight:
            if self.t >= T:
                self.node.get_logger().info('Simulation finished.')
                self.land()

                if self.save_log: # the simulation had started and has now been terminated
                    additional_info = f"_{target}_T{T}_f{f}_mode{mode}"
                    if filtering:
                        additional_info = f"_{target}_T{T}_f{f}_mode{mode}_Filtered"
                    new_filename = self.filename.value + additional_info
                    self.save_data(new_filename) # save log data to file for evaluation

                    # if self.plot:
                    #     self.node.get_logger().info('Printing the figures')
                    #     os.system("python3 ../crazyflie_online_tracker/plot_mine.py")

                # exit()


            elif self.drone_ready == False:
                if self.check_drone_at_position(pos=self.hover_position ) == False:
                    self.go_to_position(self.hover_position )
                    self.node.get_logger().info("Going to hover position")
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
            self.publish_setpoint(self.setpoint)

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

        # self.node.get_logger().info("Time:, delta_t: " + str(self.t) + ", " + str(self.delta_t))


    def set_to_manual_mode(self):
        # Publish empty setpoints to ensure the drone remains stationary
        empty_setpoint = FullState()
        
        # Publish the empty setpoint multiple times
        for _ in range(10):
            empty_setpoint.acc.z = 0.1
            self.publish_setpoint(empty_setpoint)
            time.sleep(0.1)
        
        self.node.get_logger().info("Published empty setpoint for manual mode")

    def go_to_position(self, desired_pos, desired_velocity=np.array([0.0, 0.0, 0.0])):

        drone_state = self.drone_state_raw_log[-1]

        target_state = np.array([desired_pos[0], desired_pos[1], desired_pos[2], 
                                 desired_velocity[0], desired_velocity[1], desired_velocity[2], 
                                 0, 0, 0]).reshape(9, 1)

        self.target_state_raw_log.append(target_state)

          # option 2: rotated error: USEDs
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

        self.publish_setpoint(setpoint)

        # compute disturbance w_t according to the latest drone and target state estimation
        if len(self.target_state_log) < 2:
            return
        else:
            last_target = self.target_state_log[-2]  # r_t

            curr_target = self.target_state_log[-1]  # r_{t+1}

        # w_t = Ar_t - r_{t+1}
        disturbance = self.A @ last_target - curr_target
        self.disturbances.append(disturbance)

    def publish_setpoint(self, setpoint):

        # # Rescale thrust from (0,0.56) to (0,60000)
        body_thrust = setpoint.acc.z 
        thrust_motor = body_thrust / 4
        setpoint.acc.z  = self.thrust_newton_to_cmd(thrust_motor)

        setpoint.header.stamp.sec = int(self.t)

        if len(self.target_state_raw_log) > 0:
            target_state = self.target_state_raw_log[-1]

            setpoint.pose.position.x = float(target_state[0])
            setpoint.pose.position.y = float(target_state[1])
            setpoint.pose.position.z = float(target_state[2])

            setpoint.twist.linear.x = float(target_state[3])
            setpoint.twist.linear.y = float(target_state[4])
            setpoint.twist.linear.z = float(target_state[5])

        self.setpoint = setpoint
        self.controller_command_pub.publish(self.setpoint)

        # self.node.get_logger().info("Thrust: " + str(setpoint.acc.z))
    

           

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
    
    def track_setpoint(self):
        self.compute_setpoint()
        self.publish_setpoint(self.setpoint)

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
        self.controller_state = ControllerStates.landing
        response = self.send_land_request()
        self.node.get_logger().info("Landing response: %s" % response)
        # self.controller_state = ControllerStates.idle

def main(args=None):

    default_controller = DefaultController()

if __name__ == '__main__':
    main()


    