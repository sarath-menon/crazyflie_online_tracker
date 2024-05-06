#!/usr/bin/env python3
from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial.transform import Rotation
import os
import rclpy
from rclpy.node import Node
from crazyflie_online_tracker_interfaces.msg import ControllerState, CommandOuter, CrazyflieState, TargetState
from datetime import datetime
from scipy import linalg
from std_msgs.msg import Empty
import yaml
import queue
from crazyflie_interfaces.msg import FullState
import time
from crazyflie_online_tracker_interfaces.msg import ControllerState, CommandOuter, CrazyflieState, TargetState
from geometry_msgs.msg import Twist
import signal
from crazyflie_online_tracker_interfaces.srv import DroneStatus
import time
from crazyflie_interfaces.msg import FullState
from crazyflie_interfaces.srv import Land
from rosgraph_msgs.msg import Clock


# copied from dfall package, for taking off and landing
motor_poly = [5.484560e-4, 1.032633e-6, 2.130295e-11]


# Load data from the YAML file
yaml_path = os.path.join(os.path.dirname(__file__), '../param/data.yaml')

with open(yaml_path, 'r') as file:
    yaml_data = yaml.safe_load(file)

# Extract the required variables
g = yaml_data['g']
m = yaml_data['m']
f = yaml_data['f']
delta_t = float(1/f)
T = yaml_data['T']
path = yaml_data['path']
filtering = yaml_data['filtering']
is_sim = yaml_data['is_sim']
target = yaml_data['target']
mode = yaml_data['mode']


class Controller(Node):
    def __init__(self):

        rclpy.init()
        super().__init__("Controller")

        self.drone_state_raw_log = [] # restore all received drone state measurements.
        self.target_state_raw_log = [] # restore all received target state measurements.
        self.drone_state_log = [] # restore drone states that are used for computing the setpoints(at 10Hz)
        self.target_state_log = [] # restore target states that are used for computing the setpoints(at 10Hz)
        self.action_log = [] # restore actions(at 10Hz)
        self.action_DF_log = [] # the bias part of the action
        self.default_action_log = [] # action computed via LQR
        self.optimal_action_log = [] # action computed knowing all future disturbances
        self.disturbances = []
        self.filtered_drone_state_raw_log = []
        self.filtered_drone_state_log = []
        self.system_output_raw_log = []
        self.system_output_log = []

        self.hover_height = 0.4
        self.hover_yaw = 0
        self.g = g
        self.m = m


        # INITIALIZATION OF CONTROLLER STATE
        self.controller_state = ControllerStates.idle

        self.m = m

        self.controller_state_sub = self.create_subscription(ControllerState, 'controllerState', self.callback_controller_state, 10)
        self.controller_command_pub = self.create_publisher(FullState, '/cf231/cmd_vel', 10)
        self.controller_state_pub = self.create_publisher(ControllerState, 'controllerStateKF', 10)

        self.drone_state_sub = self.create_subscription(CrazyflieState, 'crazyflieState', self.callback_state_drone, 10)
        self.target_state_sub = self.create_subscription(TargetState, 'targetState', self.callback_state_target, 10)

        self.clock_sub = self.create_subscription(Clock, 'clock', self.timer_callback, 10)

        # service clients
        self.land_cli = self.create_client(Land, '/cf231/land')
        while not self.land_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = Land.Request()

         # declare params
        self.declare_parameter('filename', 'Filename')
        self.declare_parameter('clock_frequency', 1000.0)

        # get params
        self.filename = self.get_parameter('filename').value
        self.clock_frequency = self.get_parameter('clock_frequency').value

        self.callback_wait = self.clock_frequency / f
        self.i = 0
        self.get_logger().info(f"callback_wait: {self.callback_wait}")

        # crazyflie sim time
        self.t = 0

        # for timing in algorithms
        self.Time = 0        # print(self.backend.time())
        self.Time_T = 0

        # to compute delta_t
        self.T_prev = 0.0

        # drone becomes ready when it reaches hover position
        self.drone_ready = False

        self.set_to_manual_mode()
        time.sleep(2)

        self.T_duration = T + self.t

        # for shutfown, fix later
        # self.create_subscription(Empty, 'shutdown', self.safe_shutdown, 10)


        
        self.takeoff_phase = 0 # 0 for unstarted, 1 for spin motor, 2 for moving up, 3 for goto setpoint
        self.setpoint_takeoff = None # record the position where the drone starts to take off
        self.setpoint_landing = None # record the position where the drone starts to land
        self.last_setpoint = None
        self.t_takeoff_start = None
        self.t_landing_start = None

        signal.signal(signal.SIGINT, self.exit_handler)

        # linear dynamic model: x_{t+1} = Ax_t + Bu_t
        # x = [x,y,z,vx,vy,vz,roll,pitch,yaw]
        # u = [thrust, omega_x, omega_y, omega_z] (omega is the angular rate with respect to the body frame)
        # linearized at hover pose: roll=pitch=yaw=0, vx=vy=vz=0, angular rate=0
        self.delta_t = delta_t
        A_outer = np.array([[0, 0, 0,   1, 0, 0,    0, 0, 0],
                            [0, 0, 0,   0, 1, 0,    0, 0, 0],
                            [0, 0, 0,   0, 0, 1,    0, 0, 0],

                            [0, 0, 0,   0, 0, 0,    0, self.g, 0],
                            [0, 0, 0,   0, 0, 0,   -self.g, 0, 0],
                            [0, 0, 0,   0, 0, 0,    0, 0, 0],

                            [0, 0, 0,   0, 0, 0,    0, 0, 0],
                            [0, 0, 0,   0, 0, 0,    0, 0, 0],
                            [0, 0, 0,   0, 0, 0,    0, 0, 0]])
        self.A = np.eye(9) + self.delta_t*A_outer
        B_outer = np.array([[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],

                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                     [1/self.m, 0, 0, 0],

                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        self.B = self.delta_t*B_outer
        # loss function: l_t = (x_t - g_t)^TQ(x_t - g_t) + u_t^Tu_t

        self.Q = np.diag([80, 80, 80, 10, 10, 10, 0.01, 0.01, 0.01]) #working
        #self.Q = np.diag([80, 80, 80, 10, 10, 10, 0.01, 0.01, 0.1])

        self.Q_takeoff = np.diag([10, 10, 10, 1, 1, 1, 1, 1, 1]) #old
        
        # self.Q = self.Q_takeoff
        # self.Q += 1e-5*np.eye(9) # necessary for DARE to be solvable.

        self.R = np.diag([0.2, 2.5, 2.5, 2.5]) # working
        # self.R = np.diag([0.7, 2.5, 2.5, 2.5])
        self.R_takeoff = np.diag([0.1, 0.8, 0.8, 0.1])
        # self.R = self.R_takeoff        

        self.K_star_takeoff, self.P_star_takeoff = self.solve_K_star(self.Q_takeoff, self.R_takeoff)
        self.K_star, self.P_star = self.solve_K_star(self.Q, self.R)

        # default K from dfall system
        #                             [ x,    y,    z,  vx,   vy,   vz,roll,pitch, yaw]
        # self.K_default = np.array([[  0,    0, 0.98,   0,    0, 0.25,   0,   0,   0],  # thrust
        #                            [  0, -3.2,    0,   0, -2.0,    0, 4.0,   0,   0],  # omega_y
        #                            [3.2,    0,    0, 2.0,    0,    0,   0, 4.0,   0],  # omega_x
        #                            [  0,    0,    0,   0,    0,    0,   0,   0, 2.3]]) # omega_z

        self.save_path = os.path.expanduser(path + 'data')
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)

        # check conditions of takeoff using flags
        self.cond_time = False
        self.cond_accuracy = False


    @abstractmethod
    def compute_setpoint(self):
        pass

    def callback_state_drone(self, data):
        # self.get_logger().info('Received drone state.')

        drone_state = np.zeros((9, 1))
        drone_state[StateIndex.x] = data.pose.position.x
        drone_state[StateIndex.y] = data.pose.position.y
        drone_state[StateIndex.z] = data.pose.position.z
        drone_state[StateIndex.vx] = data.velocity.linear.x
        drone_state[StateIndex.vy] = data.velocity.linear.y
        drone_state[StateIndex.vz] = data.velocity.linear.z
        q1 = data.pose.orientation.x
        q2 = data.pose.orientation.y
        q3 = data.pose.orientation.z
        q0 = data.pose.orientation.w
        r = Rotation.from_quat([q1,q2,q3,q0])
        euler = r.as_euler('ZYX')
        drone_state[StateIndex.roll] = euler[2]
        drone_state[StateIndex.pitch] = euler[1]
        drone_state[StateIndex.yaw] = euler[0]
        # self.get_logger().info(f"state: {drone_state}")
        self.drone_state_raw_log.append(drone_state)

    def callback_state_drone_filtered(self, data):
        filtered_drone_state = np.zeros((9, 1))
        filtered_drone_state[StateIndex.x] = data.pose.position.x
        filtered_drone_state[StateIndex.y] = data.pose.position.y
        filtered_drone_state[StateIndex.z] = data.pose.position.z
        filtered_drone_state[StateIndex.vx] = data.velocity.linear.x
        filtered_drone_state[StateIndex.vy] = data.velocity.linear.y
        filtered_drone_state[StateIndex.vz] = data.velocity.linear.z
        q1 = data.pose.orientation.x
        q2 = data.pose.orientation.y
        q3 = data.pose.orientation.z
        q0 = data.pose.orientation.w
        r = Rotation.from_quat([q1, q2, q3, q0])
        euler = r.as_euler('ZYX')
        filtered_drone_state[StateIndex.roll] = euler[2]
        filtered_drone_state[StateIndex.pitch] = euler[1]
        filtered_drone_state[StateIndex.yaw] = euler[0]
        # self.get_logger().info(f"state: {filtered_drone_state}")
        self.filtered_drone_state_raw_log.append(filtered_drone_state)

    def callback_system_output(self, data):
        output = np.zeros((9, 1))
        output[StateIndex.x] = data.pose.position.x
        output[StateIndex.y] = data.pose.position.y
        output[StateIndex.z] = data.pose.position.z
        output[StateIndex.vx] = data.velocity.linear.x
        output[StateIndex.vy] = data.velocity.linear.y
        output[StateIndex.vz] = data.velocity.linear.z
        q1 = data.pose.orientation.x
        q2 = data.pose.orientation.y
        q3 = data.pose.orientation.z
        q0 = data.pose.orientation.w
        r = Rotation.from_quat([q1, q2, q3, q0])
        euler = r.as_euler('ZYX')
        output[StateIndex.roll] = euler[2]
        output[StateIndex.pitch] = euler[1]
        output[StateIndex.yaw] = euler[0]
        # self.get_logger().info(f"state: {drone_state}")
        self.system_output_raw_log.append(output)

    def callback_state_target(self, data):

        # self.get_logger().info('Received target state.')

        target_state = np.zeros((9, 1))
        target_state[StateIndex.x] = float(data.pose.position.x)
        target_state[StateIndex.y] = float(data.pose.position.y)
        target_state[StateIndex.z] = float(data.pose.position.z)
        q1 = float(data.pose.orientation.x)
        q2 = float(data.pose.orientation.y)
        q3 = float(data.pose.orientation.z)
        q0 = float(data.pose.orientation.w)
        r = Rotation.from_quat([q1,q2,q3,q0])
        euler = r.as_euler('ZYX')
        target_state[StateIndex.roll] = float(euler[2])
        target_state[StateIndex.pitch] = float(euler[1])
        target_state[StateIndex.yaw] = float(euler[0])
        target_state[StateIndex.vx] = float(data.velocity.linear.x)
        target_state[StateIndex.vy] = float(data.velocity.linear.y)
        target_state[StateIndex.vz] = float(data.velocity.linear.z)
        # target_state[StateIndex.z] = self.hover_height

        self.target_state_raw_log.append(target_state)

    def callback_controller_state(self, data: CommandOuter):
        self.controller_state = data.state
        if self.controller_state == ControllerStates.flight:
            self.get_logger().info('Controller state is changed to Flight.')
        elif self.controller_state == ControllerStates.takeoff:
            self.get_logger().info('Controller state is changed to TAKEOFF.')
        elif self.controller_state == ControllerStates.landing:
            self.get_logger().info('Controller state is changed to LANDING.')


    def safe_shutdown(self):
        # place holder
        pass

    def solve_K_star(self,Q,R):
        # compute the gain matrix of LQR(assuming all target vectors are zero)
        P_star = linalg.solve_discrete_are(self.A, self.B, Q, R)
        B_transpose_P = self.B.T @ P_star
        K_star = linalg.inv(R + B_transpose_P @ self.B) @ B_transpose_P @ self.A
        return K_star, P_star

    def compute_setpoint_viaLQR(self, K_star, error_inertial, curr_yaw):
        """
        The default controller used for takeoff and landing. The value of LQR gains are taken from dfall_pkg/src/nodes/DefaultControllerService.cpp
        """
        # self.get_logger().info("error_inertial: "+str(error_inertial))
        # clip the error
        max_error_xy = 0.3
        error_x_inertial = max(min(error_inertial[0], max_error_xy), -max_error_xy)
        error_y_inertial = max(min(error_inertial[1], max_error_xy), -max_error_xy)
        max_error_z = 0.4
        error_z_inertial = max(min(error_inertial[2], max_error_z), -max_error_z)
        error_yaw_inertial = wrap_angle(error_inertial[8])
        max_error_yaw = np.deg2rad(60)
        error_yaw_inertial = max(min(error_yaw_inertial, max_error_yaw), -max_error_yaw)
        # convert error from inertial frame to body frame to compensate the linearization error of yaw!=0
        # self.get_logger().info("real error: " + str(error_inertial))
        sinyaw = np.sin(curr_yaw)
        cosyaw = np.cos(curr_yaw)
        error_body = error_inertial.copy()
        error_body[0] = error_x_inertial*cosyaw + error_y_inertial*sinyaw
        error_body[1] = error_y_inertial*cosyaw - error_x_inertial*sinyaw
        error_body[2] = error_z_inertial
        error_body[3] = error_inertial[3]*cosyaw + error_inertial[4]*sinyaw
        error_body[4] = error_inertial[4]*cosyaw - error_inertial[3]*sinyaw
        # self.get_logger().info("error_body: "+str(error_body))
        u = -K_star@error_body.reshape([9, 1])
        thrust = u[0] + self.m*self.g
        #thrust =  self.m*self.g

        roll_rate = u[1]
        pitch_rate = u[2]
        yaw_rate = u[3]
        return thrust, roll_rate, pitch_rate, yaw_rate

    def compute_setpoint_viaLQR_controller(self, K_star, error_inertial, curr_yaw):
        """
        The default controller used for takeoff and landing. The value of LQR gains are taken from dfall_pkg/src/nodes/DefaultControllerService.cpp
        """
        # self.get_logger().info("error_inertial: "+str(error_inertial))
        # clip the error
        max_error_xy = 0.3
        error_x_inertial = max(min(error_inertial[0], max_error_xy), -max_error_xy)
        error_y_inertial = max(min(error_inertial[1], max_error_xy), -max_error_xy)
        max_error_z = 0.4
        error_z_inertial = max(min(error_inertial[2], max_error_z), -max_error_z)
        error_yaw_inertial = wrap_angle(error_inertial[8])
        max_error_yaw = np.deg2rad(60)
        # convert error from inertial frame to body frame to compensate the linearization error of yaw!=0
        # self.get_logger().info("real error: " + str(error_inertial))
        error_body = error_inertial.copy()
        error_body[0] = error_x_inertial
        error_body[1] = error_y_inertial
        error_body[2] = error_z_inertial
        error_body[3] = error_inertial[3]
        error_body[4] = error_inertial[4]
        u = -K_star@error_body.reshape([9, 1])
        thrust = u[0] + self.m*self.g
        # self.get_logger().info("u[0] " + str(u[0]))
        roll_rate = u[1]
        pitch_rate = u[2]
        yaw_rate = u[3]
        return thrust, roll_rate, pitch_rate, yaw_rate


    def takeoff(self, desired_pos=np.array([0, 0, 0.3])):
        drone_state = self.drone_state_raw_log[-1]
        self.setpoint_takeoff = drone_state[:3]

        if self.t_takeoff_start is None:
            self.t_takeoff_start = self.t
            time_takeoff = 0.0
        else:
            time_takeoff = self.t - self.t_takeoff_start

        time_spin_motor = 0.8 # phase1: gradually increase thrust
        time_move_up = 0.2 # phase2: gradually move to 0.3m high and desired yaw angle(dfall_pkg use 0.4m)
        
        time_goto_setpoint = 2 # phase3: fly to the desired (x, y, z) position
        min_spin_motor_cmd = 1000
        min_spin_motor_thrust_total = 4*thrust_cmd_to_newton(min_spin_motor_cmd)
        max_spin_motor_cmd = 9000
        max_spin_motor_thrust_total = 4*thrust_cmd_to_newton(max_spin_motor_cmd)
        takeoff_start_height = 0.1
        takeoff_end_height = 0.4

        command = FullState()
        
        if time_takeoff <= time_spin_motor:
            self.get_logger().info("takeoff: spin motor for "+str(time_takeoff)+' seconds')
            if self.takeoff_phase < 1:
                self.takeoff_phase = 1
            
            time_proportion = min(1, self.t/time_spin_motor)

            command.acc.z = float(min_spin_motor_thrust_total + (max_spin_motor_thrust_total - min_spin_motor_thrust_total) * time_proportion)

        elif time_takeoff <= time_spin_motor+time_move_up:
            self.get_logger().info("takeoff: move up for "+str(time_takeoff)+' seconds')
            if self.takeoff_phase < 2:
                self.takeoff_phase = 2

            time_proportion = min(1, (time_takeoff-time_spin_motor)/(0.8*time_move_up)) # gradually increase the deired height and yaw angle
            desired_height = self.setpoint_takeoff[2] + takeoff_start_height + (takeoff_end_height - takeoff_start_height)*time_proportion
            error_inertial = drone_state.copy()
            error_inertial[0] = drone_state[0] - self.setpoint_takeoff[0]
            error_inertial[1] = drone_state[1] - self.setpoint_takeoff[1]
            error_inertial[2] = drone_state[2] - desired_height
            # error_inertial[8] = 0
            error_inertial[8] = wrap_angle(drone_state[8]) * time_proportion
            [thrust, roll_rate, pitch_rate, yaw_rate] = self.compute_setpoint_viaLQR(self.K_star_takeoff, error_inertial, drone_state[8])
            # self.get_logger().info('drone state:'+str(drone_state))
            # self.get_logger().info('error:'+str(error_inertial))
            # self.get_logger().info('action:'+str([thrust, roll_rate, pitch_rate, yaw_rate]))
            # assign desired values to command as defined in the dfall decoder at the onborad firmware
     
            command.acc.z = float(thrust)
            command.twist.linear.x = float(pitch_rate)
            command.twist.linear.y = float(roll_rate)
            command.twist.linear.z = float(yaw_rate)
        # else:
        #     if self.takeoff_phase < 3:
        #         self.takeoff_phase = 3
        #         self.setpoint_takeoff[2] = self.setpoint_takeoff[2] + takeoff_end_height
        #     if desired_pos is None:
        #         desired_pos = self.setpoint_takeoff
        #     desired_pos_limited = self.limit_pos_change(self.setpoint_takeoff, desired_pos)
        #     self.get_logger().info("takeoff phase 3: goto setpoint")

            
        #     # self.get_logger().info('desired pos limited: '+str(desired_pos_limited))
            
        #     error_inertial = drone_state.copy()
        #     error_inertial[0] = drone_state[0] - desired_pos_limited[0]
        #     error_inertial[1] = drone_state[1] - desired_pos_limited[1]
        #     error_inertial[2] = drone_state[2] - desired_pos_limited[2]
        #     self.setpoint_takeoff = desired_pos_limited
        #     [thrust, roll_rate, pitch_rate, yaw_rate] = self.compute_setpoint_viaLQR(self.K_star_takeoff, error_inertial, drone_state[8])
        #     # assign desired values to command as defined in the dfall decoder at the onborad firmware
            
        #     command.acc.z = float(thrust)
        #     command.twist.linear.x = float(pitch_rate)
        #     command.twist.linear.y = float(roll_rate)
        #     command.twist.linear.z = float(yaw_rate)

        #     time_condition = time_takeoff > time_spin_motor + time_move_up + time_goto_setpoint
        #     position_tolerance = [0.1, 0.1, 0.05]
            
        #     position_differences_values = [np.abs(drone_state[i] - desired_pos[i]) for i in range(3)]


        #     position_differences = [position_differences_values[i] < position_tolerance[i] for i in range(3)]


        #     position_condition = all(position_differences)

        #     self.get_logger().info(f"Position differences values: {position_differences_values}")

        else:
            self.controller_state = ControllerStates.flight
            self.last_setpoint = self.setpoint_takeoff

        # command.thrust = self.thrust_newton_to_cmd(command.thrust)
        # self.get_logger().info('Takeoff command: '+str(command.thrust))
        return command

    def thrust_newton_to_cmd(self, thrust):
        motor_poly = [5.484560e-4, 1.032633e-6, 2.130295e-11]
        motor_cmd_min = 1000
        motor_cmd_max = 65000
        
        # copied from dfall_pkg/src/nodes/DefaultControllerService.cpp: float computeMotorPolyBackward(float thrust)
        cmd_16bit = (-motor_poly[1] + np.sqrt(max(0,motor_poly[1]**2 - 4 * motor_poly[2] * (motor_poly[0] - thrust)))) / (2 * motor_poly[2])
        if cmd_16bit < motor_cmd_min:
            cmd_16bit = motor_cmd_min
        elif cmd_16bit > motor_cmd_max:
            cmd_16bit = motor_cmd_max
        cmd_16bit = np.ushort(cmd_16bit)

        return float(cmd_16bit)



    def landing(self, desired_pos=np.array([0, 0, 0.15])):
        drone_state = self.drone_state_raw_log[-1]

        time_landing_max = 4
        landing_end_height = desired_pos[2]
        landing_spin_motor_cmd = 10000

        command = CommandOuter()

        if self.t_landing_start is None:
            self.t_landing_start = self.t
            time_landing = 0
        else:
            time_landing = self.t - self.t_landing_start
        if self.setpoint_landing is None:
            self.setpoint_landing = drone_state[:3]
        if drone_state[2] < landing_end_height or time_landing>time_landing_max:
            
            
            command.thrust = float(4 * thrust_cmd_to_newton(landing_spin_motor_cmd))
            command.omega.x = float(0)
            command.omega.y = float(0)
            command.omega.z = float(0)
            self.controller_state = ControllerStates.idle

        else:
            if desired_pos is None:
                desired_pos = self.setpoint_landing.copy()

            desired_pos_limited = self.limit_pos_change(self.setpoint_landing, desired_pos)
            # self.get_logger().info('limited pos: '+str(desired_pos_limited))
            # self.get_logger().info('drone state: '+str(drone_state))
            error_inertial = drone_state.copy()
            error_inertial[0] = drone_state[0] - desired_pos_limited[0]
            error_inertial[1] = drone_state[1] - desired_pos_limited[1]
            error_inertial[2] = drone_state[2] - desired_pos_limited[2]
            self.setpoint_landing = desired_pos_limited
            [thrust, roll_rate, pitch_rate, yaw_rate] = self.compute_setpoint_viaLQR(self.K_star_takeoff, error_inertial, drone_state[8])
            # self.get_logger().info('action: '+str([thrust, roll_rate, pitch_rate, yaw_rate]))
            # assign desired values to command as defined in the dfall decoder at the onborad firmware

            command.thrust = float(thrust)
            command.omega.x = float(pitch_rate)
            command.omega.y = float(roll_rate)
            command.omega.z = float(yaw_rate)
        return command

    def limit_pos_change(self, curr_pos, desired_pos):
        max_horizontal_change_per_second = 0.6
        max_horizontal_change = max_horizontal_change_per_second * self.delta_t
        max_vertical_change_per_second = 0.4
        max_vertical_change = max_vertical_change_per_second * self.delta_t
        actual_horizontal_change = np.linalg.norm(curr_pos[:2] - desired_pos[:2])
        actual_vertical_change = np.abs(curr_pos[2] - desired_pos[2])
        ellipse_value = actual_horizontal_change**2/max_horizontal_change**2 + actual_vertical_change**2/max_vertical_change**2
        if ellipse_value <= 1:
            return desired_pos
        else:
            desired_pos_limited = np.zeros_like(desired_pos)
            desired_pos_limited[0] = curr_pos[0] + 1/np.sqrt(ellipse_value)*(desired_pos[0]-curr_pos[0])
            desired_pos_limited[1] = curr_pos[1] + 1/np.sqrt(ellipse_value)*(desired_pos[1]-curr_pos[1])
            desired_pos_limited[2] = curr_pos[2] + 1/np.sqrt(ellipse_value)*(desired_pos[2]-curr_pos[2])
            return desired_pos_limited

    def track_setpoint(self):
        self.compute_setpoint()
        self.publish_setpoint(self.setpoint)

    def takeoff_autonomous(self):
        self.setpoint = CommandOuter()
        self.setpoint.is_takeoff = True
        self.controller_command_pub.publish(self.setpoint)
        self.controller_state = ControllerStates.takeoff
        
        self.get_logger().info("Switching from takeoff to flight state")
        self.controller_state = ControllerStates.flight

    def takeoff_manual(self):
        self.controller_state = ControllerStates.takeoff

    def land(self):
        self.controller_state = ControllerStates.landing

    def land_autonomous(self):
        self.controller_state = ControllerStates.landing
        response = self.send_land_request()
        self.get_logger().info("Landing response: %s" % response)
        # self.controller_state = ControllerStates.idle

    def timer_callback(self, msg):

        self.t = msg.clock.sec  + msg.clock.nanosec / 1e9
        self.delta_t = self.t - self.T_prev
        self.T_prev = self.t
        # self.get_logger().info(f"Time: {self.delta_t}")
        self.i += 1


        # self.get_logger().info(f"Length of drone state log: {len(self.drone_state_raw_log)}")
        # self.get_logger().info(f"Length of target state log: {len(self.target_state_raw_log)}") 

        # self.get_logger().info(f"Controller state: {self.controller_state}")

        if self.i % self.callback_wait != 0:
            return


        if len(self.drone_state_raw_log) == 0:
            self.get_logger().info('No drone state estimate is available. Skipping.')
            return
            
        if self.controller_state == ControllerStates.flight:
            if self.t >= self.T_duration:
                self.get_logger().info('Simulation finished.')
                self.land()

                if self.save_log: # the simulation had started and has now been terminated
                    additional_info = f"_{target}_T{T}_f{f}_mode{mode}"
                    if filtering:
                        additional_info = f"_{target}_T{T}_f{f}_mode{mode}_Filtered"
                    new_filename = self.filename + additional_info
                    self.save_data(new_filename) # save log data to file for evaluation

                    if self.plot:
                        self.get_logger().info('Printing the figures')
                        os.system("python3 ../crazyflie_online_tracker/plot.py")

                    exit()


            elif self.drone_ready == False:
                if self.check_drone_at_position(pos=self.hover_position ) == False:
                    self.go_to_position(self.hover_position )
                    self.get_logger().info("Going to hover position")
                else:
                    self.drone_ready = True
                    os.system("ros2 param set /state_estimator_target_virtual wait_for_drone_ready True")

            elif len(self.target_state_raw_log) == 0:
                self.get_logger().info('No target state estimate is available. Skipping.')

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
                self.get_logger().info("Drone landed")
                exit()

        elif self.controller_state == ControllerStates.landing: 
            if self.check_drone_at_position(pos=self.hover_position) == False:
                    self.go_to_position(self.hover_position)
                    self.get_logger().info("Going to hover position before landing")

            else:
                # self.setpoint = self.landing()
                # self.controller_command_pub.publish(self.setpoint)
                self.land_autonomous()
                self.get_logger().info("Landing started")

        # self.get_logger().info("Time:, delta_t: " + str(self.t) + ", " + str(self.delta_t))


    def set_to_manual_mode(self):
        # Publish empty setpoints to ensure the drone remains stationary
        empty_setpoint = FullState()
        
        # Publish the empty setpoint multiple times
        for _ in range(10):
            empty_setpoint.acc.z = 0.1
            self.publish_setpoint(empty_setpoint)
            time.sleep(0.1)
        
        self.get_logger().info("Published empty setpoint for manual mode")

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
        # self.action_log.append(action)

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

        # self.get_logger().info("Thrust: " + str(setpoint.acc.z))

    def exit_handler(self, signum, frame):
        print("Sending land command")
        self.land_autonomous()
        exit()

    def drone_status_callback(self, request, response):
        response.is_drone_ready = self.drone_ready
        self.get_logger().info("is drone ready server: %d" % response.is_drone_ready)
        return response

    def check_drone_at_position(self, pos=np.array([0, 0, 0])):
        drone_state = self.drone_state_raw_log[-1]
        x, y, z, vx, vy, vz, yaw, yaw_rate, thrust = drone_state

        tol_x = 0.1
        tol_y = 0.1
        tol_z = 0.05

        position_diff = np.abs([x - pos[0], y - pos[1], z - pos[2]])
        x_diff, y_diff, z_diff = position_diff

        # self.get_logger().info(f"x: {x_diff}, y: {y_diff}, z: {z_diff}")

        if np.all(position_diff < np.array([tol_x, tol_y, tol_z])):
            return True
        else:
            return False

    def send_land_request(self, duration=2, height=0.15):
        self.req.group_mask = 0
        self.req.height = height
        self.req.duration.sec = duration
        self.req.duration.nanosec = 0

        self.future = self.land_cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()


    def save_data(self, filename):
        now = datetime.now()
        timestr = now.strftime("%Y%m%d%H%M%S")
        filename = timestr + '_' + filename
        # depending on filtering the drone_state_log will be populated from the filtered raw log or from
        # the unfiltered one
        if filtering: # if we are in the simulation we need to save the generated system output
            if is_sim:
                np.savez(os.path.join(self.save_path, filename),
                        target_state_log = np.array(self.target_state_log),
                        drone_state_log= np.array(self.drone_state_log),
                        action_log = np.array(self.action_log),
                        action_DF_log = np.array(self.action_DF_log),
                        disturbance_log = np.array(self.disturbances),
                        default_action_log = np.array(self.default_action_log),
                        optimal_action_log = np.array(self.optimal_action_log),
                        filtered_drone_state_log = np.array(self.filtered_drone_state_log),
                        system_output=np.array(self.system_output_raw_log))
            else:
                np.savez(os.path.join(self.save_path, filename),
                         target_state_log=np.array(self.target_state_log),
                         drone_state_log=np.array(self.drone_state_log),
                         action_log=np.array(self.action_log),
                         action_DF_log=np.array(self.action_DF_log),
                         disturbance_log=np.array(self.disturbances),
                         default_action_log=np.array(self.default_action_log),
                         optimal_action_log=np.array(self.optimal_action_log),
                         filtered_drone_state_log=np.array(self.filtered_drone_state_log))
        else:
            np.savez(os.path.join(self.save_path, filename),
                     target_state_log=np.array(self.target_state_log),
                     drone_state_log=np.array(self.drone_state_log),
                     action_log=np.array(self.action_log),
                     action_DF_log=np.array(self.action_DF_log),
                     disturbance_log=np.array(self.disturbances),
                     default_action_log=np.array(self.default_action_log),
                     optimal_action_log=np.array(self.optimal_action_log))
        self.get_logger().info("Trajectory data has been saved to" + self.save_path + '/' + filename + '.npz')

    def load_data(self, filename):
        try:
            data = np.load(os.path.join(self.save_path, filename))
            self.target_state_log = data['target_state_log']
            self.drone_state_algorithm_log = data['drone_state_log']
            self.action_algorithm_log = data['action_log']
            self.get_logger().info("Load data from" + self.save_path + '/', filename + '.npz')
            return True
        except:
            return False


def wrap_angle(angle):
    while angle > np.pi:
        angle -= 2*np.pi
    while angle < -np.pi:
        angle += 2*np.pi
    return angle

def thrust_cmd_to_newton(cmd):
    thrust = motor_poly[2]*cmd**2 + motor_poly[1]*cmd + motor_poly[0]
    return thrust
    
class StateIndex():
    x = 0
    y = 1
    z = 2
    vx = 3
    vy = 4
    vz = 5
    roll = 6
    pitch = 7
    yaw = 8

class MotionIndex():
    stop = 0
    forward = 1
    right = 2
    backward = 3
    left = 4

class ControllerStates():
    flight = 0
    takeoff = 1
    landing = 2
    idle = 3
