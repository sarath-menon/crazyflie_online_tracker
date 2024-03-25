#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import yaml
import math
import os
import scipy.interpolate
from scipy import linalg
from scipy.spatial.transform import Rotation
from crazyflie_online_tracker_interfaces.msg import TargetState
from crazyflie_online_tracker_interfaces.srv import PublishSingleTarget
from .state_estimator import StateEstimator, MotionIndex
import std_msgs.msg
import time

# Load data from the YAML file
yaml_path = os.path.join(os.path.dirname(__file__), '../param/data.yaml')

with open(yaml_path, 'r') as file:
    data = yaml.safe_load(file)

# Extract the required variables
g = data['g']
m = data['m']
f = data['f']
T = data['T']
compl = data['compl']
is_sim = data['is_sim']
add_noise = data['add_noise']
target = data['target']
mode = data['mode']
filtering = data['filtering']


class TargetStateEstimator(StateEstimator):
    '''
    This class creates artificial reference targets and publishes them online
    '''
    def __init__(self):
        super().__init__()

        # ros2 config
        rclpy.init()
        self.node = rclpy.create_node("state_estimator_target_virtual")

        # publishers and subscribers
        self.state_pub = self.node.create_publisher(TargetState, 'targetState', 10)
        self.service = self.node.create_service(PublishSingleTarget, '/publish_single_target', self.handle_publish_single_target)
        # self.delta_t = 0.1 # model discretion timestep
        self.delta_t = float(1/f)
        self.target = target

        self.node.declare_parameter('wait_for_simulator_initialization', False)
        self.wait_for_simulator_initialization = self.node.get_parameter('wait_for_simulator_initialization')

         # timer callbacks
        timer_period = 0.5  # seconds
        self.timer = self.node.create_timer(timer_period, self.timer_callback)

        if compl == 'sqrt':
            self.compl = np.sqrt(T*f)
        elif compl == 'T3/2':
            self.compl = T**(2/3)*f
        else:
            self.compl = T*f
        self.num_changes = math.floor(f*T/self.compl)
        self.dyn_count = 0
        
        self.curr_state = None # record last state for recursive update

        # for STATIC target
        if is_sim:
            self.initial_state_stationary = np.array([0.6, 0.2, 0, 0])
            self.switch_state = np.array([0.6, 0.5, 0, 0])
        else:
            self.initial_state_stationary = np.array([1.2, 0.4, 0, 0])
        self.S_stationary = np.eye(4)
        self.v_Stationary = np.zeros((4, 1))

        # for CONSTANT VELOCITY target
        self.initial_state_const_vel = np.array([0, 0, 0.2, 0]).reshape((4, 1))
        self.S_const_vel = np.eye(4)
        self.S_const_vel[0, 2] = self.delta_t
        self.S_const_vel[1, 3] = self.delta_t

        # for CONSTANT ACCELERATION target
        self.initial_state_const_accel = np.array([0, 0, 0.1, 0]).reshape((4, 1))
        self.v_const_accel = np.array([0, 0, 0, 0.01]).reshape((4, 1))

        # for DECAYING VELOCITY target
        self.S_decay_vel = np.eye(4)

        self.initial_state_decay_vel = np.array([0.6, 0.2, 0.1, 0.1]).reshape((4, 1))
        decay_factor = 0.0
        self.S_decay_vel[0, 2] = 0.0 #self.delta_t
        self.S_decay_vel[1, 3] = 0.0 #self.delta_t
        self.S_decay_vel[2, 2] = decay_factor
        self.S_decay_vel[3, 3] = decay_factor

        self.S_alt_decay = self.S_decay_vel.copy()

        self.switch = 1.0

        # for CIRCULAR target
        self.S_circular = self.S_const_vel.copy()
        if is_sim:
            self.radius = 0.5
            self.velocity = 0.3
            self.initial_state_circular = np.array([0.6, 0.2, self.velocity, 0]).reshape([4, 1])
        else:
            self.radius = 0.3
            self.velocity = 0.15 #0.3
            self.initial_state_circular = np.array([0.7, 0.4, self.velocity, 0]).reshape([4, 1])
        self.circ_vel =  self.velocity
        delta_theta = self.delta_t*self.velocity/self.radius
        self.S_circular[2:, 2:] = np.array([[np.cos(delta_theta), -np.sin(delta_theta)],
                                            [np.sin(delta_theta),  np.cos(delta_theta)]])

        # for SQUARE target
        self.motion = MotionIndex.stop
        self.forwardMax = 0.5
        self.sideMax = 0.5
        self.initial_state_square = np.array([0.5, 0.2, 0, 0.1]).reshape((4, 1))
        
        # for HEART target
        vel_heart = 0.15
        self.initial_state_heart = np.array([0.7, 0.7, 0, vel_heart]).reshape((4, 1))

        # for PERIODIC target
        self.t = None
        self.initial_state_square_wave = np.array([0.6, 0.1, 0.0, 0.]).reshape((4, 1))
        self.initial_state_sin = np.array([0, 0, 0.2, 0]).reshape((4, 1))

        # for WAYPOINT FOLLOWING target
        self.waypoints_original = np.array([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]])
        figure_8_part1 = np.array([[0.9, 0.85], [1.5, 1], [1.8, 0.65]])
        figure_8_part2 = figure_8_part1[::-1] @ np.array([[1, 0], [0, -1]])
        figure_8_part3 = figure_8_part1 @ np.array([[-1, 0], [0, 1]])
        figure_8_part4 = figure_8_part1[::-1] @ np.array([[-1, 0], [0, -1]])
        scale = 0.3
        self.waypoints_original = np.concatenate(([[0, 0]],
                                                  figure_8_part1,
                                                  figure_8_part2,
                                                  [[0, 0]],
                                                  figure_8_part3,
                                                  figure_8_part4,
                                                  #   [[0, 0]]
                                                  ))
        self.waypoints_original = np.array(self.waypoints_original.tolist() * 20)
        self.waypoints_original = self.waypoints_original * scale
        self.waypoints_original[:, 1] += 0.5
        # self.waypoints_original[:, 0] += 0.7
        self.smoothness_factor = 0.1 * scale ** 2

        self.waypoints_smoothed = []
        self.is_waypoints_smoothed = False

        self.mode = mode # 0 for targets moving on the x-y plane, 1 for xz, 2 for yz
        self.default_x = 0.8 # within [-1.6, 2] is safe for hardware experiment
        self.default_y = 0.7 # [0.25,1.2]
        self.default_z = 0.4 # [0.2, 1.2]

        if is_sim:
            self.count = 0
        else:
            self.count = 50 # don't start to move until the drone has taken off

        rclpy.spin(self.node)
        self.node.destroy_node()
        rclpy.shutdown()


    def timer_callback(self):
        # self.node.get_logger().info(f"Publishing target")

        if rclpy.ok():
            self.publish_state()
            # if self.wait_for_simulator_initialization:
            #     time.sleep(4)
            #     count -= 1
            #     if count < 0:
            #         self.wait_for_simulator_initialization = False


    def publish_state(self):
        if target == 'stationary_target':
            self.get_stationary_target()
        elif target == 'square_target':
            self.get_square_target(0.1)
        elif target == 'circular_target':
            self.get_circular_target()
        elif target == 'decay_velocity':
            self.get_decay_velocity_target()
        elif target == 'sin_target':
            self.get_sin_target(2, 0.3)
        elif target == 'square_wave':
            self.get_square_wave_target(0.2, 1)
        elif target == 'const_accel':
            self.get_constant_accel_target()
        elif target == 'heart_target':
            self.get_heart_shaped_target()
        elif target == 'waypoint':
            self.get_waypoint_following_target()
        elif target == 'switching_target':
            self.get_switching_target()


        if self.curr_state is not None:
            self.set_state_msg(add_noise=add_noise)
            self.state_pub.publish(self.state)
            return True
        else:
            self.get_logger().debug("The target state has not been assigned with any value.")
            return False


    def get_stationary_target(self):
        if self.curr_state is None:
            self.curr_state = self.initial_state_stationary

    def get_switching_target(self):

        if self.curr_state is None:
            # self.curr_state = self.initial_state_decay_vel
            self.curr_state = self.initial_state_circular
        self.dyn_count += 1

        '''
        Target used as a benchmark to compare with other online controllers; uncomment/comment to disable/enable
        '''

        # if self.dyn_count > self.num_changes:
        #     # print('AAAAAA')
        #     self.dyn_count = 0
        #     self.curr_state[2] *= -1
        #     self.curr_state[3] *= -1
        #     # self.circ_vel += 0.04
        #     self.radius = -self.radius*1.01
        #
        # delta_theta = self.delta_t * self.velocity / self.radius
        # self.S_circular[2:, 2:] = np.array([[np.cos(delta_theta), -np.sin(delta_theta)],
        #                                     [np.sin(delta_theta), np.cos(delta_theta)]])
        # # # Apply the target update
        # self.curr_state = self.S_circular @ self.curr_state

        # Switching circle/ setpoints
        # if self.dyn_count >  self.num_changes:
        #     self.dyn_count = 0
        #
        #     self.S_circular[1, 1] = -1 * self.S_circular[1, 1]
        #     # self.S_circular[0, 0] = -1 * self.S_circular[0, 0]
        #
        # # Apply the target update
        # self.curr_state = self.S_decay_vel @ self.curr_state



        '''
        Target used in the example of tuning the best forgetting factor gamma; uncomment/comment to disable/enable
        '''

        if self.dyn_count > self.num_changes:
            self.dyn_count = 0
            self.curr_state[1] += 0.1
            self.curr_state[0] -= 0.1
            # self.circ_vel += 0.04
            self.radius = self.radius * np.random.uniform(0.7,1.5)

        delta_theta = self.delta_t * self.velocity / self.radius
        self.S_circular[2:, 2:] = np.array([[np.cos(delta_theta), -np.sin(delta_theta)],
                                            [np.sin(delta_theta), np.cos(delta_theta)]])

        # Apply the target update
        self.curr_state = self.S_circular @ self.curr_state

    def get_circular_target(self):
        if self.curr_state is None or self.count > 0:
            self.count -=1 # wait for taking off
            self.curr_state = self.initial_state_circular
        else:
            self.curr_state = self.S_circular@self.curr_state

    def get_constant_velocity_target(self, add_noise=add_noise):
        if self.curr_state is None:
            self.curr_state = self.initial_state_const_vel
        else:
            if add_noise:
                S_noise = np.zeros((4, 4))
                S_noise[:2,:2] = np.random.normal(0, 0.05, size=(2,2))
                S_noise[2:,2:] = np.random.normal(0, 0.0, size=(2,2))
                S_with_noise = self.S_const_vel+S_noise
                S = self.projection(S_with_noise, 1)
            else:
                S = self.S_const_vel
            self.curr_state = S@self.curr_state

    def get_decay_velocity_target(self):
        if self.curr_state is None:
            self.curr_state = self.initial_state_decay_vel
        else:
            self.curr_state = self.S_decay_vel@self.curr_state

    def get_constant_accel_target(self):
        if self.curr_state is None:
            self.curr_state = self.initial_state_const_accel
        else:
            self.curr_state = self.S_const_vel@self.curr_state + self.v_const_accel

    def get_heart_shaped_target(self):
        vel = 0.2 # [m/s]
        period = 7 # [s]
        delta_theta = -2*np.pi*self.delta_t/period # direction change per timestep
        radius = vel*period/2/np.pi # circular radius
        S_circular_clockwise = self.S_const_vel.copy()
        S_circular_clockwise[2:, 2:] = np.array([[np.cos(delta_theta), -np.sin(delta_theta)],
                                                 [np.sin(delta_theta),  np.cos(delta_theta)]])
        
        if self.curr_state is None or self.count > 0:
            self.count -=1
            self.curr_state = self.initial_state_heart
        else:
            direction = np.arctan2(self.curr_state[3], self.curr_state[2])
            # self.get_logger().info('Direction:' + str(np.rad2deg(direction)))
            # self.get_logger().info('state:' + str(self.curr_state))
            if -3*np.pi/4 < direction and direction<= np.pi/2 and (self.curr_state[0]-self.initial_state_heart[0])>=0:
                self.get_logger().info('Circular top right.')
                self.curr_state = S_circular_clockwise@self.curr_state
            elif direction <= -3*np.pi/4:
                self.get_logger().info('Straight bottom right.')
                self.curr_state = self.S_const_vel@self.curr_state
                if (self.curr_state[0]-self.initial_state_heart[0]) < 0:
                    self.get_logger().info('Change velocity direction at bottom')
                    vy = self.curr_state[3][0]
                    self.curr_state -= np.array([0, 0, 0, 2*vy]).reshape((4, 1))
            elif direction >= 3*np.pi/4:
                if (self.curr_state[0]-self.initial_state_heart[0]) >= -((1+np.sqrt(2)/2)*radius):
                    self.get_logger().info('Straight bottom left.')
                    self.curr_state = self.S_const_vel@self.curr_state
                else:
                    self.get_logger().info('Circular top left.')
                    self.curr_state = S_circular_clockwise@self.curr_state
            elif direction < 3*np.pi/4 and (self.curr_state[0]-self.initial_state_heart[0])<=0:
                if (self.curr_state[0]-self.initial_state_heart[0]) <= -radius or (self.curr_state[1]-self.initial_state_heart[1]) >= 0:
                    self.get_logger().info('Circular top left.')
                    self.curr_state = S_circular_clockwise@self.curr_state
                else:
                    self.get_logger().info('Reach initial position. Stay.')
                    self.curr_state[2:] -= self.curr_state[2:]
            elif self.curr_state[2]==0 and self.curr_state[3]==0:
                self.get_logger().info('Reach initial position. Stay.')

    def get_square_target(self, velocity, add_noise = add_noise):
        if self.curr_state is None:
            self.curr_state = self.initial_state_square
        if self.motion == MotionIndex.stop:
            self.set_velocity([0, 0])
        if self.motion == MotionIndex.forward:
            self.set_velocity([0, velocity])
        if self.motion == MotionIndex.right:
            self.set_velocity([velocity, 0])
        if self.motion == MotionIndex.backward:
            self.set_velocity([0, -velocity])
        if self.motion == MotionIndex.left:
            self.set_velocity([-velocity, 0])
        self.get_constant_velocity_target(add_noise)
        if self.motion == MotionIndex.forward and self.curr_state[1] - self.initial_state_square[1] > self.forwardMax:
            self.motion = MotionIndex.right
        if self.motion == MotionIndex.right and self.curr_state[0] - self.initial_state_square[0] > self.sideMax:
            self.motion = MotionIndex.backward
        if self.motion == MotionIndex.backward and self.curr_state[1] - self.initial_state_square[1] < 0:
            self.motion = MotionIndex.left
        if self.motion == MotionIndex.left and self.curr_state[0] - self.initial_state_square[0] < 0:
            self.motion = MotionIndex.forward
    
    def get_sin_target(self, A, vx):
        if self.curr_state is None:
            self.curr_state = self.initial_state_sin
        # y=Asin(kx)
        if self.t is None:
            self.t = 0
        else:
            self.t += self.delta_t
        self.set_velocity([vx, A*np.sin(np.pi*self.t)])
        self.get_constant_velocity_target()

    def get_square_wave_target(self, A, T):
        if self.curr_state is None:
            self.curr_state = self.initial_state_square_wave
        if self.t is None:
            self.t = 0
        else:
            self.t += self.delta_t
        if (self.t // T) % 4 == 0 or (self.t // T) % 4 == 2:
            self.set_velocity([A/T, 0])
        elif (self.t // T) % 4 == 1:
            self.set_velocity([0, A/T])
        else:
            self.set_velocity([0, -A/T])
        self.get_constant_velocity_target()

    def get_waypoint_following_target(self):
        if self.curr_state is None:
            self.curr_state = np.zeros((4, 1))
        if not self.is_waypoints_smoothed:
            self.preprocess_waypoints()
            self.is_waypoints_smoothed = True
        if self.t is None:
            self.t = 0
            self.prev_t = 0
        else:
            self.prev_t = self.t
            self.t += 1
        if self.t == len(self.waypoints_smoothed):
            self.t = 0
            # velocity estimation
        vx = 1 / self.delta_t * (self.waypoints_smoothed[int(self.t)][0] - self.waypoints_smoothed[int(self.prev_t)][0])
        vy = 1 / self.delta_t * (self.waypoints_smoothed[int(self.t)][1] - self.waypoints_smoothed[int(self.prev_t)][1])

        # vx *= 1.1
        # vy *= 1.1

        # publish the waypoints
        self.curr_state[0] = self.waypoints_smoothed[int(self.t)][0]
        self.curr_state[1] = self.waypoints_smoothed[int(self.t)][1]
        self.curr_state[2] = vx
        self.curr_state[3] = vy

    def preprocess_waypoints(self):
        # smooth the waypoints by spline fitting
        dist = [0]
        for idx in range(1, len(self.waypoints_original)):
            dist.append(np.linalg.norm(self.waypoints_original[idx, :] - self.waypoints_original[idx - 1, :]))
        dist_along = np.cumsum(dist)
        spline, u = scipy.interpolate.splprep(self.waypoints_original.T, u=dist_along, s=self.smoothness_factor)
        interp_d = np.arange(u[0], u[-1], 0.02)
        waypoints_smoothed = scipy.interpolate.splev(interp_d, spline)
        self.waypoints_smoothed = np.array(waypoints_smoothed).T

    def set_velocity(self, v):
        self.curr_state[2] = v[0]
        self.curr_state[3] = v[1]

    def set_state_msg(self, add_noise=add_noise):
        curr_state = self.curr_state.copy()
        self.state = TargetState()
        # self.state.header = std_msgs.msg.Header()
        self.state.header.stamp = self.node.get_clock().now().to_msg()
        

        if self.mode == 1:
            self.state.pose.position.x = float(curr_state[0])
            self.state.pose.position.y = float(self.default_y)
            self.state.pose.position.z = float(curr_state[1])
            self.state.velocity.linear.x = float(curr_state[2])
            self.state.velocity.linear.y = 0.0
            self.state.velocity.linear.z = float(curr_state[3])
        elif self.mode == 2:
            self.state.pose.position.x = float(self.default_x)
            self.state.pose.position.y = float(curr_state[0])
            self.state.pose.position.z = float(curr_state[1])
            self.state.velocity.linear.x = 0.0
            self.state.velocity.linear.y = float(curr_state[2])
            self.state.velocity.linear.z = float(curr_state[3])
        else:
            self.state.pose.position.x = float(curr_state[0])
            self.state.pose.position.y = float(curr_state[1])
            self.state.pose.position.z = float(self.default_z)
            self.state.velocity.linear.x = float(curr_state[2])
            self.state.velocity.linear.y = float(curr_state[3])
            self.state.velocity.linear.z = 0.0
        if add_noise:
            self.state.pose.position.x += (np.random.rand(1)-0.5)*0.05
            self.state.pose.position.y += (np.random.rand(1)-0.5)*0.05
            self.state.pose.position.z += (np.random.rand(1)-0.5)*0.05
            self.state.velocity.linear.x += (np.random.rand(1)-0.5)*0.02
            self.state.velocity.linear.y += (np.random.rand(1)-0.5)*0.02
            self.state.velocity.linear.z += (np.random.rand(1)-0.5)*0.02
        # r = R.from_euler('ZYX', [-np.pi/2, 0, 0])
        r = Rotation.from_euler('ZYX', [0, 0, 0])
        quaternion = r.as_quat()
        self.state.pose.orientation.x = float(quaternion[0])
        self.state.pose.orientation.y = float(quaternion[1])
        self.state.pose.orientation.z = float(quaternion[2])
        self.state.pose.orientation.w = float(quaternion[3])

    def handle_publish_single_target(self, req):
        # publish single target state message by calling service PublishSingleTarget
        self.state = TargetState()
        self.state.pose.position.x = req.x
        self.state.pose.position.y = req.y
        self.state.velocity.linear.x = req.vx
        self.state.velocity.linear.y = req.vy
        succeed = self.publish_state()
        return succeed

    def projection(self, S, bound):
        """
        Project M to the closest(Frobenius norm) matrix whose l2 norm is bounded.
        """
        m,n = S.shape
        U, s, Vh = linalg.svd(S)
        for i in range(len(s)):
            s[i] = min(s[i], bound)
        S_projected = U@linalg.diagsvd(s, m, n)@Vh
        return S_projected
        

def main(args=None):
    np.random.seed(10)
    target_state_estimator = TargetStateEstimator()
    target_state_estimator.motion = MotionIndex.forward

    

if __name__ == '__main__':
    main()
