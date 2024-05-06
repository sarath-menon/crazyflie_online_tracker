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

# Extract the required variables
m = data['m']
g = data['g']
f = data['f']
T = data['T']
W = data['W_RLS']
is_sim = data['is_sim']
gamma = data['gamma_RLS']
mode = data['mode']
target = data['target']
delta_t = float(1/f)
filtering = data['filtering']
process_noise_var = data['process_noise_var']
measurement_noise_var = data['measurement_noise_var']


class RLSController(Controller):
    '''
    This is the proposed PLOT controller without an affine term.
    '''
    def __init__(self):
        super().__init__()

        # self.controller_state_sub = self.node.create_subscription(ControllerState, 'controllerState', self.callback_controller_state, 10)
        # self.controller_command_pub = self.node.create_publisher(FullState, '/cf231/cmd_vel', 10)
        # self.controller_state_pub = self.node.create_publisher(ControllerState, 'controllerStateKF', 10)

        # self.drone_state_sub = self.node.create_subscription(CrazyflieState, 'crazyflieState', self.callback_state_drone, 10)
        # self.target_state_sub = self.node.create_subscription(TargetState, 'targetState', self.callback_state_target, 10)

        # self.clock_sub = self.node.create_subscription(Clock, 'clock', self.timer_callback, 10)

        # # declare params
        # self.node.declare_parameter('filename', 'Filename')
        # self.node.declare_parameter('wait_for_drone_ready', False)

        # # get params
        # self.filename = self.node.get_parameter('filename')

        # YAML params
        self.m = m
        self.gamma = gamma # forgetting factor
        self.W = W # prediction horizon
        self.Df = np.inf


        # initialize target dynamics estimation
        self.idx_of_interest = [0, 1, 2, 3, 4, 5] # the indices of target states that can be measured and learned. In our case is [x,y,z,vx,vy,vz]
        self.n_of_interest = len(self.idx_of_interest)
        S_target= np.eye(self.n_of_interest) # r_t=[x, y, z, vx, vy, vz]'
        S_target_aug = S_target

        self.S_target_aug_all = [[S_target_aug.copy()]*self.W for _ in range(self.W)]


        P = 01e-5*np.eye(self.n_of_interest) # P_t = \gamma*P_{t-1} + r_{t-1}r_{t-1}'
        self.P_all = [[P.copy()]*self.W for _ in range(self.W)]

        self.pred_log = []
        self.error_log = []

        # precompute LQR gains
        self.compute_A_tilde_power()
        self.solve_M_optimal_all()

        # # initialize target position in case it is not detected by the camera.
        # self.desired_pos = np.array([0, 0.65, 0.3])

        # self.disturbances_predicted = []

         # declare params
        # self.node.declare_parameter('add_initial_target', False)
        self.node.declare_parameter('synchronize_target', False)
        # self.node.declare_parameter('filename', 'Filename')

        # get params
        # self.add_initial_target = self.node.get_parameter('add_initial_target')
        self.add_initial_target = self.node.get_parameter('synchronize_target')
        # self.add_initial_target = self.node.get_parameter('filename')

        # # timer calbacks
        # timer_period = 0.5  # seconds
        # self.timer = self.node.create_timer(timer_period, self.timer_callback)

        # Set to True to save data for post-processing
        self.save_log = True

        # Set to True to generate a plot immediately
        self.plot = True

         # takeoff drone
        self.initial_position = np.array([0, 0, 0])
        self.hover_position = np.array([0, 0, 0.4])

        self.takeoff_manual()
        
        rclpy.spin(self.node)
        # node.destroy_node()
        # rclpy.shutdown()

    # def timer_callback(self):

    #     if self.controller_state != ControllerStates.stop:
    #         ready = False

    #         self.node.get_logger().info(f"Length of drone state log: {self.drone_state_raw_log.qsize()}")
    #         self.node.get_logger().info(f"Length of target state log: {len(self.target_state_raw_log)}")

    #         # Check if initial target is to be added and if the target state log is empty
    #         if self.add_initial_target and len(self.target_state_raw_log)==0:
    #             initial_target = np.zeros((9, 1))
    #             initial_target[:3] = self.desired_pos.reshape((3, 1))
    #             self.target_state_raw_log.append(initial_target)

    #         # Check if both drone state log and target state log are not empty
    #         if self.drone_state_raw_log.qsize()>0 and \
    #             len(self.target_state_raw_log)>0:
    #             ready = True
                
    #         # Check if current time is less than or equal to T
    #         if self.t <= T:
    #             if ready:
    #                 # Check if the controller state is normal
    #                 if self.controller_state == ControllerStates.normal:
    #                     self.get_new_states()
    #                     self.RLS_update()
    #                 self.publish_setpoint()
    #                 self.t += self.delta_t

    #             else:
    #                 self.node.get_logger().info('No drone or target state estimation is available. Skipping.')
    #         else:
    #             # Check if the controller state is normal
    #             if self.controller_state == ControllerStates.normal:
    #                 self.publish_setpoint(is_last_command=True)
    #                 self.node.get_logger().info('Simulation finished.')
    #             else:
    #                 self.publish_setpoint()


    #     elif self.controller_state == ControllerStates.stop:
    #         self.node.get_logger().info('controller state is set to STOP. Terminating.')
    #         if self.save_log:
    #             filename = self.node.get_parameter('filename')
    #             additional_info = f"_{target}_T{T}_f{f}_gam{round(self.gamma,2)}_W{W}_mode{mode}"
    #             new_filename = filename.get_parameter_value().string_value + additional_info
    #             self.save_data(new_filename)
    #             time.sleep(2)
    #             if self.plot:
    #                os.system("ros2 run crazyflie_online_tracker plot.py")


    

    def compute_setpoint(self):

        self.get_new_states()
        self.RLS_update()


        drone_state = self.drone_state_log[-1]
        target_state = self.target_state_log[-1]
        

        error = drone_state - target_state
        [thrust, roll_rate, pitch_rate, yaw_rate] = self.compute_setpoint_viaLQR(self.K_star,error, drone_state[8])
        error_feedback = np.array([thrust, pitch_rate, roll_rate, yaw_rate])
        self.predict_future_targets()
        future_disturbance_feedback = np.zeros((4,1))

        for i in range(self.W):
            future_disturbance_feedback -= self.M_optimal_all[i]@self.disturbances_predicted[i]
            # self.node.get_logger().info('future_disturbance '+str(i)+ ' : '+str(self.disturbances_predicted[i]))
            # self.node.get_logger().info('future_disturbance_feedback '+str(i)+ ' : '+str(self.M_optimal_all[i]@self.disturbances_predicted[i]))
        
        roll_rate = future_disturbance_feedback[1].copy()
        pitch_rate = future_disturbance_feedback[2].copy()


        future_disturbance_feedback[1] = pitch_rate
        future_disturbance_feedback[2] = roll_rate

        action = error_feedback + future_disturbance_feedback
        self.action_DF_log.append(future_disturbance_feedback)
        self.action_log.append(action)


        # self.node.get_logger().info('optimal action[RLS]: '+str(action))
        # convert command from numpy array to ros message
        setpoint = FullState()

        setpoint.acc.z = float(action[0])
        setpoint.twist.angular.x = float(action[1]) # pitch rate
        setpoint.twist.angular.y = float(action[2]) # roll rate
        setpoint.twist.angular.z = float(action[3]) # yaw rate

        self.setpoint = setpoint

    def get_new_states(self):
        '''
        Read the latest target and drone state measurements and compute the disutbance accordingly.
        '''
        drone_state = self.drone_state_raw_log[-1]
        target_state = self.target_state_raw_log[-1]

        if self.node.get_parameter('synchronize_target').get_parameter_value().bool_value:
            # this block of code somehow helps to compensate for the differences of the ros node initialization times
            # e.g. for SOME horizon lengths, the target states used for estimation is [0.3,0.6,0.9,...], while for others it is [0.3,0.3,0.6,0.9,...]
            # this impacts the regret comparison of controllers with different W tracking the same target
            # only switch it on when necessary.
            while len(self.target_state_log) > 0 and np.all(target_state == self.target_state_log[-1]):
                rclpy.sleep(0.05)
                target_state = self.target_state_raw_log[-1]
        self.drone_state_log.append(drone_state)
        self.target_state_log.append(target_state)
        self.node.get_logger().info("current time: " + str(self.t))
        # self.node.get_logger().info("current state[RLS]: " + str(drone_state))
        # self.node.get_logger().info("observe target[RLS]:" + str(target_state))

        # compute disturbance w_t according to the latest drone and target state estimation
        if len(self.target_state_log) < 2:
            return
        else:
            last_target = self.target_state_log[-2]  # r_t
            curr_target = self.target_state_log[-1]  # r_{t+1}
        # w_t = Ar_t - r_{t+1}
        disturbance = self.A @ last_target - curr_target
        self.disturbances.append(disturbance)

        # compute prediction error for debugging
        error = [np.zeros((self.n_of_interest, 1))] * self.W
        for k in range(1, self.W + 1):
            if len(self.pred_log) >= k:
                pred_state = self.pred_log[-k][k - 1]
                error[k - 1] = pred_state - target_state[self.idx_of_interest]
                # self.node.get_logger().info('predicted target based on r_{t-' + str(k)+'}: '+str(pred_state))
        # self.node.get_logger().info('prediction error: '+str(error))
        self.error_log.append(error)

    def RLS_update(self):
        '''
        Update the augmented target dynamic estimation.
        A small regulation factor is added to avoid the singularity issue in inverting P.
        '''
        timesteps = len(self.target_state_log)
        for k in range(1, self.W+1): # update k-step-ahead prediction
            learner_idx = (timesteps - 1) % k
            if timesteps > k:
                curr_target_state = self.target_state_log[-1][self.idx_of_interest]
                curr_target_state_aug = curr_target_state
                last_target_state = self.target_state_log[-k-1][self.idx_of_interest]
                last_target_state_aug = last_target_state

                self.P_all[k-1][learner_idx] = self.gamma*self.P_all[k-1][learner_idx] + last_target_state_aug@last_target_state_aug.T
                P_inv = np.linalg.inv(self.P_all[k-1][learner_idx]+10*np.eye(self.n_of_interest))
                self.S_target_aug_all[k-1][learner_idx] = self.S_target_aug_all[k-1][learner_idx] \
                    + (curr_target_state_aug - self.S_target_aug_all[k-1][learner_idx]@last_target_state_aug)@last_target_state_aug.T@P_inv

    def predict_future_targets(self):
        '''
        Predict the future targets and compute the corresponding disturbances up to horizon W.
        Note that only the indices of interest are predicted, other indices are set to default values.
        '''
        timesteps = len(self.target_state_log)
        pred_target_state_all = []
        curr_target_state = self.target_state_log[-1][self.idx_of_interest]
        last_pred_target_state_full = self.target_state_log[-1]
        self.disturbances_predicted = []
        for k in range(1, self.W+1):
            learner_idx = (timesteps - 1) % k
            S_target_k_step = self.S_target_aug_all[k-1][learner_idx][:self.n_of_interest, :self.n_of_interest]
            pred_target_state = S_target_k_step@curr_target_state
            pred_target_state_all.append(pred_target_state)
            pred_target_state_full = np.zeros((9, 1))
            pred_target_state_full[self.idx_of_interest] = pred_target_state
            disturbance = self.A@last_pred_target_state_full - pred_target_state_full
            self.disturbances_predicted.append(disturbance)
            last_pred_target_state_full = pred_target_state_full
        self.pred_log.append(pred_target_state_all)

    def compute_A_tilde_power(self):
        """
        Precompute (A-BK)^i, i=0,...,H
        """
        self.A_tilde_power = [np.eye(9)]
        for _ in range(1, self.W):
            new = (self.A - self.B@self.K_star)@self.A_tilde_power[-1]
            self.A_tilde_power.append(new)

    def solve_M_optimal_all(self):
        '''
        compute optimal disturbance-feedback gains
        '''
        self.M_optimal_all = []
        inv = linalg.inv(self.R + self.B.T @ self.P_star @ self.B)
        for i in range(self.W):
            M_optimal_i = inv@self.B.T@self.A_tilde_power[i].T@self.P_star
            self.M_optimal_all.append(M_optimal_i)

    def projection(self, M, bound):
        """
        Project M to the closest(Frobenius norm) matrix whose l2 norm is bounded.
        """
        U, s, Vh = linalg.svd(M)
        for i in range(4):
            s[i] = min(s[i], bound)
        M_projected = U @ linalg.diagsvd(s, 7, 7) @ Vh
        return M_projected

        self.setpoint = setpoint

    # def save_data(self, filename):
    #     now = datetime.now()
    #     timestr = now.strftime("%Y%m%d%H%M%S")
    #     filename = timestr + '_' + filename
    #     # depending on filtering the drone_state_log will be populated from the filtered raw log or from
    #     # the unfiltered one
    #     if filtering: # if we are in the simulation we need to save the generated system output
    #         if is_sim:
    #             np.savez(os.path.join(self.save_path, filename),
    #                     target_state_log = np.array(self.target_state_log),
    #                     drone_state_log= np.array(self.drone_state_log),
    #                     action_log = np.array(self.action_log),
    #                     action_DF_log = np.array(self.action_DF_log),
    #                     disturbance_log = np.array(self.disturbances),
    #                     default_action_log = np.array(self.default_action_log),
    #                     optimal_action_log = np.array(self.optimal_action_log),
    #                     filtered_drone_state_log = np.array(self.filtered_drone_state_log),
    #                     system_output=np.array(self.system_output_raw_log))
    #         else:
    #             np.savez(os.path.join(self.save_path, filename),
    #                      target_state_log=np.array(self.target_state_log),
    #                      drone_state_log=np.array(self.drone_state_log),
    #                      action_log=np.array(self.action_log),
    #                      action_DF_log=np.array(self.action_DF_log),
    #                      disturbance_log=np.array(self.disturbances),
    #                      default_action_log=np.array(self.default_action_log),
    #                      optimal_action_log=np.array(self.optimal_action_log),
    #                      filtered_drone_state_log=np.array(self.filtered_drone_state_log))
    #     else:
    #         np.savez(os.path.join(self.save_path, filename),
    #                  target_state_log=np.array(self.target_state_log),
    #                  drone_state_log=np.array(self.drone_state_log),
    #                  action_log=np.array(self.action_log),
    #                  action_DF_log=np.array(self.action_DF_log),
    #                  disturbance_log=np.array(self.disturbances),
    #                  default_action_log=np.array(self.default_action_log),
    #                  optimal_action_log=np.array(self.optimal_action_log),
    #                  learnt_S_log = np.array(self.S_target_aug_all))
    #     self.node.get_logger().info("Trajectory data has been saved to" + self.save_path + '/' + filename + '.npz')

def main(args=None):
    controller = RLSController()


   



if __name__ == '__main__':
    main()