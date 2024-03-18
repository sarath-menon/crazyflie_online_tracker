#!/usr/bin/env python3
import rospy
import numpy as np
import time
import yaml
import os
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.spatial.transform import Rotation as R
from controller import Controller
from crazyflie_online_tracker_interfaces.msg import CommandOuter, TargetState
import sys
import glob

# Load data from the YAML file
yaml_path = os.path.join(os.path.dirname(__file__), '../param/data.yaml')

with open(yaml_path, 'r') as file:
    data = yaml.safe_load(file)

# Extract the required variables
m = data['m']
g = data['g']
f = data['f']
T = data['T']
path = data['path']
#target = data['target']

class OptimalOfflineController(Controller):
    """
    The optimal offline controller reads the target trajectory from the log file(defined in data/<filename>.npz) 
    and compute the optimal action with respect to the target trajectory.
    """
    def __init__(self, filename) -> None:
        super().__init__()
        succeed = self.load_data(filename) # load target states to compute the optimal policy. 
                                 # load drone states and actions of the algorithm to compute its dynamic regret.
        if not succeed:
            rospy.loginfo('Failed to load data.')

        self.idx_total = len(self.action_algorithm_log)
        self.idx_curr = 0

        self.compute_LQR_gain()
        self.compute_disturbances()

        self.cost_algorithm_log = []
        self.cost_optimal_log = []
        self.error_optimal_log = []
        self.error_algorithm_log = []

        # Initialize controller node
        self.target_state_pub = rospy.Publisher('targetState', TargetState, queue_size=10, latch=True) # publish target state for matlab plotting
        rospy.init_node('optimal_offline_controller')
        
    def compute_LQR_gain(self):
        self.K_d_all = []
        A_tilde_power = np.eye(9)
        self.A_tilde_power = [A_tilde_power]
        for _ in range(self.idx_total):
            K_d = linalg.inv(self.R + self.B.T @ self.P_star @ self.B)@self.B.T@A_tilde_power.T@self.P_star
            self.K_d_all.append(K_d)
            A_tilde_power = (self.A - self.B@self.K_star)@A_tilde_power
            self.A_tilde_power.append(A_tilde_power)

    def compute_disturbances(self):
        self.disturbances = []
        for idx in range(self.idx_total - 1):
            disturbance = self.A@self.target_state_log[idx] - self.target_state_log[idx+1]
            self.disturbances.append(disturbance)
        
    def compute_setpoint(self):
        # compute the optimal action for the optimal policy trajectory
        drone_state = self.drone_state_raw_log[-1]
        self.drone_state_log.append(drone_state)
        error = drone_state - self.target_state_log[self.idx_curr]
        [thrust, roll_rate, pitch_rate, yaw_rate] = self.compute_setpoint_viaLQR(self.K_star,error, drone_state[8])
        error_feedback = np.array([thrust, pitch_rate, roll_rate, yaw_rate])

        future_disturbance_feedback = np.zeros((4,1))
        for idx in range(self.idx_curr, self.idx_total-1):
            future_disturbance_feedback -= self.K_d_all[idx-self.idx_curr]@self.disturbances[idx]
        roll_rate = future_disturbance_feedback[1].copy()
        pitch_rate = future_disturbance_feedback[2].copy()
        future_disturbance_feedback[1]=pitch_rate
        future_disturbance_feedback[2]=roll_rate
        action = error_feedback + future_disturbance_feedback
        self.action_DF_log.append(future_disturbance_feedback)
        self.action_log.append(action)
        rospy.loginfo('optimal action: '+str(action))
        # convert command from numpy array to ros message
        setpoint = CommandOuter()
        setpoint.thrust = action[0]
        setpoint.omega.x = action[1] # pitch rate
        setpoint.omega.y = action[2] # roll rate
        setpoint.omega.z = action[3] # yaw rate
        self.setpoint = setpoint
        # compute the optimal action in hindsight for the algorithm trajectory
        drone_state_algorithm = self.drone_state_algorithm_log[self.idx_curr]
        error_algorithm = drone_state_algorithm - self.target_state_log[self.idx_curr]
        [thrust, roll_rate, pitch_rate, yaw_rate] = self.compute_setpoint_viaLQR(self.K_star,error_algorithm, drone_state[8])
        error_feedback_algorithm = np.array([thrust, roll_rate, pitch_rate, yaw_rate])
        action_algorithm = error_feedback_algorithm + future_disturbance_feedback
        action_algorithm[0] = action_algorithm[0]
        roll_rate_algorithm = action_algorithm[1].copy()
        pitch_rate_algorithm= action_algorithm[2].copy()
        action_algorithm[1] = pitch_rate_algorithm
        action_algorithm[2] = roll_rate_algorithm
        self.optimal_action_log.append(action_algorithm)
        rospy.loginfo('optimal action for algorithm trajectory: '+str(action_algorithm))

    def publish_setpoint(self, is_last_command=False):
        if is_last_command:
            self.setpoint = CommandOuter()
            self.setpoint.is_last_command = True
        else:
            self.compute_setpoint()
        self.controller_command_pub.publish(self.setpoint)

    def publish_target(self):
        state = TargetState()
        state.pose.position.x = self.target_state_log[self.idx_curr, 0]
        state.pose.position.y = self.target_state_log[self.idx_curr, 1]
        state.pose.position.z = self.target_state_log[self.idx_curr, 2]
        state.pose.orientation.w = 1
        state.velocity.linear.x = self.target_state_log[self.idx_curr, 3]
        state.velocity.linear.y = self.target_state_log[self.idx_curr, 4]
        state.velocity.linear.z = self.target_state_log[self.idx_curr, 5]
        self.target_state_pub.publish(state)

    def compute_stage_cost(self):
        # cost incurred by the online algorithm
        error_algorithm = self.drone_state_algorithm_log[self.idx_curr] - self.target_state_log[self.idx_curr]
        action_algorithm = self.action_algorithm_log[self.idx_curr]
        cost_algorithm = error_algorithm.T@self.Q@error_algorithm + action_algorithm.T@self.R@action_algorithm
        self.cost_algorithm_log.append(cost_algorithm)
        self.error_algorithm_log.append(error_algorithm)

        # cost incurred by the optimal offline policy
        error_optimal = self.drone_state_log[self.idx_curr] - self.target_state_log[self.idx_curr]
        action_optimal = self.action_log[self.idx_curr]
        cost_optimal = error_optimal.T@self.Q@error_optimal + action_optimal.T@self.R@action_optimal
        self.cost_optimal_log.append(cost_optimal)
        self.error_optimal_log.append(error_optimal)    
    

if __name__ == '__main__':

    save_path = os.path.expanduser(path + 'data')

    args = sys.argv
    list_of_experiments = glob.glob(save_path + '/*')
    latest_experiment = max(list_of_experiments, key=os.path.getctime)
    filename = latest_experiment.replace(save_path + '/', '')

    optimal_offline_controller = OptimalOfflineController(filename+'.npz')
    #T = 50 #500 #44 from yaml
    if T > (optimal_offline_controller.idx_total-1) * 0.1:
        T = (optimal_offline_controller.idx_total-1) * 0.1
    #publish_frequency = rospy.get_param('publish_frequency')
    wait_for_simulator_initialization = rospy.get_param('wait_for_simulator_initialization')
    #rate = rospy.Rate(publish_frequency)
    rate = rospy.Rate(100)
    count = 5
    while not rospy.is_shutdown():
        if len(optimal_offline_controller.drone_state_raw_log) > 0:
            optimal_offline_controller.publish_target()
            optimal_offline_controller.publish_setpoint()
            optimal_offline_controller.compute_stage_cost()
            optimal_offline_controller.t += optimal_offline_controller.delta_t
            optimal_offline_controller.idx_curr += 1
            if wait_for_simulator_initialization:
                rospy.sleep(4)
                count -= 1
                if count < 0:
                    wait_for_simulator_initialization = False
        rate.sleep()
        if optimal_offline_controller.t > T:
            cost_algorithm = np.array(optimal_offline_controller.cost_algorithm_log).sum()
            cost_optimal = np.array(optimal_offline_controller.cost_optimal_log).sum()
            regret = cost_algorithm - cost_optimal
            rospy.loginfo('The algorithm cost is' + str(cost_algorithm))
            rospy.loginfo('The optimal cost is' + str(cost_optimal))
            rospy.loginfo('The dynamic regret is' + str(regret))
            additional_info = f"_OPTIMAL_f{f}"
            new_filename = filename + additional_info
            optimal_offline_controller.save_data(new_filename)
            rospy.sleep(2)
            os.system("rosrun crazyflie_online_tracker plot.py")
            break
