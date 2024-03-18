#!/usr/bin/env python3
import rospy
import numpy as np
import time
import yaml
import os
from scipy import linalg
from crazyflie_online_tracker.msg import CommandOuter
from controller import Controller, ControllerStates

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
gamma = data['gamma_RLS']
mode = data['mode']
target = data['target']
delta_t = float(1/f)
filtering = data['filtering']
process_noise_var = data['process_noise_var']
measurement_noise_var = data['measurement_noise_var']


class RLSController(Controller):
    '''
    This is the proposed PLOT controller with an affine term.
    '''
    def __init__(self):
        super().__init__()
        self.m = m
        self.gamma = gamma # forgetting factor
        self.W = W # prediction horizon
        self.Df = np.inf

        # Tuning gamma based on Corollary 4.3
        self.T = T*f
        self.c1 = 1.34

        # Optimal
        # self.gamma = 1 - self.c1 *self.T**(-0.25) #0.8

        # self.gamma = 1 - self.c1 * self.T ** (-0.1) # 0.37
        # self.gamma = 1 - self.c1 * self.T ** -(0.2) #0.71
        # self.gamma = 1 - self.c1 * self.T ** (-0.30)  # 0.86
        # self.gamma = 1 - self.c1 * self.T ** (-0.5) # 0.97
        self.gamma = 1 - self.c1 * self.T ** (-1) # 1.0

        # initialize target dynamics estimation
        self.idx_of_interest = [0, 1, 2, 3, 4, 5] # the indices of target states that can be measured and learned. In our case is [x,y,z,vx,vy,vz]
        self.n_of_interest = len(self.idx_of_interest)
        # S_target= np.eye(self.n_of_interest) # r_t=[x, y, z, vx, vy, vz]'
        S_target = np.ones((self.n_of_interest, self.n_of_interest))*1e-5  # r_t=[x, y, z, vx, vy, vz]'
        v_target = np.zeros((self.n_of_interest, 1))
        S_target_aug = np.vstack((np.hstack((S_target, v_target)), np.hstack((np.zeros((1, self.n_of_interest)), np.ones((1, 1))))))

        self.S_target_aug_all = [[S_target_aug.copy()]*self.W for _ in range(self.W)]
        

        P = np.eye(self.n_of_interest+1) # P_t = \gamma*P_{t-1} + r_{t-1}r_{t-1}'
        self.P_all = [[P.copy()]*self.W for _ in range(self.W)]

        self.pred_log = []
        self.error_log = []

        # precompute LQR gains
        self.compute_A_tilde_power()
        self.solve_M_optimal_all()

        # initialize target position in case it is not detected by the camera.
        self.desired_pos = np.array([0, 0.65, 0.3])

        self.disturbances_predicted = []

        rospy.init_node('RLS_controller')

    def get_new_states(self):
        '''
        Read the latest target and drone state measurements and compute the disutbance accordingly.
        '''
        drone_state = self.drone_state_raw_log[-1]
        target_state = self.target_state_raw_log[-1]
        if rospy.get_param('synchronize_target'):
            # this block of code somehow helps to compensate for the differences of the ros node initialization times
            # e.g. for SOME horizon lengths, the target states used for estimation is [0.3,0.6,0.9,...], while for others it is [0.3,0.3,0.6,0.9,...]
            # this impacts the regret comparison of controllers with different W tracking the same target
            # only switch it on when necessary.
            while len(self.target_state_log) > 0 and np.all(target_state == self.target_state_log[-1]):
                rospy.sleep(0.05)
                target_state = self.target_state_raw_log[-1]
        self.drone_state_log.append(drone_state)
        self.target_state_log.append(target_state)
        # rospy.loginfo("current time: " + str(self.t))
        # rospy.loginfo("current state[RLS]: " + str(drone_state))
        # rospy.loginfo("observe target[RLS]:" + str(target_state))

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
                # rospy.loginfo('predicted target based on r_{t-' + str(k)+'}: '+str(pred_state))
        # rospy.loginfo('prediction error: '+str(error))
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
                curr_target_state_aug = np.vstack((curr_target_state, 1))
                last_target_state = self.target_state_log[-k-1][self.idx_of_interest]
                last_target_state_aug = np.vstack((last_target_state, 1))

                self.P_all[k-1][learner_idx] = self.gamma*self.P_all[k-1][learner_idx] + last_target_state_aug@last_target_state_aug.T
                P_inv = np.linalg.inv(self.P_all[k-1][learner_idx]+01e-5*np.eye(1+self.n_of_interest))
                self.S_target_aug_all[k-1][learner_idx] = self.S_target_aug_all[k-1][learner_idx] \
                    + (curr_target_state_aug - self.S_target_aug_all[k-1][learner_idx]@last_target_state_aug)@last_target_state_aug.T@P_inv
                # breakpoint()
                self.S_target_aug_all[k-1][learner_idx] = self.projection(self.S_target_aug_all[k-1][learner_idx], self.Df)

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
            v_target_k_step = self.S_target_aug_all[k-1][learner_idx][:self.n_of_interest, self.n_of_interest].reshape((self.n_of_interest, 1))
            pred_target_state = S_target_k_step@curr_target_state + v_target_k_step
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
        for i in range(7):
            s[i] = min(s[i], bound)
        M_projected = U @ linalg.diagsvd(s, 7, 7) @ Vh
        return M_projected

    def compute_setpoint(self):
        if self.controller_state == ControllerStates.normal:
            # rospy.loginfo("controller state: normal")
            drone_state = self.drone_state_log[-1]
            target_state = self.target_state_log[-1]
            error = drone_state - target_state
            [thrust, roll_rate, pitch_rate, yaw_rate] = self.compute_setpoint_viaLQR(self.K_star,error, drone_state[8])
            # rospy.loginfo('optimal yaw_rate: '+str(yaw_rate))
            error_feedback = np.array([thrust, pitch_rate, roll_rate, yaw_rate])
            self.predict_future_targets()
            future_disturbance_feedback = np.zeros((4,1))
            for i in range(self.W):
                future_disturbance_feedback -= self.M_optimal_all[i]@self.disturbances_predicted[i]
                # rospy.loginfo('future_disturbance '+str(i)+ ' : '+str(self.disturbances_predicted[i]))
                # rospy.loginfo('future_disturbance_feedback '+str(i)+ ' : '+str(self.M_optimal_all[i]@self.disturbances_predicted[i]))
            roll_rate = future_disturbance_feedback[1].copy()
            pitch_rate = future_disturbance_feedback[2].copy()
            future_disturbance_feedback[1] = pitch_rate
            future_disturbance_feedback[2] = roll_rate

            action = error_feedback + future_disturbance_feedback
            self.action_DF_log.append(future_disturbance_feedback)
            self.action_log.append(action)
            # rospy.loginfo('optimal action[RLS]: '+str(action))
            # convert command from numpy array to ros message
            setpoint = CommandOuter()
            setpoint.thrust = action[0]
            setpoint.omega.x = action[1] # pitch rate
            setpoint.omega.y = action[2] # roll rate
            setpoint.omega.z = action[3] # yaw rate
            # rospy.loginfo(error_feedback)
        elif self.controller_state == ControllerStates.takeoff:
            rospy.loginfo("controller state: takeoff")
            drone_state = self.drone_state_raw_log[-1]
            setpoint = self.takeoff(drone_state)
        elif self.controller_state == ControllerStates.landing:
            rospy.loginfo("controller state: landing")
            drone_state = self.drone_state_raw_log[-1]
            setpoint = self.landing(drone_state)
        self.setpoint = setpoint


    def publish_setpoint(self, is_last_command=False):
        if is_last_command:
            self.setpoint = CommandOuter()
            self.setpoint.is_last_command = True
        else:
            self.compute_setpoint()
        self.controller_command_pub.publish(self.setpoint)

if __name__ == '__main__':
    RLS_controller = RLSController()
    wait_for_simulator_initialization = rospy.get_param('wait_for_simulator_initialization')
    add_initial_target = rospy.get_param('add_initial_target')
    total_time = 0
    count = 5
    rate = rospy.Rate(f)
    rospy.sleep(2)

    # Set to True to save data for post-processing
    save_log = True

    # Used for timing the algorithms
    Time = 0
    Time_T = 0

    # Set to True to generate a plot immediately
    plot = True
    while not (rospy.is_shutdown() or RLS_controller.controller_state == ControllerStates.stop):
        ready = False
        if add_initial_target and len(RLS_controller.target_state_raw_log)==0:
            initial_target = np.zeros((9, 1))
            initial_target[:3] = RLS_controller.desired_pos.reshape((3, 1))
            RLS_controller.target_state_raw_log.append(initial_target)
        if len(RLS_controller.drone_state_raw_log)>0 and \
            len(RLS_controller.target_state_raw_log)>0:
            ready = True
        if RLS_controller.t <= T:
            if ready:
                t0 = time.time()
                if RLS_controller.controller_state == ControllerStates.normal:
                    RLS_controller.get_new_states()
                    RLS_controller.RLS_update()
                RLS_controller.publish_setpoint()
                t1 = time.time()
                Time += (t1-t0)/(T*f)
                RLS_controller.t += RLS_controller.delta_t
                if wait_for_simulator_initialization:
                    rospy.sleep(4)
                    count -= 1
                    if count < 0:
                        wait_for_simulator_initialization = False
            else:
                rospy.loginfo('No drone or target state estimation is available. Skipping.')
        else:
            if RLS_controller.controller_state == ControllerStates.normal:
                RLS_controller.publish_setpoint(is_last_command=True)
                rospy.loginfo('Simulation finished.')
            else:
                RLS_controller.publish_setpoint()
        rate.sleep()


    if RLS_controller.controller_state == ControllerStates.stop:
        rospy.loginfo('controller state is set to STOP. Terminating.')

        if save_log:
            filename = rospy.get_param('filename')
            additional_info = f"_{target}_T{T}_f{f}_gam{round(RLS_controller.gamma,2)}_W{W}_mode{mode}"
            # additional_info = f"_{target}_T{T}_f{f}_mode{mode}"
            new_filename = filename + additional_info
            RLS_controller.save_data(new_filename)
            time.sleep(2)
            if plot:
               rospy.loginfo('Time: ' + str(Time))
               os.system("rosrun crazyflie_online_tracker plot.py")


