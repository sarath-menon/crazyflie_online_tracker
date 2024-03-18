#!/usr/bin/env python3
import rospy
import numpy as np
from scipy import linalg
import time
import yaml
import os

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
target = data['target']
delta_t = float(1/f)

class RiccatitronController(Controller):
    '''
    This is the Riccatitron controller proposed in 'Logarithmic Regret for Adversarial Online Control'.
    '''
    def __init__(self):
        super().__init__()
        self.m = m
        self.gamma = 1 # forgetting factor
        self.M = 5 # DAP length, m in paper
        self.h = 5 # horizon length, h in paper
        self.eta = 0.2 #learning rate
        self.radius = np.inf # radius of M
        self.decay = 0.2# decay parameter of M
        
        # precompute LQR gains
        self.compute_A_tilde_power()
        self.solve_M_optimal_all()

        # initialize base learners
        self.BL_all = [[np.zeros((4, 9))]*self.M for _ in range(self.h+1)]
        self.E_all = [[np.eye(4)]*self.M for _ in range(self.h+1)]
        self.q_log = []

        rospy.init_node('Riccatitron_controller')

    def get_new_states(self):
        '''
        Read the latest target and drone state measurements and compute the disturbance accordingly.
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
        # rospy.loginfo("current state[Riccatitron]: "+str(drone_state))
        # rospy.loginfo("observe target[Riccatitron]:" + str(target_state))
        
        # compute disturbance w_t according to the latest drone and target state estimation
        if len(self.target_state_log)<2:
            return
        else:
            last_target = self.target_state_log[-2] # r_t
            curr_target = self.target_state_log[-1] # r_{t+1}
        # w_t = Ar_t - r_{t+1}
        disturbance = self.A@last_target - curr_target
        self.disturbances.append(disturbance)

    def BL_ONS_update(self):
        '''
        update base learners with ONS
        '''
        start = time.time()
        t = len(self.target_state_log)
        if t > self.h + 1:
            tau = int((t-1)%(self.h+1))
            t_last_applied = t-self.h -1
            Sigma = self.R + self.B.T @ self.P_star @ self.B
            q_optimal = self.compute_q_optimal(t_last_applied)
            q_applied = self.q_log[t_last_applied-1]
            # rospy.loginfo('tau: '+str(tau))
            # rospy.loginfo('q_optimal: '+str(q_optimal))
            # rospy.loginfo('q_applied: '+str(q_applied))
            for i in range(1, self.M+1):
                gradient = 2*Sigma@(q_applied-q_optimal)@self.disturbances[t_last_applied-1-1].T
                # gradient = 2*(q_applied-q_optimal)@self.disturbances[t_last_applied-1-1].T
                self.E_all[tau][i-1] = self.gamma*self.E_all[tau][i-1] + gradient@gradient.T
                self.BL_all[tau][i-1] -= self.eta*linalg.inv(self.E_all[tau][i-1])@gradient

                self.BL_all[tau][i-1] = self.projection(self.BL_all[tau][i-1], self.radius*self.decay**(i-1))
                # self.BL_all[tau][i-1] -= self.eta*gradient
        # rospy.loginfo('update time'+str(time.time()-start))

    def projection(self, M, bound):
        """
        Project M to the closest(Frobenius norm) matrix whose l2 norm is bounded.
        """
        U, s, Vh = linalg.svd(M)
        for i in range(4):
            s[i] = min(s[i], bound)
        M_projected = U@linalg.diagsvd(s, 4, 9)@Vh
        return M_projected
    
    def compute_q_optimal(self, t):
        q_optimal = np.zeros((4,1))
        for i in range(self.h+1):
            q_optimal += self.M_optimal_all[i]@self.disturbances[t+i-1]
        return q_optimal

    def compute_A_tilde_power(self):
        """
        Precompute (A-BK)^i, i=0,...,H
        """
        self.A_tilde_power = [np.eye(9)]
        for _ in range(1, self.h+1):
            new = (self.A - self.B@self.K_star)@self.A_tilde_power[-1]
            self.A_tilde_power.append(new)

    def solve_M_optimal_all(self):
        '''
        compute optimal disturbance-feedback gains
        '''
        self.M_optimal_all = []
        inv = linalg.inv(self.R + self.B.T @ self.P_star @ self.B)
        for i in range(self.h+1):
            M_optimal_i = inv@self.B.T@self.A_tilde_power[i].T@self.P_star
            self.M_optimal_all.append(M_optimal_i)
    
    def compute_setpoint(self):
        if self.controller_state == ControllerStates.normal:
            # rospy.loginfo("controller state: normal")
            drone_state = self.drone_state_log[-1]
            target_state = self.target_state_log[-1]
            error = drone_state - target_state
            [thrust, roll_rate, pitch_rate, yaw_rate] = self.compute_setpoint_viaLQR(self.K_star,error, drone_state[8])
            # rospy.loginfo('optimal yaw_rate: '+str(yaw_rate))
            error_feedback = np.array([thrust, pitch_rate, roll_rate, yaw_rate])
            # rospy.loginfo('error: '+str(error))
            # rospy.loginfo('error_feedback: '+str(error_feedback))
            disturbance_feedback = np.zeros((4, 1))
            if len(self.disturbances) >= self.M:
                tau = int((len(self.target_state_log)-1)%(self.h+1))
                for i in range(min(len(self.disturbances), self.M)):
                    disturbance_feedback += self.BL_all[tau][i]@self.disturbances[-1-i]
                # rospy.loginfo('future_disturbance '+str(i)+ ' : '+str(self.disturbances_predicted[i]))
                # rospy.loginfo('disturbance_feedback '+str(i)+ ' : '+str(disturbance_feedback))
            self.q_log.append(disturbance_feedback.copy())
            roll_rate = disturbance_feedback[1].copy()
            pitch_rate = disturbance_feedback[2].copy()
            disturbance_feedback[1] = pitch_rate
            disturbance_feedback[2] = roll_rate

            action = error_feedback - disturbance_feedback
            self.action_DF_log.append(-disturbance_feedback)
            self.action_log.append(action)
            # rospy.loginfo('optimal action[Riccatitron]: '+str(action))
            # convert command from numpy array to ros message
            setpoint = CommandOuter()
            setpoint.thrust = action[0]
            setpoint.omega.x = action[1] # pitch rate
            setpoint.omega.y = action[2] # roll rate
            setpoint.omega.z = action[3] # yaw rate
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

    # Initialization of the Controller Class
    riccatitron_controller = RiccatitronController()

    wait_for_simulator_initialization = rospy.get_param('wait_for_simulator_initialization')
    add_initial_target = rospy.get_param('add_initial_target')

    count = 5
    M = riccatitron_controller.M
    rate = rospy.Rate(f)

    rospy.sleep(2)

    # Save Data
    save_log = True

    # Plot Data
    plot = True

    # Used for timing the algorithm
    Time = 0

    while not (rospy.is_shutdown() or riccatitron_controller.controller_state == ControllerStates.stop):
        ready = False
        if len(riccatitron_controller.drone_state_raw_log)>0 and len(riccatitron_controller.target_state_raw_log)>0:
            ready = True

        if riccatitron_controller.t <= T:
            if ready:
                t0 = time.time()
                if riccatitron_controller.controller_state == ControllerStates.normal:
                    # Call the Riccatitron Algorithm
                    # rospy.loginfo('t: '+str(riccatitron_controller.t))
                    riccatitron_controller.get_new_states()
                    riccatitron_controller.BL_ONS_update()

                # Send the setpoint, aka the input
                riccatitron_controller.publish_setpoint()
                t1 = time.time()
                Time += (t1-t0)/(T*f)
                # Move the timestep
                riccatitron_controller.t += riccatitron_controller.delta_t

                # Used only if Simulink/MATLAB simulator is used
                if wait_for_simulator_initialization:
                    rospy.sleep(4)
                    count -= 1
                    if count < 0:
                        wait_for_simulator_initialization = False
            else:
                rospy.loginfo('No drone or target state estimation is available. Skipping.')
        else:
            if riccatitron_controller.controller_state == ControllerStates.normal:
                riccatitron_controller.publish_setpoint(is_last_command=True)
                rospy.loginfo('Simulation finished.')
            else:
                riccatitron_controller.publish_setpoint()
        rate.sleep()

    if riccatitron_controller.controller_state == ControllerStates.stop:
        # the simulation had started and has now been terminated
        rospy.loginfo('controller state is set to STOP. Terminating.')
        if save_log:
            # save log data to file for evaluation
            filename = rospy.get_param('filename')
            additional_info = f"_{target}_T{T}_f{f}_M{M}"
            new_filename = filename + additional_info
            riccatitron_controller.save_data(new_filename)

            if plot:
                rospy.loginfo('Time: ' + str(Time))

                # Generate a plot of the results
                os.system("rosrun crazyflie_online_tracker plot.py")

