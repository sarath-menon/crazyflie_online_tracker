#!/usr/bin/env python3
import rospy
import numpy as np
import time
import yaml
import os
from scipy import linalg
from crazyflie_online_tracker_interfaces.msg import CommandOuter
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


class SSOGDController(Controller):
    '''
    This is the SSOGD controller proposed in the work 'online linear quadratic tracking with regret guarantees'
    '''
    def __init__(self):
        super().__init__()
        # policy: pi_t(x) = -Kx + v_t
        self.K = self.K_star
        self.v = None
        self.v_log = []
        # update rule: v_t = v_{t-1} - 2*alpha*((I-KS)^TRu_{t-1}+S^TQe_t)
        self.S = linalg.inv(np.eye(9) - self.A + self.B@self.K) @ self.B
        self.alpha = np.array([0.001, 0.001, 0.001, 0.001]).reshape((4, 1))
        # self.check_alpha()

        rospy.init_node('SSOGD_controller')

    def check_alpha(self):
        '''
        check that \rho(A_tilde)<1
        '''
        M = 2*(self.S.T@self.Q@self.B+(np.eye(4) - self.K@self.S).T@self.R)
        H = 2*(self.S.T@self.Q@(self.A-self.B@self.K) - (np.eye(4) - self.K@self.S).T@self.R@self.K)
        A_tilde = np.vstack((np.hstack((np.eye(4) - self.alpha*M,               -self.alpha*H)), 
                             np.hstack((                  self.B, self.A - self.B@self.K))))
        eigvals = linalg.eigh(A_tilde, eigvals_only=True)
        if np.abs(eigvals[0]) >= 1 or np.abs(eigvals[-1])>=1:
            rospy.loginfo("Learning rate is too large, the system is unstable.")

    def update_policy(self):
        last_action = self.action_log[-1]
        drone_state = self.drone_state_log[-1]
        target_state = self.target_state_log[-1]
        curr_error = drone_state - target_state
        self.v = self.v - 2*np.multiply(self.alpha,((np.eye(4)-self.K@self.S).T@self.R@last_action + self.S.T@self.Q@curr_error))
        self.v_log.append(self.v.copy())
    
    def compute_setpoint(self):
        # rospy.loginfo("K_star: " + str(self.K_star))
        drone_state = self.drone_state_raw_log[-1]
        target_state = self.target_state_raw_log[-1]
        if self.controller_state == ControllerStates.normal:
            # rospy.loginfo("controller state: normal")
            self.drone_state_log.append(drone_state)
            self.target_state_log.append(target_state)
            curr_error = drone_state - target_state
            # rospy.loginfo("current state: "+str(drone_state))
            # rospy.loginfo("observe target:" + str(target_state))
            # rospy.loginfo("error:" + str(curr_error))
            if self.v is None:
                # self.v = self.K @ curr_error
                self.v = np.zeros((4, 1))
                self.v_log = [self.v]
            [thrust, roll_rate, pitch_rate, yaw_rate] = self.compute_setpoint_viaLQR(self.K_star,curr_error, drone_state[8])
            action_LQR = np.array([thrust, pitch_rate, roll_rate, yaw_rate])
            # rospy.loginfo("LQR action:" + str(action_LQR))
            self.default_action_log.append(action_LQR)
            
            action_DF = self.v.copy()
            roll_rate = self.v[1].copy()
            pitch_rate = self.v[2].copy()
            action_DF[1] = pitch_rate
            action_DF[2] = roll_rate

            action = action_LQR + action_DF
            self.action_log.append(action)
            self.action_DF_log.append(action_DF)
            # rospy.loginfo('action: '+str(action))
            
            setpoint = CommandOuter()
            setpoint.thrust = action[0]
            setpoint.omega.x = action[1] # pitch rate
            setpoint.omega.y = action[2] # roll rate
            setpoint.omega.z = action[3] # yaw rate
        elif self.controller_state == ControllerStates.takeoff:
            rospy.loginfo("controller state: takeoff")
            setpoint = self.takeoff(drone_state)
        elif self.controller_state == ControllerStates.landing:
            rospy.loginfo("controller state: landing")
            setpoint = self.landing(drone_state)
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

    def publish_setpoint(self, is_last_command=False):
        if is_last_command:
            self.setpoint = CommandOuter()
            self.setpoint.is_last_command = True
        else:
            self.compute_setpoint()
        self.controller_command_pub.publish(self.setpoint)
        

if __name__ == '__main__':
    SSOGD_controller = SSOGDController()
    # publish_frequency = rospy.get_param('publish_frequency')
    wait_for_simulator_initialization = rospy.get_param('wait_for_simulator_initialization')
    count = 5
    rate = rospy.Rate(f)
    rospy.sleep(2)

    # Set to True to save data for post-processing
    save_log = True

    # Used for timing the algorithms
    Time = 0
    
    # Set to True to generate a plot immediately
    plot = True

    while not (rospy.is_shutdown() or SSOGD_controller.controller_state == ControllerStates.stop):
        ready = False
        if len(SSOGD_controller.drone_state_raw_log)>0 and \
           len(SSOGD_controller.target_state_raw_log)>0:
            ready = True
        if SSOGD_controller.t <= T:
            if ready:

                t0 = time.time()
                SSOGD_controller.publish_setpoint()
                if SSOGD_controller.controller_state == ControllerStates.normal and \
                    len(SSOGD_controller.action_log)>0:
                    SSOGD_controller.update_policy()
                t1 = time.time()
                Time += (t1-t0)/(T*f)
                SSOGD_controller.t += SSOGD_controller.delta_t
                if wait_for_simulator_initialization:
                    rospy.sleep(4)
                    count -= 1
                    if count < 0:
                        wait_for_simulator_initialization = False
            else:
                rospy.loginfo('No drone or target state estimation is available. Skipping.')
        else:
            if SSOGD_controller.controller_state == ControllerStates.normal:
                SSOGD_controller.publish_setpoint(is_last_command=True)
            else:
                SSOGD_controller.publish_setpoint()
        rate.sleep()

    if SSOGD_controller.controller_state == ControllerStates.stop:
        rospy.loginfo('controller state is set to STOP. Terminating.')
        if save_log: # the simulation had started and has now been terminated
            filename = rospy.get_param('filename')
            additional_info = f"_{target}_T{T}_f{f}"
            new_filename = filename + additional_info
            SSOGD_controller.save_data(new_filename)
            time.sleep(2)
            if plot:
                rospy.loginfo('Time: ' + str(Time))
                os.system("rosrun crazyflie_online_tracker plot.py")


