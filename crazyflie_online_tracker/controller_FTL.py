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


class FTLController(Controller):
    '''
    Implemenation of the controller proposed by paper 'Tracking adversarial targets'
    '''
    def __init__(self):
        super().__init__()
        # policy: pi_t(x) = -Kx + c_t
        B_tilde = self.B @ np.linalg.inv(np.sqrt(self.R))
        self.B = B_tilde
        self.solve_K_star(self.Q, self.R)
        self.c_log = []
        self.solve_P()
        self.P_sum = 0
        self.P_log = []
        self.L_sum = 0
        self.L_log = []
        self.bias_initialized = False
        self.discount = 1.0

        rospy.init_node('FTL_controller')

    def solve_P(self):
        self.A_aug = np.vstack((np.hstack((self.A, self.B)),
                                np.hstack((-self.K_star @ self.A, -self.K_star @ self.B)))).T
        Q_aug = np.vstack((np.hstack((self.Q, np.zeros((9, 4)))),
                           np.hstack((np.zeros((4, 9)), np.eye(4)))))
        self.P = linalg.solve_discrete_lyapunov(self.A_aug, Q_aug)

    def solve_L(self):
        target_state = self.target_state_raw_log[-1]
        # rearrange eq(4) and solve the linear system AL = b
        b = 2 * self.A_aug @ self.P.T @ np.vstack((np.zeros((9, 1)), self.c)) \
            - np.vstack((2 * self.Q @ target_state, np.zeros((4, 1))))
        A = np.eye(13) - self.A_aug
        self.L = linalg.solve(A, b)
        self.L_sum = self.discount * self.L_sum + self.L
        self.L_log += [self.L]

    def update_policy(self):
        self.P_sum = self.discount*self.P_sum + self.P # P_t is unchanged according to lemma3
        self.solve_L()
        self.c = -0.5 * linalg.inv(self.P_sum[-4:, -4:]) @ self.L_sum[-4:]
        self.c_log += [self.c]

    def compute_setpoint(self):
        drone_state = self.drone_state_raw_log[-1]
        target_state = self.target_state_raw_log[-1]
        if self.controller_state == ControllerStates.normal:
            # rospy.loginfo("controller state: normal")
            if not self.bias_initialized:
                self.bias_initialized = True
                self.c = self.K_star@target_state
                self.c_log.append(self.c.copy())
            action = np.linalg.inv(np.sqrt(self.R))@(-self.K_star @ drone_state + self.c)
            action[0] = action[0] + self.m*self.g
            roll_rate = action[1].copy()
            pitch_rate = action[2].copy()
            action[1] = pitch_rate
            action[2] = roll_rate
            self.action_log.append(action)

            error = drone_state - target_state
            self.error_feedback = self.compute_setpoint_viaLQR(self.K_star, error, drone_state[8])
            self.error_feedback = np.array(self.error_feedback)

            disturbance_feedback = self.error_feedback - action
            self.action_DF_log.append(disturbance_feedback)
            # rospy.loginfo('action: '+str(action))
            default_action = np.linalg.inv(np.sqrt(self.R))@(-self.K_star @ (drone_state-target_state))
            default_action[0] = default_action[0] + self.m*self.g
            roll_rate = default_action[1].copy()
            pitch_rate = default_action[2].copy()
            default_action[1] = pitch_rate
            default_action[2] = roll_rate
            self.default_action_log.append(default_action)

            setpoint = CommandOuter()
            setpoint.thrust = action[0]
            setpoint.omega.x = action[1] # pitch rate
            setpoint.omega.y = action[2] # roll rate
            setpoint.omega.z = action[3] # yaw rate

            self.drone_state_log.append(drone_state)
            self.target_state_log.append(target_state)
            # rospy.loginfo("current time: " + str(self.t))
            # rospy.loginfo("current state[RLS]: " + str(drone_state))
            # rospy.loginfo("observe target[RLS]:" + str(target_state))

            # compute disturbance w_t according to the latest drone and target state estimation
            if len(self.target_state_log) > 1:
                last_target = self.target_state_log[-2]  # r_t
                curr_target = self.target_state_log[-1]  # r_{t+1}
                # w_t = Ar_t - r_{t+1}
                disturbance = self.A @ last_target - curr_target
                self.disturbances.append(disturbance)

        elif self.controller_state == ControllerStates.takeoff:
            rospy.loginfo("controller state: takeoff")
            setpoint = self.takeoff(drone_state)
        elif self.controller_state == ControllerStates.landing:
            rospy.loginfo("controller state: landing")
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
    FTL_controller = FTLController()
    wait_for_simulator_initialization = rospy.get_param('wait_for_simulator_initialization')
    add_initial_target = rospy.get_param('add_initial_target')

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

    while not (rospy.is_shutdown() or FTL_controller.controller_state == ControllerStates.stop):
        ready = False
        if add_initial_target and len(FTL_controller.target_state_raw_log) == 0:
            initial_target = np.zeros((9, 1))
            initial_target[:3] = FTL_controller.desired_pos.reshape((3, 1))
            FTL_controller.target_state_raw_log.append(initial_target)
        if len(FTL_controller.drone_state_raw_log) > 0 and \
                len(FTL_controller.target_state_raw_log) > 0:
            ready = True
        if FTL_controller.t <= T:
            if ready:

                if FTL_controller.controller_state == ControllerStates.normal:
                    t0 = time.time()
                    FTL_controller.publish_setpoint()
                    FTL_controller.update_policy()
                    t1 = time.time()

                    Time_T +=1
                    Time += (t1 - t0)

                FTL_controller.t += FTL_controller.delta_t
                if wait_for_simulator_initialization:
                    rospy.sleep(4)
                    count -= 1
                    if count < 0:
                        wait_for_simulator_initialization = False
            else:
                rospy.loginfo('No drone or target state estimation is available. Skipping.')
        else:
            if FTL_controller.controller_state == ControllerStates.normal:
                FTL_controller.publish_setpoint(is_last_command=True)
                rospy.loginfo('Simulation finished.')
            else:
                FTL_controller.publish_setpoint()
        rate.sleep()

    if FTL_controller.controller_state == ControllerStates.stop:
        rospy.loginfo('controller state is set to STOP. Terminating.')

        if save_log:
            filename = rospy.get_param('filename')
            additional_info = f"_{target}_T{T}_f{f}"
            # additional_info = f"_{target}_T{T}_f{f}_mode{mode}"
            new_filename = filename + additional_info
            FTL_controller.save_data(new_filename)
            time.sleep(2)
            if plot:
                rospy.loginfo('Time: ' + str(Time / (Time_T)))
                os.system("rosrun crazyflie_online_tracker plot.py")


