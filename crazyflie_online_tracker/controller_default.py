#!/usr/bin/env python3
import rospy
import numpy as np
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
mode = data['mode']
delta_t = float(1/f)
filtering = data['filtering']


class DefaultController(Controller):
    '''
    Naive LQR (or simply LQR) controller
    '''
    def __init__(self):
        super().__init__()
        self.m = m
        rospy.init_node('default_controller')

    def compute_setpoint(self):
        drone_state = self.drone_state_raw_log[-1]
        target_state = self.target_state_raw_log[-1]

        if self.controller_state == ControllerStates.normal:
            # rospy.loginfo("controller state: normal")
            self.drone_state_log.append(drone_state)
            self.target_state_log.append(target_state)


            # rospy.loginfo("observe target[default]:" + str(target_state))
            # rospy.loginfo("current state[default]: " + str(drone_state))

            # option 1: rotated error + smoothed setpoint: NOT USED
            if self.last_setpoint is None:
                self.last_setpoint = drone_state[:3]
            desired_pos_limited = self.limit_pos_change(self.last_setpoint, target_state[:3])
            error_limited = drone_state - target_state
            error_limited[0] = drone_state[0] - desired_pos_limited[0]
            error_limited[1] = drone_state[1] - desired_pos_limited[1]
            error_limited[2] = drone_state[2] - desired_pos_limited[2]
            self.last_setpoint = desired_pos_limited
            # rospy.loginfo("setpoint limited:" + str(desired_pos_limited))
            [thrust, roll_rate, pitch_rate, yaw_rate] = self.compute_setpoint_viaLQR_controller(self.K_star, error_limited, drone_state[8])
            action_rotated_limited = np.array([thrust, pitch_rate, roll_rate, yaw_rate])

            # option 2: rotated error: USED
            error = drone_state - target_state
            [thrust, roll_rate, pitch_rate, yaw_rate] = self.compute_setpoint_viaLQR(self.K_star, error, drone_state[8])
            action_rotated = np.array([thrust, pitch_rate, roll_rate, yaw_rate])

            # option 3: naive LQR: NOT USED
            error = drone_state - target_state
            action_naive = -self.K_star@error
            action_naive[0] = action_naive[0] + self.m*self.g
            roll_rate = action_naive[1].copy()
            pitch_rate = action_naive[2].copy()
            action_naive[1] = pitch_rate
            action_naive[2] = roll_rate

            action = action_rotated # option 2
            self.action_log.append(action)
            setpoint = CommandOuter()
            setpoint.thrust = action[0]
            setpoint.omega.x = action[1] # pitch rate
            setpoint.omega.y = action[2] # roll rate
            setpoint.omega.z = action[3] # yaw rate
            # rospy.loginfo('error:'+str(action))

            disturbance_feedback = np.zeros((4,1))
            self.action_DF_log.append(disturbance_feedback)


        elif self.controller_state == ControllerStates.takeoff:
            rospy.loginfo("controller state: takeoff")
            setpoint = self.takeoff(drone_state)

            # rospy.loginfo('takeoff action:' + str(setpoint))
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

    default_controller = DefaultController()
    wait_for_simulator_initialization = rospy.get_param('wait_for_simulator_initialization')
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

    while not (rospy.is_shutdown() or default_controller.controller_state == ControllerStates.stop):
        ready = False
        if len(default_controller.drone_state_raw_log) > 0 or len(default_controller.filtered_drone_state_raw_log)>0:
            if len(default_controller.target_state_raw_log) > 0:
                ready = True
            if default_controller.t <= T:
                if ready:
                    t0 = time.time()
                    default_controller.publish_setpoint()
                    t1 = time.time()
                    Time+= (t1-t0)
                    Time_T+= 1
                    # rospy.loginfo('default controller published the setpoint')
                    default_controller.t += default_controller.delta_t
                    if wait_for_simulator_initialization:
                        rospy.sleep(4)
                        count -= 1
                        if count < 0:
                            wait_for_simulator_initialization = False
                else:
                    rospy.loginfo('No drone or target state estimation is available. Skipping.')
            else:
                if default_controller.controller_state == ControllerStates.normal:
                    default_controller.publish_setpoint(is_last_command=True)
                    rospy.loginfo('Simulation finished.')
                else:
                    default_controller.publish_setpoint()
            rate.sleep()

    if default_controller.controller_state == ControllerStates.stop:
        rospy.loginfo('controller state is set to STOP. Terminating.')
        if save_log: # the simulation had started and has now been terminated
            filename = rospy.get_param('filename')
            additional_info = f"_{target}_T{T}_f{f}_mode{mode}"
            if filtering:
                additional_info = f"_{target}_T{T}_f{f}_mode{mode}_Filtered"
            new_filename = filename + additional_info
            default_controller.save_data(new_filename) # save log data to file for evaluation
            time.sleep(2)
            if plot:
                rospy.loginfo('Printing the figures')
                rospy.loginfo('Time: ' + str(Time/Time_T))
                os.system("rosrun crazyflie_online_tracker plot.py")





