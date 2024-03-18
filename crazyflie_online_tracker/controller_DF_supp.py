#!/usr/bin/env python3
import rospy
import numpy as np
from scipy import linalg
import time
import yaml
import os

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
delta_t = float(1/f)


class DFSuppController(Controller):
    """
    Implemenation of the controller proposed by paper 'Online Control with Adversarial Disturbance'(with supplementary information)
    The state and disturbance are defined according to 'online linear quadratic tracking with regret guarantees'
    """

    def __init__(self, T):
        super().__init__()
        self.m = m  # 0.042 for our crazyflie # change to 0.025 for simulation
        # precompute time-invariant parameters
        self.solve_K_star(self.Q,self.R)
        self.compute_matrix_bounds()
        # Simulation time
        self.T = T
        self.t = 0
        # Horizon H = \gamma^{-1}log(T\kappa^2)
        self.H_max = 5# set maximum horizon length, as the theoritical value is too long.
        self.H = min(self.H_max, np.rint(1 / self.gamma * np.log(self.T * (self.kappa ** 2))))
        # Policy: \pi_t(x) = -Kx + \sum_{i=1}^{H}(M^[i-1]w_{t-i})
        # Initial feedback matrices
        self.M = [np.zeros((4, 9))] * self.H
        self.M_log = [self.M]
        # Update rule: M_{t+1}^[i] = proj(M_t^[i] - \eta*GRADIENT)
        self.const = 0.1 # \eta = const/sprt(T). Try reduce const if the controller diverges.
        self.eta = self.const / np.sqrt(self.T)
        self.bound_max = np.inf
        # precompute matrices used for policy update
        self.compute_A_tilde_power()
        # record actions for debugging
        self.stage_cost_log = []
        # Initialize controller node
        rospy.init_node('DF_supp_controller')

    def compute_matrix_bounds(self):
        """
        Compute bounds related to system dynamics, according to assumption 3.1 and definition 3.3
        """
        self.kappa_B = linalg.norm(self.B, ord=2)
        eigvals, eigvecs = linalg.eig(self.A - self.B @ self.K_star)
        self.kappa = max([linalg.norm(self.K_star, ord=2),
                          linalg.norm(eigvecs, ord=2),
                          linalg.norm(linalg.inv(eigvecs), ord=2)])
        self.gamma = 1 - np.amax(np.abs(eigvals))

    def compute_A_tilde_power(self):
        """
        Precompute (A-BK)^i, i=0,...,H
        """
        self.A_tilde_power = [np.eye(9)]
        for _ in range(1, self.H + 1):
            new = (self.A - self.B @ self.K_star) @ self.A_tilde_power[-1]
            self.A_tilde_power.append(new)

    def compute_Psi(self, t, i):
        """
        Compute the disturbance-state tranfer matrix \Psi_{t, i}^{K, H} according to definition 4.2
        """
        if i <= self.H:
            Psi = self.A_tilde_power[i]
            for j in range(0, i):
                Psi = Psi + self.A_tilde_power[j] @ self.B @ self.M[i - j - 1]
        else:
            Psi = np.zeros((9, 9))
            for j in range(i - self.H, self.H + 1):
                Psi = Psi + self.A_tilde_power[j] @ self.B @ self.M[i - j - 1]
        return Psi

    def compute_ideal_state(self):
        """
        Compute the ideal state y_t(M_{t-1-H:t-1}) according to definition 4.4
        """
        t = len(self.M_log) - 1
        # t = len(self.disturbances)
        self.y = np.zeros((9, 1))
        for i in range(0, min(t, 2 * self.H + 1)):
            Psi = self.compute_Psi(t - 1, i)
            self.y = self.y + Psi @ self.disturbances[-i - 1]

    def compute_y_gradient(self, k):
        """
        Compute the partial derivative of ideal state y_t(M) w.r.t.M^[k]
        The analytical solution and python code of the partial derivative are from https://www.matrixcalculus.org/
        """
        gradient_y = np.zeros((9, 4, 9))
        for i in range(k + 1, min(self.H + k + 2, len(self.disturbances))):
            gradient_y += np.einsum('ij, k', self.A_tilde_power[i - k - 1] @ self.B,
                                    self.disturbances[-i - 1].reshape((9,)))
        return gradient_y

    def compute_ideal_action(self):
        """
        Compute the ideal action v_t(M_{t-1-H:t}) according to definition 4.4
        """
        self.v = -self.K_star @ self.y
        n_disturbances = len(self.disturbances)
        for i in range(1, min(n_disturbances, self.H) + 1):
            self.v = self.v + self.M[i - 1] @ self.disturbances[-i]

    def projection(self, M, bound):
        """
        Project M to the closest(Frobenius norm) matrix whose l2 norm is bounded.
        """
        U, s, Vh = linalg.svd(M)
        for i in range(4):
            s[i] = min(s[i], bound)
        M_projected = U @ linalg.diagsvd(s, 4, 9) @ Vh
        return M_projected

    def update_policy(self):
        """
        Update coefficients M^[0], ..., M^[H-1] according to Algorithm 1
        """
        self.compute_ideal_state()
        self.compute_ideal_action()
        M_new = [np.zeros((4, 9))] * self.H
        for i in range(min(len(self.disturbances), self.H)):
            # Compute \frac{\partial{g_t(M)}}{\partial{M^[i]}}
            #        = 2y^T*Q*\frac{\partial{y_t(M)}}{\partial{M^[i]}} + 2v^T*R*\frac{\partial{v_t(M)}}{\partial{M^[i]}}
            #        = (2y^T*Q - 2v^T*R*K)*\frac{\partial{y_t(M)}}{\partial{M^[i]}} + 2R*v*w_{t-i-1}
            gradient_y = self.compute_y_gradient(i)  # dimension 9*4*9
            coef = 2 * self.y.T @ self.Q - 2 * self.v.T @ self.R @ self.K_star  # dimension 1*9
            matmul = np.zeros((4, 9))  # result of matrix multiplication coef*gradient_y
            for p in range(4):
                for q in range(9):
                    matmul[p, q] = coef @ gradient_y[:, p, q]
            gradient = matmul + 2 * self.R @ self.v @ self.disturbances[-1 - i].T
            new = self.M[i] - self.eta * gradient
            bound = min(self.bound_max, self.kappa ** 3 * self.kappa_B) * (1 - self.gamma) ** (i + 1)
            new_projected = self.projection(new, bound)
            M_new[i] = new_projected
        self.M = M_new
        self.M_log.append(self.M.copy())

    def estimate_last_disturbance(self):
        # observe latest drone and target states
        drone_state = self.drone_state_raw_log[-1]
        target_state = self.target_state_raw_log[-1]
        if rospy.get_param('synchronize_target'):
            while len(self.target_state_log) > 0 and np.all(target_state == self.target_state_log[-1]):
                rospy.sleep(0.01)
                target_state = self.target_state_raw_log[-1]
        # rospy.loginfo("current state: " + str(drone_state))
        # rospy.loginfo("observe target:" + str(target_state))
        self.drone_state_log.append(drone_state)
        self.target_state_log.append(target_state)
        # compute disturbance w_t according to the latest drone and target state estimation
        if len(self.target_state_log) < 2:
            return
        else:
            last_target = self.target_state_log[-2]  # r_t
            curr_target = self.target_state_log[-1]  # r_{t+1}
        # w_t = Ar_t - r_{t+1}
        disturbance = self.A @ last_target - curr_target
        # if len(self.disturbances)>2*self.H:
        #     del self.disturbances[0]
        self.disturbances.append(disturbance)
        # rospy.loginfo("disturbance:" + str(disturbance))

    def compute_setpoint(self):
        """
        Compute action u_t
        """
        drone_state = self.drone_state_log[-1]
        target_state = self.target_state_log[-1]
        if self.controller_state == ControllerStates.normal:
            curr_error = drone_state - target_state
            # rospy.loginfo("error:" + str(curr_error))

            [thrust, roll_rate, pitch_rate, yaw_rate] = self.compute_setpoint_viaLQR(self.K_star, curr_error, drone_state[8])
            action_LQR = np.array([thrust, pitch_rate, roll_rate, yaw_rate])
            self.default_action_log.append(action_LQR)
            # rospy.loginfo("LQR action:" + str(action_LQR))

            action_disturbance_feedback = np.zeros((4, 1))
            if len(self.disturbances) >= self.H:
                for i in range(1, self.H + 1):
                    # for i in range(1, min(len(self.disturbances), self.H)+1):
                    action_disturbance_feedback = action_disturbance_feedback + self.M[i - 1] @ self.disturbances[-i]
                    # rospy.loginfo('M['+str(i)+'-1]wt-'+str(i)+'= '+str(self.M[i-1]@self.disturbances[-i]))
            roll_rate = action_disturbance_feedback[1].copy()
            pitch_rate = action_disturbance_feedback[2].copy()
            action_disturbance_feedback[1] = pitch_rate
            action_disturbance_feedback[2] = roll_rate

            action = action_LQR + action_disturbance_feedback
            self.action_log.append(action)
            self.action_DF_log.append(action_disturbance_feedback)
            # rospy.loginfo('action: '+str(action_disturbance_feedback))

            # stage_cost = curr_error.T@self.Q@curr_error + action.T@self.R@action
            # self.stage_cost_log.append(stage_cost)
            # convert command from numpy array to ros message
            setpoint = CommandOuter()
            setpoint.thrust = action[0]
            setpoint.omega.x = action[1]  # pitch rate
            setpoint.omega.y = action[2]  # roll rate
            setpoint.omega.z = action[3]  # yaw rate
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
    DF_supp_controller = DFSuppController(T)
    H = DF_supp_controller.H
    eta = DF_supp_controller.eta
    rounded_eta = round(eta, 2)
    wait_for_simulator_initialization = rospy.get_param('wait_for_simulator_initialization')
    rate = rospy.Rate(f)

    # Set to True to save data for post-processing
    save_log = True
    # Used for timing the algorithms
    Time = 0

    rospy.sleep(2)
    count = 5

    # Set to True to generate a plot immediately
    plot = True

    while not (rospy.is_shutdown() or DF_supp_controller.controller_state == ControllerStates.stop):
        ready = False
        if len(DF_supp_controller.drone_state_raw_log)>0 and \
           len(DF_supp_controller.target_state_raw_log)>0:
            ready = True
        if DF_supp_controller.t <= DF_supp_controller.T:
            if ready:
                t0 = time.time()
                DF_supp_controller.estimate_last_disturbance()
                DF_supp_controller.publish_setpoint()
                if DF_supp_controller.controller_state == ControllerStates.normal:
                    DF_supp_controller.update_policy()
                t1 = time.time()
                Time += (t1-t0)/(T*f)
                DF_supp_controller.t += DF_supp_controller.delta_t
                if wait_for_simulator_initialization:
                    rospy.sleep(4)
                    count -= 1
                    if count < 0:
                        wait_for_simulator_initialization = False
            else:
                rospy.loginfo('No drone or target state estimation is available. Skipping.')
        else:
            DF_supp_controller.estimate_last_disturbance()
            if DF_supp_controller.controller_state == ControllerStates.normal:
                DF_supp_controller.publish_setpoint(is_last_command=True)
            else:
                DF_supp_controller.publish_setpoint()
        rate.sleep()

    if DF_supp_controller.controller_state == ControllerStates.stop:
        rospy.loginfo('controller state is set to STOP. Terminating.')
        if save_log: # the simulation had started and has now been terminated
            filename = rospy.get_param('filename')
            additional_info = f"_{target}_T{T}_f{f}_H{H}_eta{rounded_eta}"
            new_filename = filename + additional_info
            DF_supp_controller.save_data(new_filename) # save log data to file for evaluation
            time.sleep(2)
            if plot:
                rospy.loginfo('Time: ' + str(Time))
                os.system("rosrun crazyflie_online_tracker plot.py")
    rospy.loginfo(f"H: {H} and eta: {eta}")

