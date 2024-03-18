#!/usr/bin/env python3
import numpy as np
import os
import time
import glob
import yaml
from PIL import Image
from scipy import linalg
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

font = {'size': 22}

matplotlib.rc('font', **font)

# Load data from the YAML file
yaml_path = os.path.join(os.path.dirname(__file__), '../param/data.yaml')

image_path = os.path.join(os.path.dirname(__file__), '../images/')

if not os.path.isdir(image_path):
    os.makedirs(image_path)

with open(yaml_path, 'r') as file:
    yaml_data = yaml.safe_load(file)

# Extract the required variables
g = yaml_data['g']
m = yaml_data['m']
f = yaml_data['f']
T = yaml_data['T']
W = yaml_data['W_RLS']
path = yaml_data['path']

save_path = os.path.expanduser(path + 'data')
if not os.path.isdir(save_path):
    os.makedirs(save_path)

delta_t = float(1/f)

#  Data Loader Function
def load_data(filename):
    data = np.load(os.path.join(save_path, filename))
    target_state_log = data['target_state_log']
    drone_state_algorithm_log = data['drone_state_log']
    action_algorithm_log = data['action_log']
    try:
        action_DF_log = data['action_DF_log']
    except:
        action_DF_log = []
    try:
        disturbances = data['disturbance_log']
    except:
        disturbances = []
    try:
        default_action_log = data['default_action_log']
    except:
        default_action_log = []
    try:
        optimal_action_log = data['optimal_action_log']
    except:
        optimal_action_log = []
    try:
        S_matrices = data['learnt_S_log'] #
    except:
        S_matrices = []

    return target_state_log, drone_state_algorithm_log,  action_algorithm_log, action_DF_log, disturbances, default_action_log, optimal_action_log, S_matrices


class Plotter():
    '''
    This Class reads the saved .npz data files and provides figures
    '''
    def __init__(self):
        self.Q = np.diag([80, 80, 80, 10, 10, 10, 0.01, 0.01, 0.1])
        self.R = np.diag([0.7, 2.5, 2.5, 2.5])

        self.A_outer = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0],

                            [0, 0, 0, 0, 0, 0, 0, g, 0],
                            [0, 0, 0, 0, 0, 0, -g, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],

                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.A = np.eye(9) + delta_t * self.A_outer
        self.B_outer = np.array([[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],

                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [1 / m, 0, 0, 0],

                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        self.B = delta_t * self.B_outer

        self.P_star = linalg.solve_discrete_are(self.A, self.B, self.Q, self.R)
        self.B_transpose_P = self.B.T @ self.P_star
        self.K_star = linalg.inv(self.R + self.B_transpose_P @ self.B) @ self.B_transpose_P @ self.A
        # self.F = self.A - self.B @ self.K_star
        # self.eigenvalues, self.eigenvectors = linalg.eig(self.F)
        # self.aa = 0


    def compute_cost_and_error(self, drone_states, target_states, actions):
        T = len(actions)
        cost_log, error_log = [], []
        for t in range(T):
            error = drone_states[t] - target_states[t]
            action = actions[t]
            position_error = np.linalg.norm(error[:3])
            velocity_error = np.linalg.norm(error[3:6])
            error_log.append([position_error, velocity_error])
            cost_log.append(error.T @ self.Q @ error + action.T @ self.R @ action)
        return cost_log, error_log

    def plot_action(self, action_logs, title):
        '''
        Plot the Control Action
        '''
        action = action_logs[0]
        length = action.shape[0]
        default_action = action_logs[1]
        optimal_action = action_logs[2]
        if len(default_action) > 0:
            plot_default = True
            length = min(length, default_action.shape[0])
        else:
            plot_default = False

        if len(optimal_action) > 0:
            plot_optimal = True
            length = min(length, optimal_action.shape[0])
        else:
            plot_optimal = False

        fig, ax = plt.subplots(4, 1,figsize=(12, 12))
        time = np.array(range(length)) * 0.1
        ax[0].plot(time, action[:length, 0, :])
        if plot_default:
            ax[0].plot(time, default_action[:length, 0, :])
        if plot_optimal:
            ax[0].plot(time, optimal_action[:length, 0, :])
        ax[0].set_xlabel("time[s]")
        ax[0].set_ylabel("total thrust[N]")
        ax[0].grid()

        ax[1].plot(time, action[:length, 1, :])
        if plot_default:
            ax[1].plot(time, default_action[:length, 1, :])
        if plot_optimal:
            ax[1].plot(time, optimal_action[:length, 1, :])
        ax[1].set_xlabel("time[s]")
        ax[1].set_ylabel("pitch rate[rad/s]")
        ax[1].grid()

        ax[2].plot(time, action[:length, 2, :])
        if plot_default:
            ax[2].plot(time, default_action[:length, 2, :])
        if plot_optimal:
            ax[2].plot(time, optimal_action[:length, 2, :])
        ax[2].set_xlabel("time[s]")
        ax[2].set_ylabel("roll rate[rad/s]")
        ax[2].grid()

        ax[3].plot(time, action[:length, 3, :])
        if plot_default:
            ax[3].plot(time, default_action[:length, 3, :])
        if plot_optimal:
            ax[3].plot(time, optimal_action[:length, 3, :])
        ax[3].set_xlabel("time[s]")
        ax[3].set_ylabel("yaw rate[rad/s]")
        ax[3].grid()

        fig.suptitle(title)
        if name == 'RLS':
            legends = ['RLS-MPC' + ' input']
        elif name == 'DAP':
            legends = ['DAP' + ' input']
        elif name == 'SSO':
            legends = ['SS-OGD' + ' input']
        elif name == 'ric':
            legends = ['Riccatitron' + ' input']
        else:
            legends = ['LQR' + ' input']
        if plot_default:
            legends.append('default action')
        if plot_optimal:
            legends.append('optimal action')
        fig.legend(legends)

    def plot_just_state_vertical(self,drone_state_log, target_state_log,save_img=False, name=None):
        '''
        Plot the y and z coordinates
        '''
        legends = []
        for _, n in enumerate(name):
            alg_name = n[15:18]
            if alg_name == 'RLS':
                legends.append('PLOT')
            elif alg_name == 'DAP':
                legends.append('DAP')
            elif alg_name == 'SSO':
                legends.append('SS-OGD')
            elif alg_name == 'ric':
                legends.append('Riccatitron')
            elif alg_name == 'FTL':
                legends.append('FTL')
            else:
                legends.append('LQR')

        colors = ['g',
                  'tab:purple',
                  # 'tab:red',
                  'tab:cyan',
                  'tab:gray',
                  'tab:olive',
                  ]
        target = target_state_log[0]

        fig, ax = plt.subplots(2, 1,figsize=(9, 10))

        timev = np.array(range(target.shape[0])) * 0.1

        ax[0].plot(timev, target[:, 1, :], color=colors[0], linewidth=3, label='Target')
        # ax[0].set_xlabel("time[s]")
        ax[0].set_ylabel("y[m]")
        ax[0].grid()

        ax[1].plot(timev, target[:, 2, :],color=colors[0], linewidth=3, label='Target')
        ax[1].set_xlabel("time[s]")
        ax[1].set_ylabel("z[m]")
        ax[1].grid()

        for idx, state in enumerate(drone_state_log):

            ax[0].plot(timev, state[:, 1, :], color=colors[1 + idx], linewidth=2, label=legends[idx])
            ax[1].plot(timev, state[:, 2, :], color=colors[1 + idx], linewidth=2, label=legends[idx])

        handles, labels = plt.gca().get_legend_handles_labels()
        labels, ids = np.unique(labels, return_index=True)
        handles = [handles[i] for i in ids]
        # order = [0, 2, 1, 3]
        fig.legend(handles, labels, loc='upper right', handlelength=0.5)
        if save_img:
            plt.savefig(image_path + name[0] + '_' + 'state_plot_vertical' + '.eps', format='eps')
            print('Figure Saved')
            time.sleep(0.01)
            plt.close(fig)
        plt.show()

    def plot_just_state(self,drone_state_log, target_state_log,save_img=False, name=None):
        '''
        Plot the x and y coordinates
        '''
        legends = []
        for _, n in enumerate(name):
            alg_name = n[15:18]
            if alg_name == 'RLS':
                legends.append('PLOT')
            elif alg_name == 'DAP':
                legends.append('DAP')
            elif alg_name == 'SSO':
                legends.append('SS-OGD')
            elif alg_name == 'ric':
                legends.append('Riccatitron')
            elif alg_name == 'FTL':
                legends.append('FTL')
            else:
                legends.append('LQR')

        colors = ['g',
                  'tab:purple',
                  # 'tab:red',
                  'tab:cyan',
                  'tab:gray',
                  'tab:olive',
                  ]
        target = target_state_log[0]

        fig, ax = plt.subplots(2, 1,figsize=(9, 10))

        timev = np.array(range(target.shape[0])) * 0.1

        ax[0].plot(timev, target[:, 0, :], color=colors[0], linewidth=3, label='Target')
        # ax[0].set_xlabel("time[s]")
        ax[0].set_ylabel("x[m]")
        ax[0].grid()

        ax[1].plot(timev, target[:, 1, :],color=colors[0], linewidth=3, label='Target')
        ax[1].set_xlabel("time[s]")
        ax[1].set_ylabel("y[m]")
        ax[1].grid()

        for idx, state in enumerate(drone_state_log):

            ax[0].plot(timev, state[:, 0, :], color=colors[1 + idx], linewidth=2, label=legends[idx])
            ax[1].plot(timev, state[:, 1, :], color=colors[1 + idx], linewidth=2, label=legends[idx])

        handles, labels = plt.gca().get_legend_handles_labels()
        labels, ids = np.unique(labels, return_index=True)
        handles = [handles[i] for i in ids]
        # order = [0, 2, 1, 3]
        fig.legend(handles, labels, loc='top right')
        if save_img:
            plt.savefig(image_path + name[0] + '_' + 'state_plot' + '.eps', format='eps')
            print('Figure Saved')
            time.sleep(0.01)
            plt.close(fig)
        plt.show()

    def plot_state(self,state_log, title):
        '''
        Plot all the states
        '''
        if len(state_log) > 1:
            plotTarget = True
            drone_state_log = state_log[0]
            target_state_log = state_log[1]
        else:
            plotTarget = False
            drone_state_log = state_log[0]

        fig, ax = plt.subplots(3, 3,figsize=(12, 12))

        time = np.array(range(drone_state_log.shape[0])) * 0.1
        ax[0, 0].plot(time, drone_state_log[:, 0, :])
        if plotTarget:
            ax[0, 0].plot(time, target_state_log[:, 0, :])
        ax[0, 0].set_xlabel("time[s]")
        ax[0, 0].set_ylabel("x[m]")
        ax[0, 0].grid()

        ax[0, 1].plot(time, drone_state_log[:, 1, :])
        if plotTarget:
            ax[0, 1].plot(time, target_state_log[:, 1, :])
        ax[0, 1].set_xlabel("time[s]")
        ax[0, 1].set_ylabel("y[m]")
        ax[0, 1].grid()

        ax[0, 2].plot(time, drone_state_log[:, 2, :])
        if plotTarget:
            ax[0, 2].plot(time, target_state_log[:, 2, :])
        ax[0, 2].set_xlabel("time[s]")
        ax[0, 2].set_ylabel("z[m]")
        ax[0, 2].grid()

        ax[1, 0].plot(time, drone_state_log[:, 3, :])
        if plotTarget:
            ax[1, 0].plot(time, target_state_log[:, 3, :])
        ax[1, 0].set_xlabel("time[s]")
        ax[1, 0].set_ylabel("vx[m/s]")
        ax[1, 0].grid()

        ax[1, 1].plot(time, drone_state_log[:, 4, :])
        if plotTarget:
            ax[1, 1].plot(time, target_state_log[:, 4, :])
        ax[1, 1].set_xlabel("time[s]")
        ax[1, 1].set_ylabel("vy[m/s]")
        ax[1, 1].grid()

        ax[1, 2].plot(time, drone_state_log[:, 5, :])
        if plotTarget:
            ax[1, 2].plot(time, target_state_log[:, 5, :])
        ax[1, 2].set_xlabel("time[s]")
        ax[1, 2].set_ylabel("vz[m/s]")
        ax[1, 2].grid()

        ax[2, 0].plot(time, drone_state_log[:, 6, :])
        if plotTarget:
            ax[2, 0].plot(time, target_state_log[:, 6, :])
        ax[2, 0].set_xlabel("time[s]")
        ax[2, 0].set_ylabel("roll[rad]")
        ax[2, 0].grid()

        ax[2, 1].plot(time, drone_state_log[:, 7, :])
        if plotTarget:
            ax[2, 1].plot(time, target_state_log[:, 7, :])
        ax[2, 1].set_xlabel("time[s]")
        ax[2, 1].set_ylabel("pitch[rad]")
        ax[2, 1].grid()

        ax[2, 2].plot(time, drone_state_log[:, 8, :])
        if plotTarget:
            ax[2, 2].plot(time, target_state_log[:, 8, :])
        ax[2, 2].set_xlabel("time[s]")
        ax[2, 2].set_ylabel("yaw[rad]")
        ax[2, 2].grid()

        fig.suptitle(title)
        if plotTarget:
            fig.legend(['drone state', 'target state'])

    def create_video(self,drone_state_log, target_state_log, save_img = False, name = 'test',length=np.inf):
        '''
        Create a video in the x-y plane
        '''

        # Choose the length of the video in Timesteps
        length = 300
        for state in drone_state_log:
            if len(state) > 0:
                length = int(min(length, len(state)))
        if name == 'RLS':
            legends = ['PLOT']
        elif name == 'DAP':
            legends = ['DAP']
        elif name == 'SSO':
            legends = ['SS-OGD']
        elif name == 'ric':
            legends = ['Riccatitron']
        else:
            legends = ['LQR']

        colors = ['g',
                  'tab:purple',
                  'tab:red',
                  'tab:brown',
                  'tab:pink',
                  'tab:gray',
                  'tab:olive',
                  'tab:cyan'
                  ]

        target = target_state_log[0]
        drone_state = drone_state_log[0]
        plt.style.use("ggplot")
        # initializing a figure
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))


        # plt.axis('equal')
        axes.set_xlabel('x[m]')
        axes.set_ylabel('y[m]')

        axes.set_xlim(0.2, 2.0)
        axes.set_ylim(-0.1, 1.3)

        # lists storing x and y values
        x_target, y_target = [], []
        x_drone, y_drone = [], []


        # Drone state markings
        plt.scatter(state[0, 0, :], state[0, 1, :], color=colors[1 + idx], s=255,  marker="o")

        # Target markings
        plt.scatter(target[0, 0, :], target[0, 1, :], color=colors[0], s=255, marker="o")

        # Target Initial Plot
        self.my_line1 = axes.scatter(target[0, 0, :], target[0, 1, :], color=colors[0], s=255, label='Target',
                                     marker="*")

        # Drone State Initial Plot
        self.my_line2 = axes.scatter(drone_state[0, 0, :], drone_state[0, 1, :], linewidth=2, color=colors[1],
                                     label=legends[0], marker="*")

        handles, labels = plt.gca().get_legend_handles_labels()
        labels, ids = np.unique(labels, return_index=True)
        handles = [handles[i] for i in ids]
        order = [0, 1]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper left')

        def animate(i):

            self.my_line1.remove()
            self.my_line2.remove()

            x_target.append(target[i,0,:])
            y_target.append(target[i,1,:])

            x_drone.append(drone_state[i, 0, :])
            y_drone.append(drone_state[i, 1, :])

            axes.plot(x_target, y_target, color=colors[0], linewidth=5, label='Target')
            axes.plot(x_drone, y_drone, color=colors[1], linewidth=2, label=legends[0])

            self.my_line1 = axes.scatter(x_target[-1], y_target[-1], color=colors[0], linewidth=2, s=255, label='Target',
                                        marker="*")
            self.my_line2 = axes.scatter(x_drone[-1], y_drone[-1], color=colors[1], s=255, linewidth=2, label=legends[0],marker="*")

        anim = animation.FuncAnimation(fig, animate, frames=length, interval=1000*delta_t,repeat = False)

        if save_img:
            print('Saving Video ...')
            writervideo = animation.FFMpegWriter(fps=f)
            video_path = image_path + name + ".mp4"
            anim.save(video_path, writervideo)
            plt.close()

        plt.show()

    def create_video_vertical(self,drone_state_log, target_state_log, save_img = False, name = 'test',length=np.inf):
        '''
        Create a video in the y-z plane
        '''

        # Choose the length of the video in Timesteps
        length = 110
        for state in drone_state_log:
            if len(state) > 0:
                length = int(min(length, len(state)))
        if name == 'RLS':
            legends = ['RLS-MPC']
        elif name == 'DAP':
            legends = ['DAP']
        elif name == 'SSO':
            legends = ['SS-OGD']
        elif name == 'ric':
            legends = ['Riccatitron']
        else:
            legends = ['LQR']

        colors = ['g',
                  'tab:purple',
                  'tab:red',
                  'tab:brown',
                  'tab:pink',
                  'tab:gray',
                  'tab:olive',
                  'tab:cyan'
                  ]

        target = target_state_log[0]
        drone_state = drone_state_log[0]
        plt.style.use("ggplot")
        # initializing a figure
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))


        # plt.axis('equal')
        axes.set_xlabel('y[m]')
        axes.set_ylabel('z[m]')

        # lists storing x and y values
        x_target, y_target = [], []
        x_drone, y_drone = [], []

        # Target Initial Plot
        axes.plot(target[0,1,:],target[0,2,:], color=colors[0], linewidth=5, label='Target')

        # Drone State Initial Plot
        axes.plot(drone_state[0,1, :], drone_state[0, 2, :], linewidth=2, color=colors[1], label=legends[0])

        handles, labels = plt.gca().get_legend_handles_labels()
        labels, ids = np.unique(labels, return_index=True)
        handles = [handles[i] for i in ids]
        order = [0, 1]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper left')

        def animate(i):
            x_target.append(target[i,1,:])
            y_target.append(target[i,2,:])

            x_drone.append(drone_state[i, 1, :])
            y_drone.append(drone_state[i, 2, :])

            axes.plot(x_target, y_target, color=colors[0], linewidth=5, linestyle='dashed', label='Target')
            axes.plot(x_drone, y_drone, color=colors[1], linewidth=2, label=legends[0])

        anim = animation.FuncAnimation(fig, animate, frames=length, interval=1000*delta_t,repeat = False)

        if save_img:
            print('Saving Video ...')
            writervideo = animation.FFMpegWriter(fps=f)
            video_path = image_path + name + ".mp4"
            anim.save(video_path, writervideo)
            plt.close()

        plt.show()

    def create_video_traj(self,drone_state_log,target_state_log, save_img = False, name = 'test',length=np.inf):
        '''
        Create a video without the trajectory in the x-y plane
        '''

        # Choose the length of the video in Timesteps
        # length = 110
        for state in drone_state_log:
            if len(state) > 0:
                length = int(min(length, len(state)))
        if name == 'RLS':
            legends = ['PLOT']
        elif name == 'DAP':
            legends = ['DAP']
        elif name == 'SSO':
            legends = ['SS-OGD']
        elif name == 'ric':
            legends = ['Riccatitron']
        else:
            legends = ['LQR']

        colors = ['g',
                  'tab:purple',
                  'tab:red',
                  'tab:brown',
                  'tab:pink',
                  'tab:gray',
                  'tab:olive',
                  'tab:cyan'
                  ]

        target = target_state_log[0]
        drone_state = drone_state_log[0]

        # initializing a figure
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))

        # set limit for x and y axis
        # axes.set_ylim(-2.1, 2.1)
        # axes.set_xlim(0.4, 0.8)

        # plt.axis('equal')
        axes.set_xlabel('x[m]')
        axes.set_ylabel('y[m]')

        # lists storing x and y values


        # Target Initial Plot
        self.my_line1 = axes.scatter(target[0,0,:],target[0,1,:], color=colors[0], s=255, label='Target', marker = "*")

        # Drone State Initial Plot
        self.my_line2 = axes.scatter(drone_state[0, 0, :], drone_state[0, 1, :], linewidth=2, color=colors[1], label=legends[0])

        handles, labels = plt.gca().get_legend_handles_labels()
        labels, ids = np.unique(labels, return_index=True)
        handles = [handles[i] for i in ids]
        order = [0, 1]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper left')

        def animate(i):
            # axes.clear()
            self.my_line1.remove()
            self.my_line2.remove()


            x_target = target[i,0,:]
            y_target = target[i,1,:]

            x_drone = drone_state[i, 0, :]
            y_drone = drone_state[i, 1, :]
            self.my_line1 = axes.scatter(x_target, y_target, color=colors[0], s=255, linestyle='dashed', label='Target', marker = "*")
            self.my_line2 = axes.scatter(x_drone, y_drone, color=colors[1], linewidth=2, label=legends[0])

        anim = animation.FuncAnimation(fig, animate, frames=length, interval=1000*delta_t,repeat = False)

        if save_img:
            print('Saving Video ...')
            writervideo = animation.FFMpegWriter(fps=f)
            video_path = image_path + name + ".mp4"
            anim.save(video_path, writervideo)
            plt.close()

        plt.show()

    def create_video_traj_vertical(self,drone_state_log,target_state_log, save_img = False, name = 'test',length=np.inf):
        '''
        Create a video without the trajectory in the y-z plane
        '''

        # Choose the length of the video in Timesteps
        # length = 110
        for state in drone_state_log:
            if len(state) > 0:
                length = int(min(length, len(state)))
        if name == 'RLS':
            legends = ['RLS-MPC']
        elif name == 'DAP':
            legends = ['DAP']
        elif name == 'SSO':
            legends = ['SS-OGD']
        elif name == 'ric':
            legends = ['Riccatitron']
        else:
            legends = ['LQR']

        colors = ['g',
                  'tab:purple',
                  'tab:red',
                  'tab:brown',
                  'tab:pink',
                  'tab:gray',
                  'tab:olive',
                  'tab:cyan'
                  ]

        target = target_state_log[0]
        drone_state = drone_state_log[0]
        plt.style.use("ggplot")
        # initializing a figure
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))

        # set limit for x and y axis
        # axes.set_ylim(-2.1, 2.1)
        # axes.set_xlim(0.4, 0.8)

        # plt.axis('equal')
        axes.set_xlabel('y[m]')
        axes.set_ylabel('z[m]')


        # Target Initial Plot
        self.my_line1 = axes.scatter(target[0,1,:],target[0,2,:], color=colors[0], s=255, label='Target', marker = "*")

        # Drone State Initial Plot
        self.my_line2 = axes.scatter(drone_state[0, 1, :], drone_state[0, 2, :], linewidth=2, color=colors[1], label=legends[0])

        handles, labels = plt.gca().get_legend_handles_labels()
        labels, ids = np.unique(labels, return_index=True)
        handles = [handles[i] for i in ids]
        order = [0, 1]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper left')

        def animate(i):
            # axes.clear()
            self.my_line1.remove()
            self.my_line2.remove()

            x_target = target[i,1,:]
            y_target = target[i,2,:]

            x_drone = drone_state[i, 1, :]
            y_drone = drone_state[i, 2, :]
            self.my_line1 = axes.scatter(x_target, y_target, color=colors[0], s=255, linestyle='dashed', label='Target', marker = "*")
            self.my_line2 = axes.scatter(x_drone, y_drone, color=colors[1], linewidth=2, label=legends[0])

        anim = animation.FuncAnimation(fig, animate, frames=length, interval=1000*delta_t,repeat = False)

        if save_img:
            print('Saving Video ...')
            writervideo = animation.FFMpegWriter(fps=f)
            video_path = image_path + name + ".mp4"
            anim.save(video_path, writervideo)
            plt.close()

        plt.show()

    def plot_trajectory_vertical(self, drone_state_log, target_state_log,length=np.inf, save_img=False, name=['traj']):
        '''
        Plot the trajectory plot in the y-z plane
        '''
        fig, ax = plt.subplots(figsize=(10, 10))

        # Choose the length of the video in Timesteps
        # length = 1000
        for state in drone_state_log:
            if len(state) > 0:
                length = int(min(length, len(state)))
        legends = []
        for _, n in enumerate(name):
            alg_name = n[15:18]
            if alg_name == 'RLS':
                legends.append('PLOT')
            elif alg_name == 'DAP':
                legends.append('DAP')
            elif alg_name == 'SSO':
                legends.append('SS-OGD')
            elif alg_name == 'ric':
                legends.append('Riccatitron')
            elif alg_name == 'FTL':
                legends.append('FTL')
            else:
                legends.append('LQR')

            ## Uncomment to pick different legends
            # alg_name = n[15:18]
            # if alg_name == 'RLS':
            #     num = n.split('_')[6].replace('gam', '')
            #     try:
            #         a = round(np.log(1.34 / (1 - float(num))) / np.log(len(action_DF_log[i] - 1)), 2)
            #     except:
            #         a = 1.0
            #
            #     legends.append(r"$\gamma_{" + str(a) + "} = $" + num)
            # elif alg_name == 'DAP':
            #     legends.append('DAP')
            # elif alg_name == 'SSO':
            #     legends.append('SS-OGD')
            # elif alg_name == 'ric':
            #     legends.append('Riccatitron')
            # else:
            #     legends.append('LQR')

        colors = ['g',
                  'tab:purple',
                  # 'tab:red',
                  'tab:cyan',
                  'tab:gray',
                  'tab:olive',
                  ]

        target = target_state_log[0]

        for idx in range(1, length):
            plt.plot([target[idx - 1, 1, :], target[idx, 1, :]], [target[idx - 1, 2, :], target[idx, 2, :]],
                     color=colors[0], linewidth=3, label='Target')
        for idx, state in enumerate(drone_state_log):
            for i in range(1, length):
                plt.plot([state[i - 1, 1, :], state[i, 1, :]], [state[i - 1, 2, :], state[i, 2, :]],
                         color=colors[1 + idx], linewidth=2, label=legends[idx])

                # Drone state markings
            plt.scatter(state[length - 1, 1, :], state[length - 1, 2, :], color=colors[1 + idx], s=400,
                        marker="*")
            plt.scatter(state[0, 1, :], state[0, 2, :], color=colors[1 + idx], s=255, label='Target',
                        marker="o")

        # Target markings
        plt.scatter(target[length - 1, 1, :], target[length - 1, 2, :], color=colors[0], s=400, label='Target',
                    marker="*")
        plt.scatter(target[0, 1, :], target[0, 2, :], color=colors[0], s=255, label='Target', marker="o")

        handles, labels = plt.gca().get_legend_handles_labels()
        labels, ids = np.unique(labels, return_index=True)
        handles = [handles[i] for i in ids]
        # order = [0, 2, 1, 3]
        plt.legend(handles, labels, loc='upper left')
        plt.axis('equal')
        ax.set_xlabel('$p_y$ [m]')
        ax.set_ylabel('$p_z$ [m]')

        plt.grid()
        fig = plt.gcf()
        if save_img:
            plt.savefig(image_path + name[0] + '_' + str(length).zfill(4) + '.eps', format='eps')
            print('Save figure to ' + 'images/' + name[0] + '_' + str(length).zfill(4) + '.eps')
            time.sleep(0.01)
            plt.close(fig)
        plt.show(block=True)


    def plot_trajectory(self, drone_state_log, target_state_log,length=np.inf, save_img=False, name=['traj']):
        '''
        Plot the Trajectory in the x-y plane
        '''
        fig, ax = plt.subplots(figsize=(6, 6.5))
        # Choose the length of the video in Timesteps
        length = 90
        for state in drone_state_log:
            if len(state) > 0:
                length = int(min(length, len(state)))
        # length = 500
        legends = []
        for _, n in enumerate(name):

            ## Uncomment for a different set of legends
            # alg_name = n[15:18]
            # if alg_name == 'RLS':
            #     legends.append('PLOT')
            # elif alg_name == 'DAP':
            #     legends.append('DAP')
            # elif alg_name == 'SSO':
            #     legends.append('SS-OGD')
            # elif alg_name == 'ric':
            #     legends.append('Riccatitron')
            # elif alg_name == 'FTL':
            #     legends.append('FTL')
            # else:
            #     legends.append('LQR')
            alg_name = n[15:18]
            if alg_name == 'RLS':
                num = n.split('_')[6].replace('gam', '')
                try:
                    a = round(np.log(1.34 / (1 - float(num))) / np.log(len(action_DF_log[0] - 1)), 2)
                except:
                    a = 1.0

                legends.append(r"$\gamma_{" + str(a) + "} = $" + num)
            elif alg_name == 'DAP':
                legends.append('DAP')
            elif alg_name == 'SSO':
                legends.append('SS-OGD')
            elif alg_name == 'ric':
                legends.append('Riccatitron')
            else:
                legends.append('LQR')

            ## Uncomment for a different set of legends
            # alg_name = n[15:18]
            # if alg_name == 'RLS':
            #     num = n.split('_')[7].replace('W', '')
            #     legends.append(r"$W = $" + num)
            # elif alg_name == 'DAP':
            #     legends.append('DAP')
            # elif alg_name == 'SSO':
            #     legends.append('SS-OGD')
            # elif alg_name == 'ric':
            #     legends.append('Riccatitron')
            # else:
            #     legends.append('LQR')

        colors = ['g',
                  'tab:purple',
                  'tab:red',
                  # 'tab:orange',
                  # 'tab:pink',
                  # 'tab:blue',
                  'tab:cyan',
                  'tab:gray',
                  'tab:olive',
                  ]

        target = target_state_log[0]

        for idx in range(1, length):
            plt.plot([target[idx - 1, 0, :], target[idx, 0, :]], [target[idx - 1, 1, :], target[idx, 1, :]],
                     color=colors[0], linewidth=3, label='Target')
        for idx, state in enumerate(drone_state_log):
            for i in range(1, length):
                plt.plot([state[i - 1, 0, :], state[i, 0, :]], [state[i - 1, 1, :], state[i, 1, :]],
                         color=colors[1 + idx], linewidth=2, label=legends[idx])

            # Drone state markings
            plt.scatter(state[length - 1, 0, :], state[length - 1, 1, :], color=colors[1 + idx], s=400,
                        marker="*")
            plt.scatter(state[0, 0, :], state[0, 1, :], color=colors[1 + idx], s=255, label='Target', marker="o")
        # Target markings
        plt.scatter(target[length-1, 0, :], target[length-1, 1, :], color=colors[0], s=400, label='Target', marker="*")
        plt.scatter(target[0, 0, :], target[0, 1, :], color=colors[0], s=255, label='Target', marker="o")

        handles, labels = plt.gca().get_legend_handles_labels()
        labels, ids = np.unique(labels, return_index=True)
        handles = [handles[i] for i in ids]
        # order = [0,1,2,3]
        # fig.legend([handles[idx] for idx in order], [labels[idx] for idx in order],ncol=2, loc='upper right',handlelength=0.5)
        fig.legend(handles, labels, ncol=2, loc='upper right',handlelength=0.3)
        plt.axis('equal')
        ax.set_xlabel('$p_x$ [m]')
        ax.set_ylabel('$p_y$ [m]')

        # ax.set_ylim(0, 1.0)
        # ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(-0.3, 1.7)
        ax.set_xlim(-0.5, 1.4)

        # ax.set_xlim(-0.1, 1.35)
        # ax.set_ylim(-0.1, 1.4)

        plt.grid()
        fig = plt.gcf()
        plt.subplots_adjust(left=0.19,bottom=0.13)

        if save_img:
            plt.savefig(image_path + name[0] + '_' + str(length).zfill(4) + '.eps', format='eps')
            print('Save figure to ' + 'images/' + name[0] + '_' + str(length).zfill(4) + '.eps')
            time.sleep(0.01)
            plt.close(fig)
        plt.show(block=True)



    def plot_regret(self, action_DF_log, disturbances_log, save_fig=False, name=None):
        '''
        Provide a Regret PLot
        '''
        T_max = np.inf
        stepsize = 2

        legends = []

        for _,n in enumerate(name):
            alg_name = n[15:18]
            if alg_name == 'RLS':
                # num = n[-8:-6]
                # num = num.replace('W','')
                # legends.append('W = ' + num)
                legends.append('PLOT')
            elif alg_name == 'DAP':
                legends.append('DAP')
            elif alg_name == 'SSO':
                legends.append('SS-OGD')
            elif alg_name == 'ric':
                legends.append('Riccatitron')
            elif alg_name == 'FTL':
                legends.append('FTL')
            else:
                legends.append('LQR')

        for action_DF in action_DF_log:
            if len(action_DF) > 0:
                T_max = int(min(T_max, len(action_DF)))

        regret_all = []
        optimal_DF = self.compute_offline_optimal_disturbance_feedback(T_max, disturbances_log[0])

        for T in range(2, T_max, stepsize):
            regret_T = []
            for action_DF in action_DF_log:
                regret = self.compute_regret(T, action_DF[:T], optimal_DF)[0]
                # regret = self.compute_regret(T, action_DF[:T], optimal_DF)[0]/np.log(T)
                regret_T.append(regret)
            regret_all.append(regret_T)


        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_ylim(0, 2600)
        colors = [
            'tab:purple',
            'tab:orange',
            'tab:red',
            'tab:blue',
            'tab:cyan',
            'tab:pink',
            'tab:gray',
            'tab:olive',
            'peru'
        ]

        for idx in range(len(action_DF_log)):
            plt.plot(0.1 * np.array(range(2, T_max, stepsize)), np.array(regret_all)[:, idx, :], color=colors[idx],
                     linewidth=3, label=legends[idx])

        plt.grid()
        plt.legend( loc='lower right', ncol = 3, handlelength=0.5)
        ax.set_xlabel('$T[s]$')
        ax.set_ylabel('Regret(T)')
        # ax.set_ylabel('Regret$(T)\;/\;\log(T)$')
        fig = plt.gcf()
        fig.subplots_adjust(left=0.16)
        # fig.suptitle('Regret')

        if save_fig:
            plt.savefig(image_path + 'regret' + '_' + str(T) + '.eps', format='eps')
            print('Save figure to ' + 'images/' + 'regret' + '.eps')
            time.sleep(0.01)
            plt.close(fig)
        plt.show(block=True)

    def plot_regret_gam(self, action_DF_log, disturbances_log, save_fig=False, name=None):
        '''
        Provide a Regret Plot for varying gamma values
        '''
        T_max = np.inf
        stepsize = 2

        legends = []

        for i,n in enumerate(name):
            alg_name = n[15:18]
            if alg_name == 'RLS':
                num = n.split('_')[6].replace('gam','')
                try:
                    a = round(np.log(1.34 / (1 -float(num)))/np.log(len(action_DF_log[i]-1)),2)
                except:
                    a = 1.0

                legends.append(r"$\gamma_{" + str(a) + "} = $" + num)
                # legends.append('PLOT')
            elif alg_name == 'DAP':
                legends.append('DAP')
            elif alg_name == 'SSO':
                legends.append('SS-OGD')
            elif alg_name == 'ric':
                legends.append('Riccatitron')
            elif alg_name == 'FTL':
                legends.append('FTL')
            else:
                legends.append('LQR')

        for action_DF in action_DF_log:
            if len(action_DF) > 0:
                T_max = int(min(T_max, len(action_DF)))
        T_max = 200
        regret_all = []
        optimal_DF = []
        for dis in disturbances_log:
            optimal_DF.append(self.compute_offline_optimal_disturbance_feedback(T_max, dis))
        for T in range(2, T_max, stepsize):
            regret_T = []
            for i,action_DF in enumerate(action_DF_log):
                regret = self.compute_regret(T, action_DF[:T], optimal_DF[i])[0]
                # regret = self.compute_regret(T, action_DF[:T], optimal_DF[i])[0]/T
                regret_T.append(regret)
            regret_all.append(regret_T)

        fig, ax = plt.subplots(figsize=(9, 9))

        colors = [
            'tab:purple',
            'tab:orange',
            'tab:red',
            'tab:blue',
            'tab:cyan',
            'tab:pink',
            'tab:gray',
            'tab:olive',
            'peru'
        ]

        for idx in range(len(action_DF_log)):
            plt.plot(0.1 * np.array(range(2, T_max, stepsize)), np.array(regret_all)[:, idx, :], color=colors[idx],
                     linewidth=3, label=legends[idx])

        plt.grid()
        plt.legend( loc='lower right', ncol = 3, handlelength=0.5)
        ax.set_xlabel('T[s]')
        ax.set_ylabel('Regret(T)')
        # ax.set_ylabel('Regret(T)/T')
        fig = plt.gcf()
        # fig.suptitle('Regret')

        if save_fig:
            plt.savefig(image_path + 'regret' + '_' + str(T) + '.eps', format='eps')
            print('Save figure to ' + 'images/' + 'regret' + '.eps')
            time.sleep(0.01)
            plt.close(fig)
        plt.show(block=True)


    def plot_gamma_example(self, action_DF_log, disturbances_log, save_img=False, name=None):
        '''
        Function to compare the effect of gammas
        '''

        T_max = []
        legends = []

        # TBD Automated
        data_n = 4
        for i, n in enumerate(name):
            alg_name = n[15:18]
            if alg_name == 'RLS':
                num = n.split('_')[6].replace('gam','')
                try:
                    a = round(np.log(1.34 / (1 -float(num)))/np.log(len(action_DF_log[i]-1)),2)
                except:
                    a = 1.0

                legends.append(r"$a = $" + str(a))
            else:
                legends.append('LQR')

        for i,action_DF in enumerate(action_DF_log):
                T_max.append(int(len(action_DF)))

        regret_all = []
        optimal_DF = []
        for i,dis in enumerate(disturbances_log):
            optimal_DF.append(np.array(self.compute_offline_optimal_disturbance_feedback(T_max[i], dis)))

        for i, action_DF in enumerate(action_DF_log):
            T = T_max[i]-1
            regret = self.compute_regret(T, action_DF[:T], optimal_DF[i])[0]

            regret_all.append(regret)
        fig, ax = plt.subplots(figsize=(7, 7))

        colors = [
            'tab:purple',
            'tab:orange',
            'tab:red',
            'tab:blue',
            'tab:cyan',
            'tab:pink',
            'tab:gray',
            'tab:olive',
            'peru'
        ]
        for idx in range(len(regret_all)):
            if idx<data_n:
                lab = legends[idx%data_n]
            else:
                lab = None
            
            plt.scatter((T_max[idx]-1)/f, regret_all[idx], color=colors[idx%data_n], linewidth=4,label=lab, marker="s")

        plt.grid()
        plt.legend(loc='upper left',ncol=1, handlelength=0.5)
        ax.set_xlabel('T[s]')
        ax.set_ylabel('Regret(T)')
        fig = plt.gcf()
        fig.subplots_adjust(left=0.16,bottom=0.10,right=0.97)

        # ax.set_ylim(600, 1600)
        ax.set_xlim(0, 220)

        if save_img:
            plt.savefig(image_path + 'regret' + '_' + str(T) + '.eps', format='eps')
            print('Save figure to ' + 'images/' + 'regret' + '.eps')
            time.sleep(0.01)
            plt.close(fig)
        plt.show(block=True)


    def compute_offline_optimal_disturbance_feedback(self, T, disturbances, look_ahead = 0):
        '''
        Computes the offline optimal disturbance feedback affine term for regret calculation
        '''
        K_d_all = self.compute_LQR_gain(T+look_ahead)
        action_log = []
        for t in range(0, T-1-look_ahead):
            action = np.zeros((4, 1))
            for i in range(t, min(len(disturbances),T-1)):
                action += -K_d_all[i-t]@disturbances[i]
            roll_rate = action[1].copy()
            pitch_rate = action[2].copy()
            action[1] = pitch_rate
            action[2] = roll_rate
            action_log.append(action)
        return np.array(action_log)

    def compute_LQR_gain(self,T):
        '''
        Calculate the feedback gains for future disturbances
        '''
        K_d_all = []
        A_tilde_power = np.eye(9)
        for _ in range(T):
            K_d = linalg.inv(self.R + self.B.T @ self.P_star @ self.B) @ self.B.T @ A_tilde_power.T @ self.P_star
            K_d_all.append(K_d)
            A_tilde_power = (self.A - self.B @ self.K_star) @ A_tilde_power
        return K_d_all

    def compute_regret(self, T, action_DF, optimal_DF):
        '''
        Calculates the Regret
        '''
        regret = 0
        for t in range(T - 1):
            regret += (action_DF[t, :] - optimal_DF[t, :]).T @ (self.R + self.B.T @ self.P_star @ self.B) @ (action_DF[t, :] - optimal_DF[t, :])

        return regret


if __name__ == '__main__':

    args = sys.argv

    # Get the list of available experiments
    list_of_experiments = glob.glob(save_path + '/*')
    latest_experiment = max(list_of_experiments, key=os.path.getctime)
    latest_experiment = latest_experiment.replace(save_path + '/','')
    latest_experiment = latest_experiment.replace('.npz','')

    # Load Data
    if len(sys.argv) > 1:
        algorithm_filenames = sys.argv[1:]
    else:
        algorithm_filenames = [latest_experiment]

    alg_names = []
    alg_full_names = []
    for _,name in enumerate(algorithm_filenames):
        # Algorithm name
        alg_names.append(name[15:18])
        alg_full_names.append(name)

    # The extracted algorithm data
    algorithm_logs = [load_data(file+'.npz') for file in algorithm_filenames]

    #  Initialization
    drone_state_log, target_state_log, action_log, action_DF_log, disturbances_log, cost_log, error_log = [], [], [], [], [], [], []

    plotter = Plotter()

    '''
    After this Line, please uncomment the function needed to provide the required type of plot
    '''

    ## Plotting multiple figures
    for idx, log in enumerate(algorithm_logs):
        # order of logs in load_data:
        # target_state_log 0, drone_state_log 1, action_log 2, action_DF_log 3,
        # disturbance_log 4, default_action_log 5, optimal_action_log 6
        target = log[0]
        drone = log[1] # will be filtered drone state if filtered = True in controller_default
        action = log[2]
        action_DF = log[3]
        disturbances = log[4]
        default_action = log[5]
        optimal_action = log[6]
        try:
            S_log = log[7]
        except:
            pass

        target_state_log.append(target)
        drone_state_log.append(drone)
        action_log.append(action)
        action_DF_log.append(action_DF)
        disturbances_log.append(disturbances)
        cost, error = plotter.compute_cost_and_error(drone, target, action)
        cost_log.append(cost)
        error_log.append(error)

        # State Plot
        # plotter.plot_state([drone, target], alg_names[idx] + '_states')

        # Control Input Plot
        # plotter.plot_action([action, default_action, optimal_action], alg_names[idx]+'_inputs')

        # Phase plot
        plotter.plot_trajectory(drone_state_log, target_state_log,name=alg_names[idx])

        # Regret Plot
        # plotter.plot_regret(action_DF_log, disturbances_log, name=alg_full_names)

        # Create a Video
        # plotter.create_video(drone_state_log, target_state_log, save_img=True, name = alg_names[idx])

        # plotter.create_video_vertical(drone_state_log, target_state_log, save_img=False, name=alg_names[idx])

        # plotter.create_video_traj(drone_state_log, target_state_log, save_img=False, name = alg_names[idx])

        # plotter.create_video_traj_vertical(drone_state_log, target_state_log, save_img=False, name=alg_names[idx])
        # plt.show(block=True)


    ## Plotting Multiple Experiments in one figure


    # Regret
    # plotter.plot_regret(action_DF_log, disturbances_log,save_fig=True, name=alg_full_names)
    # plotter.plot_regret_gam(action_DF_log, disturbances_log,save_fig=False, name=alg_full_names)

    # Trajectory
    # plotter.plot_trajectory(drone_state_log, target_state_log, save_img=True, name=alg_full_names)
    # plotter.plot_trajectory_vertical(drone_state_log, target_state_log, save_img=True, name=alg_full_names)

    # PLot State
    # plotter.plot_just_state_vertical(drone_state_log, target_state_log,save_img=True, name=alg_full_names)
    # plotter.plot_error_vertical(drone_state_log, target_state_log, name=alg_full_names)
    # plotter.plot_error(drone_state_log, target_state_log, name=alg_full_names)
    # plotter.plot_just_state(drone_state_log, target_state_log,save_img=True, name=alg_full_names)

    # Plot for gamma comparison
    # plotter.plot_gamma_example(action_DF_log, disturbances_log,save_img=True, name=alg_full_names)