#!/usr/bin/env python3
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

# copied from the dfall package
CF_ONBOARD_CONTROLLER_MODE_OFF            = 0
CF_ONBOARD_CONTROLLER_MODE_ANGULAR_RATE   = 1

motor_poly = [5.484560e-4, 1.032633e-6, 2.130295e-11]
motor_cmd_min = 1000
motor_cmd_max = 65000


class Actuator(ABC):
    '''
    Abstract class for the crazyflie actuator, which converts the outer controller commands to data packets that can be executed by the onboard controller.
    '''
    def __init__(self):
        self.actuator_pub = None # for debugging
        self.command_sub = None # subscribe to ControllerCommand

    def thrust_newton_to_cmd(self, thrust):
        # copied from dfall_pkg/src/nodes/DefaultControllerService.cpp: float computeMotorPolyBackward(float thrust)
        cmd_16bit = (-motor_poly[1] + np.sqrt(max(0,motor_poly[1]**2 - 4 * motor_poly[2] * (motor_poly[0] - thrust)))) / (2 * motor_poly[2])
        if cmd_16bit < motor_cmd_min:
            cmd_16bit = motor_cmd_min
        elif cmd_16bit > motor_cmd_max:
            cmd_16bit = motor_cmd_max
        cmd_16bit = np.ushort(cmd_16bit)
        return cmd_16bit
    
    @abstractmethod
    def callback_command(self, data):
        pass

@dataclass
class dfallPacket:
    cmd1: float = 0.0
    cmd2: float = 0.0
    cmd3: float = 0.0
    cmd4: float = 0.0
    x_value: float = 0.0
    y_value: float = 0.0
    z_value: float = 0.0
    yaw_value: float = 0.0
    x_mode: int = CF_ONBOARD_CONTROLLER_MODE_ANGULAR_RATE
    y_mode: int = CF_ONBOARD_CONTROLLER_MODE_ANGULAR_RATE
    z_mode: int = CF_ONBOARD_CONTROLLER_MODE_OFF
    yaw_mode: int = CF_ONBOARD_CONTROLLER_MODE_ANGULAR_RATE
    
