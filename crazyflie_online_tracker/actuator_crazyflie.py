#!/usr/bin/env python3
import rospy
import struct
import numpy as np
from cflib.crtp.crtpstack import CRTPPacket, CRTPPort

from crazyflie_online_tracker_interfaces.msg import CommandCF, CommandOuter, ControllerState
from controller import ControllerStates
from actuator import Actuator, dfallPacket
import crazyflie_connection_helper

CF_PACKET_DECODER_DFALL_TYPE = 8 # defined on the crazyflie dfall firmware


class CrazyflieActuator(Actuator):
    def __init__(self):
        super().__init__()
        self.cf = crazyflie_connection_helper.helper.get_cf()
        rospy.on_shutdown(self.safe_shutdown)
        rospy.init_node('crazyflie_actuator')
        self.command_sub = rospy.Subscriber("controllerCommand", CommandOuter, self.callback_command)
        self.actuator_pub = rospy.Publisher('cfCommand', CommandCF, queue_size=10)
        self.controller_state_pub = rospy.Publisher('controllerState', ControllerState, queue_size=1, latch=True)
        self.request_takeoff()
        rospy.sleep(5)
        rospy.spin()

    def request_takeoff(self):
        msg = ControllerState()
        msg.state = ControllerStates.takeoff
        self.controller_state_pub.publish(msg)

    def request_land(self):
        msg = ControllerState()
        msg.state = ControllerStates.landing
        self.controller_state_pub.publish(msg)

    def request_stop(self):
        msg = ControllerState()
        msg.state = ControllerStates.stop
        self.controller_state_pub.publish(msg)

    def safe_shutdown(self):
        rospy.loginfo("rospy shutting down. Send stop setpoint to crazyflie.")
        self.request_stop()
        self.cf.commander.send_stop_setpoint()

    def callback_command(self, data):
        rospy.loginfo('setpoint received')
        # if the experiment ends(controller sent the last command), land the crazyflie
        if data.is_last_command:
            self.request_land()
            return
        
        # dfall packet contains: cmd1234, x mode, x value, y mode, y value, z mode, z value, yaw mode, yaw value
        thrust = data.thrust
        command = dfallPacket()
        command.cmd1 = self.thrust_newton_to_cmd(thrust/4)
        command.cmd2 = self.thrust_newton_to_cmd(thrust/4)
        command.cmd3 = self.thrust_newton_to_cmd(thrust/4)
        command.cmd4 = self.thrust_newton_to_cmd(thrust/4)
        # TODO: check that following the angular rate commands won't cause angles to exceed the safety bound(70 deg in dfall)
        command.x_value = np.rad2deg(-data.omega.x)
        command.y_value = np.rad2deg(data.omega.y)
        command.yaw_value = np.rad2deg(data.omega.z)
        self.send_dfall_packet(command)
        commandCF = CommandCF() # only for debugging purpose
        commandCF.x = command.x_value
        commandCF.y = command.y_value
        commandCF.z = command.cmd1
        commandCF.yaw = command.yaw_value
        self.actuator_pub.publish(commandCF)
        # rospy.loginfo('send command to crazyflie: ' + str(commandCF))

    def send_dfall_packet(self, command):
        # copied from dfall_pkg/crazyradio/CrazyRadio.py: CrazyRadioClient._send_to_commander()
        # B:unsigned int, H:unsigned short, f:float
        modes_as_uint16 = (command.yaw_mode << 9) | (command.z_mode << 6) | (command.y_mode << 3) | (command.x_mode << 0)
        pk = CRTPPacket()
        pk.port = CRTPPort.COMMANDER_GENERIC
        pk.data = struct.pack('<BHHHHHffff', CF_PACKET_DECODER_DFALL_TYPE, 
                              modes_as_uint16, command.cmd1, command.cmd2, command.cmd3, command.cmd4, 
                              command.x_value, command.y_value, command.z_value, command.yaw_value)
        self.cf.send_packet(pk)



if __name__ == '__main__':
    cf_actuator = CrazyflieActuator()
    if crazyflie_connection_helper.helper.scf.is_link_open():
        crazyflie_connection_helper.helper.scf.close_link()
