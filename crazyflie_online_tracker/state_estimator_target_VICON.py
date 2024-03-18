#!/usr/bin/env python3
import rospy
import numpy as np
from scipy.spatial.transform import Rotation as R

from crazyflie_online_tracker_interfaces.msg import TargetState
from dfall_pkg.msg import ViconData
from state_estimator import StateEstimator


class TargetStateEstimatorVICON(StateEstimator):
    def __init__(self):
        super().__init__()
        rospy.init_node('target_state_estimator')
        self.state_pub = rospy.Publisher('targetState', TargetState, queue_size=10)
        self.VICON_sub = rospy.Subscriber('/dfall/ViconDataPublisher/ViconData', ViconData, self.VICON_callback)
        self.last_state = np.zeros((3, 1)) # record previous state measurement to estimate velocity
        self.VICON_freq = 200
        self.target_name = rospy.get_param('target_name')
        self.v_threshold = 20 # if v>20m/s in any direction, the state estimation is discarded.
        self.is_initialized = False

    def publish_state(self, state):
        self.state_pub.publish(state)
    
    def VICON_callback(self, data):
        vehicle_name = self.target_name
        vehicle_states_all = data.crazyflies
        for vehicle_state in vehicle_states_all:
            if vehicle_state.vehicle_name == vehicle_name:
                break
        state_vector = np.zeros((3, 1))
        state_vector[0] = vehicle_state.x
        state_vector[1] = vehicle_state.y
        state_vector[2] = vehicle_state.z

        self.last_state = state_vector
        # print('vehicle state vector', state_vector)
        state = TargetState()
        # offset the target to avoid collision
        state.pose.position.x = vehicle_state.x - 0.8 
        state.pose.position.y = vehicle_state.y
        state.pose.position.z = vehicle_state.z
        state.velocity.linear.x = 0
        state.velocity.linear.y = 0
        state.velocity.linear.z = 0
        r = R.from_euler('ZYX', [0, 0, 0])
        quaternion = r.as_quat()
        state.pose.orientation.x = quaternion[0]
        state.pose.orientation.y = quaternion[1]
        state.pose.orientation.z = quaternion[2]
        state.pose.orientation.w = quaternion[3]
        self.state_pub.publish(state)
        # rospy.loginfo('Crazyflie state published.')

if __name__ == '__main__':
    target_state_estimator = TargetStateEstimatorVICON()
    while not rospy.is_shutdown():
        rospy.spin()
