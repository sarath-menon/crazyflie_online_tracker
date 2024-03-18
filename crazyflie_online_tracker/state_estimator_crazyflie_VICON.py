import rospy
import numpy as np
from scipy.spatial.transform import Rotation
from crazyflie_online_tracker_interfaces.msg import CrazyflieState, CommandOuter
from dfall_pkg.msg import ViconData
from state_estimator import StateEstimator


class CrazyflieStateEstimatorVICON(StateEstimator):

    def __init__(self):
        super().__init__()
        rospy.init_node('crazyflie_state_estimator_VICON')
        self.vehicle_name = rospy.get_param('vehicle_name')
        self.state_pub = rospy.Publisher('crazyflieState', CrazyflieState, queue_size=10)
        self.VICON_sub = rospy.Subscriber('/dfall/ViconDataPublisher/ViconData', ViconData, self.VICON_callback)
        self.last_state = np.zeros((9, 1)) # record previous state measurement to estimate velocity
        self.is_initialized = False
        self.VICON_freq = 200
        self.v_threshold = 20 # if v>20m/s, the state estimation is discarded.
        self.roll_pitch_threshold = np.pi/2

    def state_vec_to_msg(self, state_vector, yaw, pitch, roll):
        state = CrazyflieState()
        state.pose.position.x = state_vector[0]
        state.pose.position.y = state_vector[1]
        state.pose.position.z = state_vector[2]
        r = Rotation.from_euler('ZYX', [yaw, pitch, roll])
        quaternion = r.as_quat()
        state.pose.orientation.x = quaternion[0]
        state.pose.orientation.y = quaternion[1]
        state.pose.orientation.z = quaternion[2]
        state.pose.orientation.w = quaternion[3]
        state.velocity.linear.x = state_vector[3]
        state.velocity.linear.y = state_vector[4]
        state.velocity.linear.z = state_vector[5]
        return state

    def publish_state(self, state):
        self.state_pub.publish(state)

    def VICON_callback(self, data):
        vehicle_states_all = data.crazyflies
        for vehicle_state in vehicle_states_all:
            if vehicle_state.vehicle_name == self.vehicle_name:
                break
        state_vector = np.zeros((9, 1))
        state_vector[0] = vehicle_state.x
        state_vector[1] = vehicle_state.y
        state_vector[2] = vehicle_state.z
        if self.is_initialized:
            state_vector[3] = (vehicle_state.x - self.last_state[0])*self.VICON_freq
            state_vector[4] = (vehicle_state.y - self.last_state[1])*self.VICON_freq
            state_vector[5] = (vehicle_state.z - self.last_state[2])*self.VICON_freq

            if (np.abs(state_vector[3]) >= self.v_threshold or np.abs(state_vector[4]) >= self.v_threshold
                    or np.abs(state_vector[5]) >= self.v_threshold):
                return
        else:
            self.is_initialized = True

        roll = vehicle_state.roll
        pitch = vehicle_state.pitch
        yaw = vehicle_state.yaw

        if np.abs(roll) >= self.roll_pitch_threshold or np.abs(pitch) >= self.roll_pitch_threshold:
            return
        state_vector[6] = roll
        state_vector[7] = pitch
        state_vector[8] = yaw
        self.last_state = state_vector # this stays, we just update the estimated one
        state_msg = self.state_vec_to_msg(state_vector, yaw, pitch, roll)
        self.state_pub.publish(state_msg)
        # print(f"state: {state_vector}")

def main(args=None):
    crazyflie_state_estimator = CrazyflieStateEstimatorVICON()
    # f = 200 # publish frequency to crazyflieState topic
    # rate = rospy.Rate(f)
    while not rospy.is_shutdown():
        rospy.spin()
        # rate.sleep()


if __name__ == '__main__':
    main()