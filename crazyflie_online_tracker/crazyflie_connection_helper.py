#!/usr/bin/env python3
import rospy
from cflib.utils import uri_helper
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie


class CrazyflieConnectionHelper:
    """
    Establish connection with the crazyflie, which can be used by both the crazyflie actuator and the crazyflie onboard state estimator(flow deck required)
    """
    def __init__(self):
        vehicle_name = rospy.get_param('vehicle_name')
        if vehicle_name=='CF59': #CFAI
            self.URI = uri_helper.uri_from_env(default='radio://0/0/2M/E7E7E7E73B') #3B=59
        elif vehicle_name=='CF11':
            self.URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E70B')
        elif vehicle_name=='CF13':
            self.URI = uri_helper.uri_from_env(default='radio://0/96/2M/E7E7E7E70D')
        elif vehicle_name=='CF14':
            self.URI = uri_helper.uri_from_env(default='radio://0/104/2M/E7E7E7E70E')
        elif vehicle_name=='CF19':
            self.URI = uri_helper.uri_from_env(default='radio://1/20/2M/E7E7E7E713')
        elif vehicle_name == 'CF04':
            self.URI = uri_helper.uri_from_env(default='radio://0/24/2M/E7E7E7E704')
        else:
            rospy.logerr("unknown vehicle name.")
        cflib.crtp.init_drivers()
        self.scf = SyncCrazyflie(self.URI, cf=Crazyflie(rw_cache='./cache'))
        self.cf = self.scf.cf
        self.scf.open_link()

    def get_scf(self):
        if self.scf is None:
            rospy.loginfo("SyncCrazyflie is not initialized.")
        return self.scf

    def get_cf(self):
        if self.cf is None:
            rospy.loginfo("Crazyflie is not initialized.")
        return self.cf


helper = CrazyflieConnectionHelper()
