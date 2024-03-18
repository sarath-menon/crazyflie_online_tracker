from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'crazyflie_online_tracker'



setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sarath',
    maintainer_email='sarathmenon.downey@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'linear_simulator = crazyflie_online_tracker.linear_simulator:main',
            'controller_RLS = crazyflie_online_tracker.controller_RLS:main',
            'state_estimator_target_virtual = crazyflie_online_tracker.state_estimator_target_virtual:main',
        ],
    },
)
