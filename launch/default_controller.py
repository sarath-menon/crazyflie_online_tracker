from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.actions import LogInfo, RegisterEventHandler
from launch.event_handlers import OnProcessStart
import time

def generate_launch_description():

    controller_node =  Node(
            package='crazyflie_online_tracker',
            executable='controller_default',
            name='controller_default',
            parameters=[{
                'clock_frequency': LaunchConfiguration('clock_frequency'),
                'filename': LaunchConfiguration('filename'),
            }]
    )

    target_node = Node(
            package='crazyflie_online_tracker',
            executable='state_estimator_target_virtual',
            name='state_estimator_target_virtual',
            parameters=[{
                'wait_for_drone_ready': LaunchConfiguration('wait_for_drone_ready'),
                'clock_frequency': LaunchConfiguration('clock_frequency'),
            }]
    )

    


    return LaunchDescription([
        DeclareLaunchArgument(
            'clock_frequency',
            default_value='1000.0',
            description='Clock frequency'
        ),
        DeclareLaunchArgument(
            'filename',
            default_value='LQR',
            description='Filename'
        ),
        DeclareLaunchArgument(
            'wait_for_drone_ready',
            default_value='False',
            description='Wait for drone ready'
        ),
        
       controller_node, 

       target_node
        
])