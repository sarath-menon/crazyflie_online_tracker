from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.actions import LogInfo, RegisterEventHandler
from launch.event_handlers import OnProcessStart
import time

def generate_launch_description():

    # sim_node = Node(
    #     package='crazyflie_online_tracker',
    #     executable='linear_simulator',
    #     name='linear_simulator',003] [controller_default]: Trajectory data has been saved to/home/sarath/drones/catkin_ws/src/crazyflie_online_tracker/data/20240501135751_RLS_circular_target_T50_f50_mode0.npz
    # ) 

    controller_node =  Node(
            package='crazyflie_online_tracker',
            executable='controller_default',
            name='controller_default',
            parameters=[{
                'publish_frequency': LaunchConfiguration('publish_frequency'),
                'add_initial_target': LaunchConfiguration('add_initial_target'),
                'synchronize_target': LaunchConfiguration('synchronize_target'),
                'filename': LaunchConfiguration('filename')
            }]
    )

    target_node = Node(
            package='crazyflie_online_tracker',
            executable='state_estimator_target_virtual',
            name='state_estimator_target_virtual',
            parameters=[{
                'wait_for_drone_ready': LaunchConfiguration('wait_for_drone_ready')
            }]
    )

    


    return LaunchDescription([
        DeclareLaunchArgument(
            'publish_frequency',
            default_value='10.0',
            description='Publish frequency'
        ),
        DeclareLaunchArgument(
            'add_initial_target',
            default_value='false',
            description='Add initial target'
        ),
        DeclareLaunchArgument(
            'synchronize_target',
            default_value='false',
            description='Synchronize target'
        ),
        DeclareLaunchArgument(
            'filename',
            default_value='RLS',
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