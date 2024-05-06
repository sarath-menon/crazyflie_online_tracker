from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.actions import LogInfo, RegisterEventHandler
from launch.event_handlers import OnProcessStart

def generate_launch_description():

    controller_node =  Node(
            package='crazyflie_online_tracker',
            executable='controller_RLS',
            name='controller_RLS',
            parameters=[{
                'filename': LaunchConfiguration('filename'),
                'clock_frequency': LaunchConfiguration('clock_frequency'),
                'synchronize_target': LaunchConfiguration('synchronize_target'),
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
            'publish_frequency',
            default_value='10.0',
            description='Publish frequency'
        ),
        DeclareLaunchArgument(
            'clock_frequency',
            default_value='1000.0',
            description='Clock frequency'
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