from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.actions import LogInfo, RegisterEventHandler
from launch.event_handlers import OnProcessStart

def generate_launch_description():

    # sim_node = Node(
    #     package='crazyflie_online_tracker',
    #     executable='linear_simulator',
    #     name='linear_simulator',
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
        
       controller_node, 

       target_node
        
])