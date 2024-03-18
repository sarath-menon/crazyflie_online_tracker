from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'publish_frequency',
            default_value='10.0',
            description='Publish frequency'
        ),
        DeclareLaunchArgument(
            'wait_for_simulator_initialization',
            default_value='false',
            description='Wait for simulator initialization'
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
        Node(
            package='crazyflie_online_tracker',
            executable='controller_RLS',
            name='controller_RLS',
            parameters=[{
                'publish_frequency': LaunchConfiguration('publish_frequency'),
                'wait_for_simulator_initialization': LaunchConfiguration('wait_for_simulator_initialization'),
                'add_initial_target': LaunchConfiguration('add_initial_target'),
                'synchronize_target': LaunchConfiguration('synchronize_target'),
                'filename': LaunchConfiguration('filename')
            }]
        ),
        Node(
            package='crazyflie_online_tracker',
            executable='linear_simulator',
            name='linear_simulator',
        ),
        Node(
            package='crazyflie_online_tracker',
            executable='state_estimator_target_virtual',
            name='state_estimator_target_virtual',
        )
])