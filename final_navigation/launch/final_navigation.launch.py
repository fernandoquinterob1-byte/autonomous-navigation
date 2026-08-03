import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    launch_sim = LaunchConfiguration('launch_sim')
    use_sim_time = LaunchConfiguration('use_sim_time')
    run_rrt_star = LaunchConfiguration('run_rrt_star')
    run_dwa = LaunchConfiguration('run_dwa')
    run_recorder = LaunchConfiguration('run_recorder')
    run_manager = LaunchConfiguration('run_manager')

    final_nav_share = get_package_share_directory('final_navigation')
    params_file = os.path.join(
        final_nav_share, 'config', 'navigation_params.yaml')

    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('el7009_diff_drive_robot'),
                'launch',
                'robot.launch.py',
            ])
        ),
        condition=IfCondition(launch_sim),
        launch_arguments={
            'rviz': 'True',
        }.items(),
    )

    costmap_node = Node(
        package='final_navigation',
        executable='costmap_node',
        name='costmap_node',
        output='screen',
        parameters=[
            params_file,
            {'use_sim_time': use_sim_time},
        ],
    )

    rrt_star_node = Node(
        package='final_navigation',
        executable='rrt_star_node',
        name='rrt_star_node',
        output='screen',
        condition=IfCondition(run_rrt_star),
        parameters=[
            params_file,
            {'use_sim_time': use_sim_time},
        ],
    )

    dwa_node = Node(
        package='final_navigation',
        executable='dwa_node',
        name='dwa_node',
        output='screen',
        condition=IfCondition(run_dwa),
        parameters=[
            params_file,
            {'use_sim_time': use_sim_time},
        ],
    )

    trajectory_recorder = Node(
        package='final_navigation',
        executable='trajectory_recorder',
        name='trajectory_recorder',
        output='screen',
        condition=IfCondition(run_recorder),
        parameters=[
            params_file,
            {'use_sim_time': use_sim_time},
        ],
    )

    navigation_manager = Node(
        package='final_navigation',
        executable='navigation_manager',
        name='navigation_manager',
        output='screen',
        condition=IfCondition(run_manager),
        parameters=[
            params_file,
            {'use_sim_time': use_sim_time},
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'launch_sim',
            default_value='false',
            description='Launch the Gazebo/RViz robot simulation.'),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time.'),
        DeclareLaunchArgument(
            'run_rrt_star',
            default_value='false',
            description='Launch the custom RRT* global planner.'),
        DeclareLaunchArgument(
            'run_dwa',
            default_value='false',
            description='Launch the custom DWA local controller.'),
        DeclareLaunchArgument(
            'run_recorder',
            default_value='false',
            description='Launch trajectory recorder and metrics publisher.'),
        DeclareLaunchArgument(
            'run_manager',
            default_value='false',
            description='Launch global navigation status manager.'),
        sim_launch,
        costmap_node,
        rrt_star_node,
        dwa_node,
        trajectory_recorder,
        navigation_manager,
    ])
