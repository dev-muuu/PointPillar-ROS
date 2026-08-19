#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_dir = get_package_share_directory('pointpillars_ros')
    rviz_config = os.path.join(pkg_dir, 'launch', 'pointpillars.rviz')

    return LaunchDescription([
        # PointPillars ROS Node
        Node(
            package='pointpillars_ros',
            executable='ros.py',
            name='pointpillars_ros',
            output='screen',
        ),
        # RViz2
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config] if os.path.exists(rviz_config) else [],
            output='screen',
        ),
    ])
