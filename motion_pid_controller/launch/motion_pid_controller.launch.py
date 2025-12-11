#!/usr/bin/python3
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.substitutions import TextSubstitution

def generate_launch_description():
    
    motion_pid_node_py = Node(
        package='motion_pid_controller',
        executable='motion_pid_controller',
        name='motion_pid_node',
        output='screen',
        parameters=[{
            "forward_speed": 1.0, # This is capped by the max speed of the robot which is 1
            "Kp" : 0.2, #0.2
            "Kd" : 0.09, #0.12975
            "Ki" : 0.00001, #0.00001
            "Kp_angle" : 2.0,
            "target_xpos" : 14.0,
            "LKp" : 0.95, #0.95
            "LKd" : 0.0,
            "LKi" : 0.0,
            "use_pose" : False,
            "use_ext_data" : False   # Enable use of external CTE and angular error data, via /cte and /angle_error topics
        }]
    )

    ld = LaunchDescription()
    
    ld.add_action(motion_pid_node_py)
    
    return ld
    