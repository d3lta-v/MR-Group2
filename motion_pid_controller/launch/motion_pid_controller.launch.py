#!/usr/bin/python3
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.substitutions import TextSubstitution

def generate_launch_description():
    
   #declare arg
    forward_speed_arg = 1.0 # This is capped by the max speed of the robot which is 1
    Kp_val = 0.2 # initialize Kp
    Kd_val = 0.08  # initialize Kd
    Ki_val = 0.00001  # initialize Ki
    Kp_angle_val = 1.5

    #node 
    motion_pid_controller_py = Node(
        package='motion_pid_controller',
        executable='motion_pid_controller.py',
        name='motion_pid_controller',
        output='screen',
        parameters=[{"forward_speed": forward_speed_arg,
                     "Kp" : Kp_val,
                     "Kd" : Kd_val,
                     "Ki" : Ki_val,
                     "Kp_angle" : Kp_angle_val,
                     }]
    )

    ld = LaunchDescription()
    
    ld.add_action(motion_pid_controller_py)
    
    return ld
    