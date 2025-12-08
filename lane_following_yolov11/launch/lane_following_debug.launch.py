# ============================================
# launch/lane_following_debug.launch.py
# (Launch with control disabled for testing)
# ============================================
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory('lane_following_yolov11')
    
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/path/to/your/yolov11n-seg.pt',
        description='Path to YOLOv11 model'
    )
    
    # Lane segmentation node with debug logging
    lane_segmentation_node = Node(
        package='lane_following_yolov11',
        executable='lane_segmentation_node',
        name='yolov11_lane_segmentation_node',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'confidence_threshold': 0.5,
            'iou_threshold': 0.45,
            'image_topic': '/zed2/zed_node/left/image_rect_color',
            'imgsz': 640,
            'device': 'cuda',
            'half_precision': True,
            'left_lane_class_id': 0,
            'right_lane_class_id': 1,
        }],
        arguments=['--ros-args', '--log-level', 'debug']
    )
    
    # Controller with control DISABLED for testing
    lane_controller_node = Node(
        package='lane_following_yolov11',
        executable='lane_controller_node',
        name='lane_controller_node',
        output='screen',
        parameters=[{
            'forward_speed': 1.0,
            'Kp': 0.005,
            'Kd': 0.01,
            'Ki': 0.0001,
            'max_angular_velocity': 0.5,
            'cmd_vel_topic': '/cmd_vel',
            'enable_control': False,  # DISABLED for testing
            'lateral_error_threshold': 0.3,
            'log_interval': 1,  # Log every update in debug mode
        }],
        arguments=['--ros-args', '--log-level', 'debug']
    )
    
    return LaunchDescription([
        model_path_arg,
        lane_segmentation_node,
        lane_controller_node,
    ])