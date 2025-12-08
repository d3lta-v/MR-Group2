# ============================================
# launch/lane_following_yolov11.launch.py
# ============================================
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('lane_following_yolov11')
    
    # Declare launch arguments
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/path/to/your/yolov11n-seg.pt',  # YOLOv11 nano segmentation model
        description='Path to YOLOv11 segmentation model weights'
    )
    
    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cuda',  # Use 'cpu' if no GPU
        description='Device for inference (cuda or cpu)'
    )
    
    enable_control_arg = DeclareLaunchArgument(
        'enable_control',
        default_value='true',
        description='Enable robot control output'
    )
    
    # Lane segmentation node
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
            'device': LaunchConfiguration('device'),
            'half_precision': True,  # Use FP16 for faster inference on GPU
            'left_lane_class_id': 0,
            'right_lane_class_id': 1,
        }],
        arguments=['--ros-args', '--log-level', 'info']
    )
    
    # Lane following controller node
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
            'enable_control': LaunchConfiguration('enable_control'),
            'lateral_error_threshold': 0.3,  # 30% of image width
            'log_interval': 10,  # Log detailed info every 10 control updates
        }],
        arguments=['--ros-args', '--log-level', 'info']
    )
    
    return LaunchDescription([
        model_path_arg,
        device_arg,
        enable_control_arg,
        lane_segmentation_node,
        lane_controller_node,
    ])