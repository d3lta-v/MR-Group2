# ============================================
# launch/full_system.launch.py
# (Launch both lane following and cone localization)
# ============================================
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory('lane_following_yolov11')
    
    # Declare launch arguments
    lane_model_arg = DeclareLaunchArgument(
        'lane_model_path',
        default_value='/path/to/lane_model.pt',
        description='Path to lane segmentation model'
    )
    
    cone_model_arg = DeclareLaunchArgument(
        'cone_model_path',
        default_value='/path/to/cone_model.pt',
        description='Path to cone detection model'
    )
    
    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cuda',
        description='Device for inference'
    )
    
    enable_control_arg = DeclareLaunchArgument(
        'enable_control',
        default_value='true',
        description='Enable robot control'
    )
    
    # Lane segmentation node
    lane_segmentation_node = Node(
        package='lane_following_yolov11',
        executable='lane_segmentation_node',
        name='yolov11_lane_segmentation_node',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('lane_model_path'),
            'confidence_threshold': 0.5,
            'iou_threshold': 0.45,
            'image_topic': '/zed2/zed_node/left/image_rect_color',
            'imgsz': 640,
            'device': LaunchConfiguration('device'),
            'half_precision': True,
            'left_lane_class_id': 0,
            'right_lane_class_id': 1,
        }]
    )
    
    # Lane controller node
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
            'lateral_error_threshold': 0.3,
            'log_interval': 10,
        }]
    )
    
    # Cone localization node
    cone_localization_node = Node(
        package='lane_following_yolov11',
        executable='cone_localization_node',
        name='cone_localization_node',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('cone_model_path'),
            'confidence_threshold': 0.6,
            'iou_threshold': 0.45,
            'image_topic': '/zed2/zed_node/left/image_rect_color',
            'camera_info_topic': '/zed2/zed_node/left/camera_info',
            'depth_topic': '/zed2/zed_node/depth/depth_registered',
            'imgsz': 640,
            'device': LaunchConfiguration('device'),
            'half_precision': True,
            'cone_class_id': 0,
            'map_frame': 'map',
            'camera_frame': 'zed2_left_camera_optical_frame',
            'robot_frame': 'base_link',
            'max_detection_distance': 20.0,
            'min_detection_distance': 0.5,
            'cone_merge_distance': 0.5,
            'cone_height': 0.3,
            'use_depth_camera': True,
            'publish_rate': 10.0,
        }]
    )
    
    return LaunchDescription([
        lane_model_arg,
        cone_model_arg,
        device_arg,
        enable_control_arg,
        lane_segmentation_node,
        lane_controller_node,
        cone_localization_node,
    ])