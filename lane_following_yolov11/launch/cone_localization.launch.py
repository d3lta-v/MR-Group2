# ============================================
# launch/cone_localization.launch.py
# ============================================
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory('lane_following_yolov11')
    
    # Declare launch arguments
    cone_model_path_arg = DeclareLaunchArgument(
        'cone_model_path',
        default_value='/path/to/your/cone_yolov11.pt',
        description='Path to YOLOv11 cone detection model weights'
    )
    
    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cuda',
        description='Device for inference (cuda or cpu)'
    )
    
    use_depth_arg = DeclareLaunchArgument(
        'use_depth_camera',
        default_value='true',
        description='Use depth camera for distance estimation'
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
            'use_depth_camera': LaunchConfiguration('use_depth_camera'),
            'publish_rate': 10.0,
        }],
        arguments=['--ros-args', '--log-level', 'info']
    )
    
    return LaunchDescription([
        cone_model_path_arg,
        device_arg,
        use_depth_arg,
        cone_localization_node,
    ])