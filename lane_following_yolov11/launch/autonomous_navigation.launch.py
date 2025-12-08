from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory('lane_following_yolov11')
    
    # Declare launch arguments
    cone_model_arg = DeclareLaunchArgument(
        'cone_model_path',
        default_value=os.path.join(pkg_dir, 'models', 'cone_model.pt'),
        description='Path to cone detection model'
    )
    
    cone_map_arg = DeclareLaunchArgument(
        'cone_map_file',
        default_value=os.path.join(pkg_dir, 'config', 'cone_map.json'),
        description='Path to pre-mapped cone positions'
    )
    
    goal_x_arg = DeclareLaunchArgument(
        'goal_x',
        default_value='2.5',
        description='Goal X coordinate'
    )
    
    goal_y_arg = DeclareLaunchArgument(
        'goal_y',
        default_value='0.0',
        description='Goal Y coordinate'
    )
    
    goal_z_arg = DeclareLaunchArgument(
        'goal_z',
        default_value='0.0',
        description='Goal Z coordinate'
    )
    
    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cuda',
        description='Device for inference (cuda or cpu)'
    )
    
    localization_arg = DeclareLaunchArgument(
        'localization_enabled',
        default_value='true',
        description='Enable cone-based localization'
    )
    
    # Autonomous navigation node
    autonomous_nav_node = Node(
        package='lane_following_yolov11',
        executable='autonomous_cone_navigation_node',
        name='autonomous_cone_navigation_node',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('cone_model_path'),
            'cone_map_file': LaunchConfiguration('cone_map_file'),
            'confidence_threshold': 0.6,
            'image_topic': '/zed2/zed_node/left/image_rect_color',
            'camera_info_topic': '/zed2/zed_node/left/camera_info',
            'depth_topic': '/zed2/zed_node/depth/depth_registered',
            'device': LaunchConfiguration('device'),
            'cone_class_id': 0,
            'camera_frame': 'zed2_left_camera_optical_frame',
            'robot_frame': 'base_link',
            'map_frame': 'map',
            'use_depth_camera': True,
            'cone_height': 0.3,
            'max_detection_distance': 10.0,
            'min_detection_distance': 0.5,
            'goal_x': LaunchConfiguration('goal_x'),
            'goal_y': LaunchConfiguration('goal_y'),
            'goal_z': LaunchConfiguration('goal_z'),
            'goal_tolerance': 0.1,
            'max_linear_velocity': 0.5,
            'max_angular_velocity': 0.5,
            'localization_enabled': LaunchConfiguration('localization_enabled'),
            'min_cone_matches_for_localization': 3,
            'cone_match_threshold': 0.8,
        }],
        arguments=['--ros-args', '--log-level', 'info']
    )
    
    return LaunchDescription([
        cone_model_arg,
        cone_map_arg,
        goal_x_arg,
        goal_y_arg,
        goal_z_arg,
        device_arg,
        localization_arg,
        autonomous_nav_node,
    ])