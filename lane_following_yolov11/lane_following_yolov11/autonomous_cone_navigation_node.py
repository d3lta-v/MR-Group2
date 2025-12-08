#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, PoseStamped, Twist, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from std_msgs.msg import ColorRGBA
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import json
import threading
from pathlib import Path

from ultralytics import YOLO
import tf2_ros
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
import tf2_geometry_msgs
from tf_transformations import quaternion_from_euler, euler_from_quaternion


class AutonomousConeNavigationNode(Node):
    def __init__(self):
        super().__init__('autonomous_cone_navigation_node')
        
        # Declare parameters
        self.declare_parameter('model_path', 'path/to/cone_model.pt')
        self.declare_parameter('cone_map_file', 'path/to/cone_map.json')
        self.declare_parameter('confidence_threshold', 0.6)
        self.declare_parameter('image_topic', '/zed2/zed_node/left/image_rect_color')
        self.declare_parameter('camera_info_topic', '/zed2/zed_node/left/camera_info')
        self.declare_parameter('depth_topic', '/zed2/zed_node/depth/depth_registered')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('cone_class_id', 0)
        self.declare_parameter('camera_frame', 'zed2_left_camera_optical_frame')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('use_depth_camera', True)
        self.declare_parameter('cone_height', 0.3)
        self.declare_parameter('max_detection_distance', 10.0)
        self.declare_parameter('min_detection_distance', 0.5)
        
        # Navigation parameters
        self.declare_parameter('goal_x', 2.5)
        self.declare_parameter('goal_y', 0.0)
        self.declare_parameter('goal_z', 0.0)
        self.declare_parameter('goal_tolerance', 0.1)  # meters
        self.declare_parameter('max_linear_velocity', 0.5)  # m/s
        self.declare_parameter('max_angular_velocity', 0.5)  # rad/s
        self.declare_parameter('localization_enabled', True)
        self.declare_parameter('min_cone_matches_for_localization', 3)
        self.declare_parameter('cone_match_threshold', 0.8)  # meters
        
        # Get parameters
        self.model_path = self.get_parameter('model_path').value
        self.cone_map_file = self.get_parameter('cone_map_file').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.image_topic = self.get_parameter('image_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.device = self.get_parameter('device').value
        self.cone_class_id = self.get_parameter('cone_class_id').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.robot_frame = self.get_parameter('robot_frame').value
        self.map_frame = self.get_parameter('map_frame').value
        self.use_depth_camera = self.get_parameter('use_depth_camera').value
        self.cone_height = self.get_parameter('cone_height').value
        self.max_detection_distance = self.get_parameter('max_detection_distance').value
        self.min_detection_distance = self.get_parameter('min_detection_distance').value
        
        # Navigation parameters
        self.goal_position = Point()
        self.goal_position.x = float(self.get_parameter('goal_x').value)
        self.goal_position.y = float(self.get_parameter('goal_y').value)
        self.goal_position.z = float(self.get_parameter('goal_z').value)
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.max_linear_velocity = self.get_parameter('max_linear_velocity').value
        self.max_angular_velocity = self.get_parameter('max_angular_velocity').value
        self.localization_enabled = self.get_parameter('localization_enabled').value
        self.min_cone_matches = self.get_parameter('min_cone_matches_for_localization').value
        self.cone_match_threshold = self.get_parameter('cone_match_threshold').value
        
        # State
        self.camera_matrix = None
        self.camera_info_received = False
        self.latest_depth_image = None
        self.depth_lock = threading.Lock()
        self.current_pose = None  # Robot's estimated pose in map frame
        self.pose_lock = threading.Lock()
        self.goal_reached = False
        
        # Load pre-mapped cone positions
        self.reference_cone_map = self._load_cone_map(self.cone_map_file)
        
        # Detected cones tracking
        self.detected_cones_history = []  # For visualization
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Load YOLOv11 model
        self.get_logger().info(f'Loading YOLOv11 model from: {self.model_path}')
        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        
        # TF2 Buffer and Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # QoS profiles
        qos_sensor = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        qos_reliable = QoSProfile(depth=10)
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, qos_sensor
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, qos_reliable
        )
        if self.use_depth_camera:
            self.depth_sub = self.create_subscription(
                Image, self.depth_topic, self.depth_callback, qos_sensor
            )
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', qos_reliable)
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/estimated_pose', qos_reliable
        )
        self.cone_markers_pub = self.create_publisher(
            MarkerArray, '/detected_cones/markers', qos_reliable
        )
        self.reference_map_pub = self.create_publisher(
            MarkerArray, '/reference_cone_map/markers', qos_reliable
        )
        self.goal_marker_pub = self.create_publisher(
            Marker, '/goal_marker', qos_reliable
        )
        self.debug_image_pub = self.create_publisher(
            Image, '/cone_nav/debug_image', QoSProfile(depth=1)
        )
        
        # Navigation timer
        self.nav_timer = self.create_timer(0.1, self.navigation_control_loop)
        
        # Visualization timer
        self.viz_timer = self.create_timer(0.5, self.publish_visualizations)
        
        # Statistics
        self.frame_count = 0
        self.localization_count = 0
        self.total_cone_detections = 0
        self.successful_matches = 0
        
        self.get_logger().info(
            f'\n{"="*60}\n'
            f'Autonomous Cone Navigation Node Initialized\n'
            f'{"="*60}\n'
            f'  Reference map: {len(self.reference_cone_map)} cones loaded\n'
            f'  Goal position: ({self.goal_position.x:.2f}, {self.goal_position.y:.2f}, {self.goal_position.z:.2f})\n'
            f'  Goal tolerance: {self.goal_tolerance}m\n'
            f'  Localization: {"ENABLED" if self.localization_enabled else "DISABLED"}\n'
            f'  Max velocities: linear={self.max_linear_velocity} m/s, angular={self.max_angular_velocity} rad/s\n'
            f'{"="*60}'
        )
    
    def _load_cone_map(self, filepath):
        """Load pre-mapped cone positions from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            cone_map = []
            for cone in data:
                cone_map.append({
                    'id': cone.get('id', len(cone_map)),
                    'position': Point(
                        x=float(cone['x']),
                        y=float(cone['y']),
                        z=float(cone.get('z', 0.0))
                    ),
                    'label': cone.get('label', f"Cone_{cone.get('id', len(cone_map))}")
                })
            
            self.get_logger().info(f'Loaded {len(cone_map)} cones from map file')
            for cone in cone_map:
                self.get_logger().info(
                    f"  {cone['label']}: ({cone['position'].x:.2f}, "
                    f"{cone['position'].y:.2f}, {cone['position'].z:.2f})"
                )
            
            return cone_map
            
        except FileNotFoundError:
            self.get_logger().error(f'Cone map file not found: {filepath}')
            self.get_logger().info('Creating empty reference map. Add cones manually.')
            return []
        except Exception as e:
            self.get_logger().error(f'Error loading cone map: {e}')
            return []
    
    def camera_info_callback(self, msg: CameraInfo):
        """Store camera intrinsics"""
        if not self.camera_info_received:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.camera_info_received = True
            self.get_logger().info(
                f'Camera intrinsics received: '
                f'fx={self.camera_matrix[0,0]:.2f}, fy={self.camera_matrix[1,1]:.2f}'
            )
    
    def depth_callback(self, msg: Image):
        """Cache latest depth image"""
        try:
            with self.depth_lock:
                self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, '32FC1')
        except Exception as e:
            self.get_logger().error(f'Depth image error: {e}')
    
    def image_callback(self, msg: Image):
        """Process camera images, detect cones, and update localization"""
        if not self.camera_info_received:
            return
        
        self.frame_count += 1
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # Detect cones
            results = self.model.predict(
                cv_image,
                conf=self.confidence_threshold,
                verbose=False,
                device=self.device,
                classes=[self.cone_class_id]
            )
            
            # Extract 3D positions
            detected_cones = self._extract_cone_positions(results[0], cv_image.shape, msg.header.stamp)
            
            if detected_cones:
                self.total_cone_detections += len(detected_cones)
                
                # Perform localization if enabled
                if self.localization_enabled:
                    self._localize_with_cones(detected_cones, msg.header.stamp)
                
                self.get_logger().info(
                    f'Frame {self.frame_count}: Detected {len(detected_cones)} cones',
                    throttle_duration_sec=1.0
                )
            
            # Publish debug image
            debug_img = self._create_debug_image(cv_image, results[0], detected_cones)
            self.debug_image_pub.publish(
                self.bridge.cv2_to_imgmsg(debug_img, 'bgr8')
            )
            
        except Exception as e:
            self.get_logger().error(f'Image callback error: {e}')
    
    def _extract_cone_positions(self, result, image_shape, timestamp):
        """Extract 3D positions of detected cones in camera frame"""
        if result.boxes is None or len(result.boxes) == 0:
            return []
        
        boxes = result.boxes
        h, w = image_shape[:2]
        cones = []
        
        for i in range(len(boxes)):
            confidence = float(boxes.conf[i].item())
            bbox = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = bbox
            
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            # Get depth
            if self.use_depth_camera and self.latest_depth_image is not None:
                depth = self._get_depth_at_pixel(int(cx), int(cy))
            else:
                bbox_height = y2 - y1
                depth = self._estimate_depth_from_bbox(bbox_height, h)
            
            if depth is None or depth < self.min_detection_distance or depth > self.max_detection_distance:
                continue
            
            # Convert to 3D point in camera frame
            point_3d = self._pixel_to_3d_point(cx, cy, depth)
            
            if point_3d is not None:
                cones.append({
                    'position_camera': point_3d,
                    'confidence': confidence,
                    'bbox': bbox,
                    'timestamp': timestamp
                })
        
        return cones
    
    def _get_depth_at_pixel(self, x, y):
        """Get depth value at pixel"""
        with self.depth_lock:
            if self.latest_depth_image is None:
                return None
            
            h, w = self.latest_depth_image.shape
            if x < 0 or x >= w or y < 0 or y >= h:
                return None
            
            # Sample 5x5 region
            x_min, x_max = max(0, x-2), min(w, x+3)
            y_min, y_max = max(0, y-2), min(h, y+3)
            region = self.latest_depth_image[y_min:y_max, x_min:x_max]
            valid = region[np.isfinite(region) & (region > 0)]
            
            return float(np.median(valid)) if len(valid) > 0 else None
    
    def _estimate_depth_from_bbox(self, bbox_height_px, image_height_px):
        """Estimate depth from bounding box height"""
        if bbox_height_px < 5:
            return None
        fy = self.camera_matrix[1, 1]
        return (self.cone_height * fy) / bbox_height_px
    
    def _pixel_to_3d_point(self, px, py, depth):
        """Convert pixel + depth to 3D point"""
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        x = (px - cx) * depth / fx
        y = (py - cy) * depth / fy
        z = depth
        
        return Point(x=float(x), y=float(y), z=float(z))
    
    def _localize_with_cones(self, detected_cones, timestamp):
        """Update robot pose using cone observations"""
        try:
            # Transform detected cones to map frame using current odometry
            cones_map_odom = []
            for cone in detected_cones:
                pose_camera = PoseStamped()
                pose_camera.header.frame_id = self.camera_frame
                pose_camera.header.stamp = timestamp
                pose_camera.pose.position = cone['position_camera']
                pose_camera.pose.orientation.w = 1.0
                
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame, self.camera_frame, timestamp,
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
                
                pose_map = tf2_geometry_msgs.do_transform_pose(pose_camera, transform)
                cone['position_map_odom'] = pose_map.pose.position
                cones_map_odom.append(cone)
            
            # Match to reference map
            matches = self._match_cones_to_reference(cones_map_odom)
            
            if len(matches) >= self.min_cone_matches:
                # Compute pose correction
                corrected_pose = self._compute_pose_correction(matches, timestamp)
                
                if corrected_pose is not None:
                    with self.pose_lock:
                        self.current_pose = corrected_pose
                    
                    self.localization_count += 1
                    self.successful_matches += len(matches)
                    
                    # Publish corrected pose
                    self._publish_corrected_pose(corrected_pose, timestamp)
                    
                    self.get_logger().info(
                        f'Localization #{self.localization_count}: '
                        f'{len(matches)} cone matches, '
                        f'pose=({corrected_pose.position.x:.2f}, {corrected_pose.position.y:.2f})',
                        throttle_duration_sec=0.5
                    )
                    
                    # Store matched cones for visualization
                    for match in matches:
                        match['detected_cone']['matched_reference'] = match['reference_cone']
            
        except Exception as e:
            self.get_logger().warn(f'Localization failed: {e}', throttle_duration_sec=2.0)
    
    def _match_cones_to_reference(self, detected_cones):
        """Match detected cones to reference map"""
        matches = []
        
        for detected in detected_cones:
            det_pos = detected['position_map_odom']
            
            # Find closest cone in reference map
            min_dist = float('inf')
            best_match = None
            
            for ref_cone in self.reference_cone_map:
                ref_pos = ref_cone['position']
                dist = np.sqrt(
                    (det_pos.x - ref_pos.x)**2 +
                    (det_pos.y - ref_pos.y)**2
                )
                
                if dist < min_dist and dist < self.cone_match_threshold:
                    min_dist = dist
                    best_match = ref_cone
            
            if best_match is not None:
                matches.append({
                    'detected_cone': detected,
                    'reference_cone': best_match,
                    'distance': min_dist
                })
                
                self.get_logger().debug(
                    f"Matched cone to {best_match['label']} (dist={min_dist:.2f}m)"
                )
        
        return matches
    
    def _compute_pose_correction(self, matches, timestamp):
        """Compute corrected robot pose from cone matches"""
        # Simple approach: compute mean translation error
        detected_points = []
        reference_points = []
        
        for match in matches:
            det_pos = match['detected_cone']['position_map_odom']
            ref_pos = match['reference_cone']['position']
            
            detected_points.append([det_pos.x, det_pos.y])
            reference_points.append([ref_pos.x, ref_pos.y])
        
        detected_points = np.array(detected_points)
        reference_points = np.array(reference_points)
        
        # Compute translation correction
        error = reference_points - detected_points
        mean_correction = np.mean(error, axis=0)
        
        # Get current odometry pose
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.robot_frame, timestamp,
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            # Apply correction
            corrected_pose = PoseStamped()
            corrected_pose.header.frame_id = self.map_frame
            corrected_pose.header.stamp = timestamp
            corrected_pose.pose.position.x = transform.transform.translation.x + mean_correction[0]
            corrected_pose.pose.position.y = transform.transform.translation.y + mean_correction[1]
            corrected_pose.pose.position.z = transform.transform.translation.z
            corrected_pose.pose.orientation = transform.transform.rotation
            
            return corrected_pose.pose
            
        except Exception as e:
            self.get_logger().warn(f'Failed to get odometry transform: {e}')
            return None
    
    def _publish_corrected_pose(self, pose, timestamp):
        """Publish corrected pose estimate"""
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.frame_id = self.map_frame
        pose_msg.header.stamp = timestamp
        pose_msg.pose.pose = pose
        
        # Set covariance (simplified)
        pose_msg.pose.covariance = [0.1] * 36
        pose_msg.pose.covariance[0] = 0.05   # x
        pose_msg.pose.covariance[7] = 0.05   # y
        pose_msg.pose.covariance[35] = 0.1   # yaw
        
        self.pose_pub.publish(pose_msg)
    
    def navigation_control_loop(self):
        """Main navigation control loop"""
        if self.goal_reached:
            return
        
        with self.pose_lock:
            current_pose = self.current_pose
        
        if current_pose is None:
            # No localization yet, use odometry
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame, self.robot_frame, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
                current_pose = PoseStamped().pose
                current_pose.position.x = transform.transform.translation.x
                current_pose.position.y = transform.transform.translation.y
                current_pose.orientation = transform.transform.rotation
            except:
                return
        
        # Compute distance and angle to goal
        dx = self.goal_position.x - current_pose.position.x
        dy = self.goal_position.y - current_pose.position.y
        distance = np.sqrt(dx**2 + dy**2)
        angle_to_goal = np.arctan2(dy, dx)
        
        # Get current robot orientation
        quat = [
            current_pose.orientation.x,
            current_pose.orientation.y,
            current_pose.orientation.z,
            current_pose.orientation.w
        ]
        _, _, current_yaw = euler_from_quaternion(quat)
        
        # Compute heading error
        heading_error = angle_to_goal - current_yaw
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))  # Normalize to [-pi, pi]
        
        # Check if goal reached
        if distance < self.goal_tolerance:
            self._stop_robot()
            self.goal_reached = True
            self.get_logger().info(
                f'\n{"="*60}\n'
                f'GOAL REACHED!\n'
                f'{"="*60}\n'
                f'  Final position: ({current_pose.position.x:.3f}, {current_pose.position.y:.3f})\n'
                f'  Target position: ({self.goal_position.x:.3f}, {self.goal_position.y:.3f})\n'
                f'  Final distance: {distance:.3f}m\n'
                f'  Total localizations: {self.localization_count}\n'
                f'  Total cone detections: {self.total_cone_detections}\n'
                f'  Successful matches: {self.successful_matches}\n'
                f'{"="*60}'
            )
            return
        
        # Compute control commands
        cmd = Twist()
        
        # Pure pursuit-like control
        if abs(heading_error) > 0.2:  # ~11 degrees
            # Rotate towards goal
            cmd.linear.x = 0.1 * self.max_linear_velocity
            cmd.angular.z = np.clip(
                2.0 * heading_error,
                -self.max_angular_velocity,
                self.max_angular_velocity
            )
        else:
            # Move towards goal
            cmd.linear.x = np.clip(
                0.5 * distance,
                0.0,
                self.max_linear_velocity
            )
            cmd.angular.z = np.clip(
                heading_error,
                -self.max_angular_velocity / 2,
                self.max_angular_velocity / 2
            )
        
        self.cmd_vel_pub.publish(cmd)
        
        self.get_logger().info(
            f'Nav: dist={distance:.2f}m, heading_err={np.degrees(heading_error):.1f}°, '
            f'vel=({cmd.linear.x:.2f}, {cmd.angular.z:.2f})',
            throttle_duration_sec=0.5
        )
    
    def _stop_robot(self):
        """Stop the robot"""
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)
    
    def publish_visualizations(self):
        """Publish visualization markers"""
        # Reference cone map
        ref_markers = MarkerArray()
        for cone in self.reference_cone_map:
            marker = Marker()
            marker.header.frame_id = self.map_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "reference_cones"
            marker.id = cone['id']
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position = cone['position']
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = 0.2
            marker.scale.z = self.cone_height
            marker.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.8)
            marker.lifetime = rclpy.duration.Duration(seconds=1.0).to_msg()
            ref_markers.markers.append(marker)
            
            # Label
            text_marker = Marker()
            text_marker.header = marker.header
            text_marker.ns = "reference_labels"
            text_marker.id = cone['id'] + 1000
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = cone['position'].x
            text_marker.pose.position.y = cone['position'].y
            text_marker.pose.position.z = cone['position'].z + 0.4
            text_marker.text = cone['label']
            text_marker.scale.z = 0.15
            text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            text_marker.lifetime = marker.lifetime
            ref_markers.markers.append(text_marker)
        
        self.reference_map_pub.publish(ref_markers)
        
        # Goal marker
        goal_marker = Marker()
        goal_marker.header.frame_id = self.map_frame
        goal_marker.header.stamp = self.get_clock().now().to_msg()
        goal_marker.ns = "goal"
        goal_marker.id = 0
        goal_marker.type = Marker.SPHERE
        goal_marker.action = Marker.ADD
        goal_marker.pose.position = self.goal_position
        goal_marker.pose.orientation.w = 1.0
        goal_marker.scale.x = goal_marker.scale.y = goal_marker.scale.z = 0.3
        goal_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)
        self.goal_marker_pub.publish(goal_marker)
    
    def _create_debug_image(self, image, result, detected_cones):
        """Create debug visualization"""
        debug_img = image.copy()
        h, w = debug_img.shape[:2]
        
        # Draw detections
        if result.boxes is not None:
            boxes = result.boxes
            for i in range(len(boxes)):
                conf = float(boxes.conf[i].item())
                bbox = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, bbox)
                
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(debug_img, f'{conf:.2f}', (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        
        # Draw info
        with self.pose_lock:
            pose = self.current_pose
        
        if pose:
            dx = self.goal_position.x - pose.position.x
            dy = self.goal_position.y - pose.position.y
            dist = np.sqrt(dx**2 + dy**2)
            
            info_y = 30
            cv2.putText(debug_img, f'Position: ({pose.position.x:.2f}, {pose.position.y:.2f})',
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            info_y += 25
            cv2.putText(debug_img, f'Goal: ({self.goal_position.x:.2f}, {self.goal_position.y:.2f})',
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            info_y += 25
            cv2.putText(debug_img, f'Distance: {dist:.2f}m',
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            info_y += 25
            cv2.putText(debug_img, f'Detected: {len(detected_cones)} cones',
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            info_y += 25
            cv2.putText(debug_img, f'Localizations: {self.localization_count}',
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return debug_img


def main(args=None):
    rclpy.init(args=args)
    node = AutonomousConeNavigationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node._stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()