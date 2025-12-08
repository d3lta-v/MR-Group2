#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, PoseStamped, PoseArray
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from collections import deque
import threading

from ultralytics import YOLO
import tf2_ros
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs


class ConeLocalizationNode(Node):
    def __init__(self):
        super().__init__('cone_localization_node')
        
        # Declare parameters
        self.declare_parameter('model_path', 'path/to/yolov11n.pt')
        self.declare_parameter('confidence_threshold', 0.6)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('image_topic', '/zed2/zed_node/left/image_rect_color')
        self.declare_parameter('camera_info_topic', '/zed2/zed_node/left/camera_info')
        self.declare_parameter('depth_topic', '/zed2/zed_node/depth/depth_registered')
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('half_precision', False)
        self.declare_parameter('cone_class_id', 0)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('camera_frame', 'zed2_left_camera_optical_frame')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('max_detection_distance', 20.0)  # meters
        self.declare_parameter('min_detection_distance', 0.5)   # meters
        self.declare_parameter('cone_merge_distance', 0.5)      # meters - merge cones closer than this
        self.declare_parameter('cone_height', 0.3)              # meters - assumed cone height
        self.declare_parameter('use_depth_camera', True)        # Use depth camera or estimate from image
        self.declare_parameter('publish_rate', 10.0)            # Hz for map publishing
        
        # Get parameters
        self.model_path = self.get_parameter('model_path').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.iou_threshold = self.get_parameter('iou_threshold').value
        self.image_topic = self.get_parameter('image_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.imgsz = self.get_parameter('imgsz').value
        self.device = self.get_parameter('device').value
        self.half_precision = self.get_parameter('half_precision').value
        self.cone_class_id = self.get_parameter('cone_class_id').value
        self.map_frame = self.get_parameter('map_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.robot_frame = self.get_parameter('robot_frame').value
        self.max_detection_distance = self.get_parameter('max_detection_distance').value
        self.min_detection_distance = self.get_parameter('min_detection_distance').value
        self.cone_merge_distance = self.get_parameter('cone_merge_distance').value
        self.cone_height = self.get_parameter('cone_height').value
        self.use_depth_camera = self.get_parameter('use_depth_camera').value
        self.publish_rate = self.get_parameter('publish_rate').value
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Camera intrinsics
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_info_received = False
        
        # Cone map (thread-safe)
        self.cone_map = []  # List of cone positions in map frame
        self.cone_observations = {}  # Track observations per cone
        self.map_lock = threading.Lock()
        
        # TF2 Buffer and Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Depth image cache
        self.latest_depth_image = None
        self.depth_lock = threading.Lock()
        
        # Load YOLOv11 model
        self.get_logger().info(f'Loading YOLOv11 model from: {self.model_path}')
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            if self.half_precision and self.device == 'cuda':
                self.model.half()
            self.get_logger().info(f'YOLOv11 model loaded on {self.device}')
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            raise
        
        # QoS profiles
        qos_sensor = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        qos_reliable = QoSProfile(depth=10)
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_sensor
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            qos_reliable
        )
        
        if self.use_depth_camera:
            self.depth_sub = self.create_subscription(
                Image,
                self.depth_topic,
                self.depth_callback,
                qos_sensor
            )
        
        # Publishers
        self.cone_markers_pub = self.create_publisher(
            MarkerArray,
            '/cone_map/markers',
            qos_reliable
        )
        
        self.cone_poses_pub = self.create_publisher(
            PoseArray,
            '/cone_map/poses',
            qos_reliable
        )
        
        self.debug_image_pub = self.create_publisher(
            Image,
            '/cone_detection/debug_image',
            QoSProfile(depth=1)
        )
        
        # Timer for publishing cone map
        self.map_publish_timer = self.create_timer(
            1.0 / self.publish_rate,
            self.publish_cone_map
        )
        
        # Statistics
        self.frame_count = 0
        self.total_detections = 0
        self.total_cones_mapped = 0
        self.failed_transforms = 0
        self.total_inference_time = 0.0
        
        self.get_logger().info(
            f'\n{"="*60}\n'
            f'Cone Localization Node Initialized\n'
            f'{"="*60}\n'
            f'  Model: {self.model_path}\n'
            f'  Device: {self.device}\n'
            f'  Confidence threshold: {self.confidence_threshold}\n'
            f'  Map frame: {self.map_frame}\n'
            f'  Camera frame: {self.camera_frame}\n'
            f'  Detection range: {self.min_detection_distance}-{self.max_detection_distance}m\n'
            f'  Cone merge distance: {self.cone_merge_distance}m\n'
            f'  Use depth camera: {self.use_depth_camera}\n'
            f'{"="*60}'
        )
    
    def camera_info_callback(self, msg: CameraInfo):
        """Store camera intrinsics"""
        if not self.camera_info_received:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            self.camera_info_received = True
            self.get_logger().info(
                f'Camera intrinsics received:\n'
                f'  fx: {self.camera_matrix[0,0]:.2f}\n'
                f'  fy: {self.camera_matrix[1,1]:.2f}\n'
                f'  cx: {self.camera_matrix[0,2]:.2f}\n'
                f'  cy: {self.camera_matrix[1,2]:.2f}'
            )
    
    def depth_callback(self, msg: Image):
        """Cache latest depth image"""
        try:
            with self.depth_lock:
                self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().error(f'Failed to convert depth image: {e}')
    
    def image_callback(self, msg: Image):
        """Process camera images and detect cones"""
        if not self.camera_info_received:
            self.get_logger().warn('Waiting for camera info...', throttle_duration_sec=2.0)
            return
        
        self.frame_count += 1
        callback_start = time.time()
        
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Run YOLO detection
            inference_start = time.time()
            results = self.model.predict(
                cv_image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                verbose=False,
                device=self.device,
                half=self.half_precision,
                classes=[self.cone_class_id]  # Only detect cones
            )
            inference_time = time.time() - inference_start
            self.total_inference_time += inference_time
            
            # Extract cone detections
            cones_camera_frame = self._extract_cone_positions(
                results[0], 
                cv_image.shape,
                msg.header.stamp
            )
            
            if cones_camera_frame:
                self.total_detections += len(cones_camera_frame)
                
                # Transform cones to map frame and update map
                cones_added = self._update_cone_map(cones_camera_frame, msg.header.stamp)
                
                self.get_logger().info(
                    f'Frame {self.frame_count} | Detected {len(cones_camera_frame)} cones | '
                    f'Added {cones_added} to map | Total map size: {len(self.cone_map)} | '
                    f'Inference: {inference_time*1000:.1f}ms',
                    throttle_duration_sec=0.5
                )
            
            # Create debug visualization
            debug_image = self._create_debug_image(cv_image, results[0], cones_camera_frame)
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
            self.debug_image_pub.publish(debug_msg)
            
            # Log statistics periodically
            if self.frame_count % 30 == 0:
                self._log_statistics()
            
        except Exception as e:
            self.get_logger().error(f'Error in image callback: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def _extract_cone_positions(self, result, image_shape, timestamp):
        """
        Extract 3D positions of detected cones in camera frame
        
        Returns:
            List of dicts with keys: 'position' (Point), 'confidence', 'bbox'
        """
        if result.boxes is None or len(result.boxes) == 0:
            return []
        
        boxes = result.boxes
        h, w = image_shape[:2]
        
        cones = []
        
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            if cls_id != self.cone_class_id:
                continue
            
            confidence = float(boxes.conf[i].item())
            bbox = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = bbox
            
            # Calculate cone center in image
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            # Get depth
            if self.use_depth_camera and self.latest_depth_image is not None:
                depth = self._get_depth_at_pixel(int(cx), int(cy))
            else:
                # Estimate depth from bounding box height
                bbox_height = y2 - y1
                depth = self._estimate_depth_from_bbox(bbox_height, h)
            
            if depth is None or depth < self.min_detection_distance or depth > self.max_detection_distance:
                self.get_logger().debug(
                    f'Cone rejected: depth={depth}, limits=[{self.min_detection_distance}, {self.max_detection_distance}]'
                )
                continue
            
            # Convert pixel coordinates to 3D point in camera frame
            point_3d = self._pixel_to_3d_point(cx, cy, depth)
            
            if point_3d is not None:
                cones.append({
                    'position': point_3d,
                    'confidence': confidence,
                    'bbox': bbox,
                    'timestamp': timestamp
                })
                
                self.get_logger().debug(
                    f'Cone detected at image ({cx:.0f}, {cy:.0f}), '
                    f'depth={depth:.2f}m, camera_frame=({point_3d.x:.2f}, {point_3d.y:.2f}, {point_3d.z:.2f})'
                )
        
        return cones
    
    def _get_depth_at_pixel(self, x, y):
        """Get depth value at pixel coordinates from depth image"""
        with self.depth_lock:
            if self.latest_depth_image is None:
                return None
            
            h, w = self.latest_depth_image.shape
            if x < 0 or x >= w or y < 0 or y >= h:
                return None
            
            # Sample 5x5 region around pixel and take median (more robust)
            x_min = max(0, x - 2)
            x_max = min(w, x + 3)
            y_min = max(0, y - 2)
            y_max = min(h, y + 3)
            
            region = self.latest_depth_image[y_min:y_max, x_min:x_max]
            valid_depths = region[np.isfinite(region) & (region > 0)]
            
            if len(valid_depths) == 0:
                return None
            
            return float(np.median(valid_depths))
    
    def _estimate_depth_from_bbox(self, bbox_height_px, image_height_px):
        """
        Estimate depth from bounding box height (pinhole camera model)
        
        Args:
            bbox_height_px: Height of bounding box in pixels
            image_height_px: Total image height in pixels
        """
        if bbox_height_px < 5:  # Too small to be reliable
            return None
        
        # Pinhole camera model: h_image = (h_real * f) / depth
        # depth = (h_real * f) / h_image
        fy = self.camera_matrix[1, 1]
        depth = (self.cone_height * fy) / bbox_height_px
        
        return depth
    
    def _pixel_to_3d_point(self, px, py, depth):
        """
        Convert pixel coordinates and depth to 3D point in camera frame
        
        Args:
            px, py: Pixel coordinates
            depth: Depth in meters
        
        Returns:
            Point in camera frame
        """
        if self.camera_matrix is None:
            return None
        
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        # Camera coordinates
        x = (px - cx) * depth / fx
        y = (py - cy) * depth / fy
        z = depth
        
        point = Point()
        point.x = float(x)
        point.y = float(y)
        point.z = float(z)
        
        return point
    
    def _update_cone_map(self, cones_camera_frame, timestamp):
        """
        Transform cones to map frame and update global cone map
        
        Returns:
            Number of new cones added to map
        """
        cones_added = 0
        
        for cone_data in cones_camera_frame:
            try:
                # Create PoseStamped in camera frame
                pose_camera = PoseStamped()
                pose_camera.header.frame_id = self.camera_frame
                pose_camera.header.stamp = timestamp
                pose_camera.pose.position = cone_data['position']
                pose_camera.pose.orientation.w = 1.0
                
                # Transform to map frame
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    self.camera_frame,
                    timestamp,
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
                
                pose_map = tf2_geometry_msgs.do_transform_pose(pose_camera, transform)
                
                # Add to map (with duplicate checking)
                if self._add_cone_to_map(pose_map.pose.position, cone_data['confidence']):
                    cones_added += 1
                    self.get_logger().debug(
                        f'Added cone to map at ({pose_map.pose.position.x:.2f}, '
                        f'{pose_map.pose.position.y:.2f}, {pose_map.pose.position.z:.2f})'
                    )
                
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                    tf2_ros.ExtrapolationException) as e:
                self.failed_transforms += 1
                self.get_logger().warn(
                    f'Transform failed: {e}',
                    throttle_duration_sec=2.0
                )
                continue
        
        return cones_added
    
    def _add_cone_to_map(self, position, confidence):
        """
        Add cone to map if not already present (checks for duplicates)
        
        Returns:
            True if cone was added, False if duplicate
        """
        with self.map_lock:
            # Check for nearby cones (merge if too close)
            for i, existing_cone in enumerate(self.cone_map):
                dist = np.sqrt(
                    (position.x - existing_cone['position'].x)**2 +
                    (position.y - existing_cone['position'].y)**2 +
                    (position.z - existing_cone['position'].z)**2
                )
                
                if dist < self.cone_merge_distance:
                    # Update existing cone position (weighted average)
                    obs_count = self.cone_observations.get(i, 1)
                    weight_new = 1.0 / (obs_count + 1)
                    weight_old = obs_count / (obs_count + 1)
                    
                    existing_cone['position'].x = (
                        weight_old * existing_cone['position'].x + 
                        weight_new * position.x
                    )
                    existing_cone['position'].y = (
                        weight_old * existing_cone['position'].y + 
                        weight_new * position.y
                    )
                    existing_cone['position'].z = (
                        weight_old * existing_cone['position'].z + 
                        weight_new * position.z
                    )
                    existing_cone['confidence'] = max(existing_cone['confidence'], confidence)
                    
                    self.cone_observations[i] = obs_count + 1
                    
                    self.get_logger().debug(
                        f'Updated existing cone {i} (observations: {self.cone_observations[i]})'
                    )
                    return False
            
            # Add new cone
            cone_id = len(self.cone_map)
            self.cone_map.append({
                'position': Point(x=position.x, y=position.y, z=position.z),
                'confidence': confidence,
                'id': cone_id
            })
            self.cone_observations[cone_id] = 1
            self.total_cones_mapped += 1
            
            return True
    
    def publish_cone_map(self):
        """Publish cone map as MarkerArray and PoseArray"""
        with self.map_lock:
            if len(self.cone_map) == 0:
                return
            
            # Create MarkerArray
            marker_array = MarkerArray()
            
            # Delete old markers
            delete_marker = Marker()
            delete_marker.action = Marker.DELETEALL
            marker_array.markers.append(delete_marker)
            
            # Create PoseArray
            pose_array = PoseArray()
            pose_array.header.frame_id = self.map_frame
            pose_array.header.stamp = self.get_clock().now().to_msg()
            
            for cone in self.cone_map:
                # Marker for visualization
                marker = Marker()
                marker.header.frame_id = self.map_frame
                marker.header.stamp = pose_array.header.stamp
                marker.ns = "cones"
                marker.id = cone['id']
                marker.type = Marker.CYLINDER
                marker.action = Marker.ADD
                
                marker.pose.position = cone['position']
                marker.pose.orientation.w = 1.0
                
                # Orange cone dimensions
                marker.scale.x = 0.2  # diameter
                marker.scale.y = 0.2
                marker.scale.z = self.cone_height
                
                # Orange color
                marker.color.r = 1.0
                marker.color.g = 0.5
                marker.color.b = 0.0
                marker.color.a = 0.9
                
                marker.lifetime = rclpy.duration.Duration(seconds=2.0).to_msg()
                
                marker_array.markers.append(marker)
                
                # Add text label
                text_marker = Marker()
                text_marker.header = marker.header
                text_marker.ns = "cone_labels"
                text_marker.id = cone['id'] + 10000
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                
                text_marker.pose.position.x = cone['position'].x
                text_marker.pose.position.y = cone['position'].y
                text_marker.pose.position.z = cone['position'].z + 0.3
                
                text_marker.text = f"C{cone['id']}\n{self.cone_observations.get(cone['id'], 1)} obs"
                text_marker.scale.z = 0.15
                
                text_marker.color.r = 1.0
                text_marker.color.g = 1.0
                text_marker.color.b = 1.0
                text_marker.color.a = 1.0
                
                text_marker.lifetime = marker.lifetime
                
                marker_array.markers.append(text_marker)
                
                # Add to PoseArray
                from geometry_msgs.msg import Pose
                pose = Pose()
                pose.position = cone['position']
                pose.orientation.w = 1.0
                pose_array.poses.append(pose)
            
            # Publish
            self.cone_markers_pub.publish(marker_array)
            self.cone_poses_pub.publish(pose_array)
    
    def _create_debug_image(self, image, result, cones_camera_frame):
        """Create debug visualization with detected cones"""
        debug_img = image.copy()
        
        # Draw YOLO detections
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                if cls_id != self.cone_class_id:
                    continue
                
                confidence = float(boxes.conf[i].item())
                bbox = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, bbox)
                
                # Draw bounding box (orange for cones)
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 165, 255), 2)
                
                # Draw label
                label = f'Cone {confidence:.2f}'
                cv2.putText(debug_img, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                
                # Draw center point
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cv2.circle(debug_img, (cx, cy), 5, (0, 255, 0), -1)
        
        # Draw 3D positions (if available)
        for i, cone_data in enumerate(cones_camera_frame):
            pos = cone_data['position']
            # Project back to image for visualization
            if self.camera_matrix is not None:
                fx = self.camera_matrix[0, 0]
                fy = self.camera_matrix[1, 1]
                cx = self.camera_matrix[0, 2]
                cy = self.camera_matrix[1, 2]
                
                px = int(fx * pos.x / pos.z + cx)
                py = int(fy * pos.y / pos.z + cy)
                
                # Draw depth info
                depth_text = f'{pos.z:.2f}m'
                cv2.putText(debug_img, depth_text, (px + 10, py),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Draw info overlay
        info_y = 30
        cv2.putText(debug_img, f'Frame: {self.frame_count}', (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 25
        cv2.putText(debug_img, f'Detected: {len(cones_camera_frame)} cones', (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 25
        cv2.putText(debug_img, f'Map size: {len(self.cone_map)} cones', (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return debug_img
    
    def _log_statistics(self):
        """Log performance statistics"""
        avg_inference_time = self.total_inference_time / max(self.frame_count, 1)
        avg_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0.0
        
        with self.map_lock:
            map_size = len(self.cone_map)
        
        self.get_logger().info(
            f'\n{"="*60}\n'
            f'Cone Localization Statistics:\n'
            f'{"="*60}\n'
            f'  Frames processed: {self.frame_count}\n'
            f'  Total cone detections: {self.total_detections}\n'
            f'  Unique cones mapped: {map_size}\n'
            f'  Failed transforms: {self.failed_transforms}\n'
            f'  Average inference time: {avg_inference_time*1000:.1f}ms\n'
            f'  Average FPS: {avg_fps:.1f}\n'
            f'  Detections per frame: {self.total_detections/max(self.frame_count,1):.2f}\n'
            f'{"="*60}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = ConeLocalizationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Cone Localization Node')
    finally:
        # Log final statistics
        with node.map_lock:
            map_size = len(node.cone_map)
            total_observations = sum(node.cone_observations.values())
        
        node.get_logger().info(
            f'\n{"="*60}\n'
            f'Final Cone Map Statistics:\n'
            f'{"="*60}\n'
            f'  Total frames: {node.frame_count}\n'
            f'  Unique cones mapped: {map_size}\n'
            f'  Total observations: {total_observations}\n'
            f'  Average observations per cone: {total_observations/max(map_size,1):.1f}\n'
            f'{"="*60}'
        )
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()