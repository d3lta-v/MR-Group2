#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from pathlib import Path

from ultralytics import YOLO


class YOLOv11LaneSegmentationNode(Node):
    def __init__(self):
        super().__init__('yolov11_lane_segmentation_node')
        
        # Declare parameters
        self.declare_parameter('model_path', 'path/to/yolov11n-seg.pt')  # Tiny model
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('image_topic', '/zed2/zed_node/left/image_rect_color')
        self.declare_parameter('imgsz', 640)  # Input image size
        self.declare_parameter('device', 'cuda')  # 'cuda' or 'cpu'
        self.declare_parameter('half_precision', False)  # FP16 for faster inference
        self.declare_parameter('left_lane_class_id', 0)  # Class ID for left lane
        self.declare_parameter('right_lane_class_id', 1)  # Class ID for right lane
        
        # Get parameters
        self.model_path = self.get_parameter('model_path').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.iou_threshold = self.get_parameter('iou_threshold').value
        self.image_topic = self.get_parameter('image_topic').value
        self.imgsz = self.get_parameter('imgsz').value
        self.device = self.get_parameter('device').value
        self.half_precision = self.get_parameter('half_precision').value
        self.left_lane_class_id = self.get_parameter('left_lane_class_id').value
        self.right_lane_class_id = self.get_parameter('right_lane_class_id').value
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Initialize YOLOv11 model
        self.get_logger().info(f'Loading YOLOv11 model from: {self.model_path}')
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            if self.half_precision and self.device == 'cuda':
                self.model.half()
                self.get_logger().info('Using FP16 half precision')
            self.get_logger().info(f'YOLOv11 model loaded successfully on {self.device}')
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            raise
        
        # QoS profile for image subscription (best effort for camera topics)
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribe to camera image
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile
        )
        
        # Publisher for lane boundary x-coordinates
        # Message format: [left_x, right_x, image_width]
        self.lane_pub = self.create_publisher(
            Float32MultiArray,
            '/lane_boundaries',
            QoSProfile(depth=10)
        )
        
        # Publisher for debug/visualization image
        self.debug_pub = self.create_publisher(
            Image,
            '/lane_segmentation/debug_image',
            QoSProfile(depth=1)
        )
        
        # Performance tracking
        self.frame_count = 0
        self.total_inference_time = 0.0
        self.successful_detections = 0
        self.failed_detections = 0
        
        # Log performance every N frames
        self.log_interval = 30
        
        self.get_logger().info(
            f'YOLOv11 Lane Segmentation Node initialized:\n'
            f'  Model: {Path(self.model_path).name}\n'
            f'  Device: {self.device}\n'
            f'  Input size: {self.imgsz}\n'
            f'  Confidence threshold: {self.confidence_threshold}\n'
            f'  IoU threshold: {self.iou_threshold}\n'
            f'  Subscribing to: {self.image_topic}\n'
            f'  Half precision: {self.half_precision}'
        )
    
    def image_callback(self, msg: Image):
        """Process incoming camera images"""
        self.frame_count += 1
        callback_start = time.time()
        
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            original_height, original_width = cv_image.shape[:2]
            
            self.get_logger().debug(f'Received image: {original_width}x{original_height}')
            
            # Run YOLOv11 inference
            inference_start = time.time()
            results = self.model.predict(
                cv_image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                verbose=False,
                device=self.device,
                half=self.half_precision
            )
            inference_time = time.time() - inference_start
            self.total_inference_time += inference_time
            
            # Extract lane boundaries from results
            left_x, right_x = self._extract_lane_boundaries(results[0], original_width, original_height)
            
            # Publish lane boundaries if valid
            if left_x is not None and right_x is not None:
                lane_msg = Float32MultiArray()
                lane_msg.data = [float(left_x), float(right_x), float(original_width)]
                self.lane_pub.publish(lane_msg)
                self.successful_detections += 1
                
                lane_width = right_x - left_x
                lane_center = (left_x + right_x) / 2.0
                image_center = original_width / 2.0
                lateral_error = lane_center - image_center
                
                self.get_logger().info(
                    f'Frame {self.frame_count} | '
                    f'Left: {left_x:.1f}px, Right: {right_x:.1f}px, '
                    f'Width: {lane_width:.1f}px, Error: {lateral_error:+.1f}px | '
                    f'Inference: {inference_time*1000:.1f}ms ({1.0/inference_time:.1f} FPS)',
                    throttle_duration_sec=0.5
                )
            else:
                self.failed_detections += 1
                self.get_logger().warn(
                    f'Frame {self.frame_count} | No valid lane boundaries detected | '
                    f'Inference: {inference_time*1000:.1f}ms',
                    throttle_duration_sec=1.0
                )
            
            # Create and publish debug visualization
            debug_image = self._create_debug_image(cv_image, results[0], left_x, right_x)
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
            self.debug_pub.publish(debug_msg)
            
            # Log performance statistics periodically
            if self.frame_count % self.log_interval == 0:
                self._log_performance_stats()
            
            callback_time = time.time() - callback_start
            self.get_logger().debug(f'Total callback time: {callback_time*1000:.1f}ms')
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def _extract_lane_boundaries(self, result, image_width, image_height):
        """
        Extract left and right lane boundary x-coordinates from YOLO segmentation results
        
        Args:
            result: YOLOv11 result object
            image_width: Original image width
            image_height: Original image height
            
        Returns:
            left_x, right_x: X-coordinates of lane boundaries (None if not detected)
        """
        if result.masks is None or len(result.masks) == 0:
            self.get_logger().debug('No segmentation masks detected')
            return None, None
        
        boxes = result.boxes
        masks = result.masks
        
        left_x = None
        right_x = None
        left_confidence = 0.0
        right_confidence = 0.0
        
        # Process each detection
        num_detections = len(boxes)
        self.get_logger().debug(f'Processing {num_detections} detections')
        
        for i in range(num_detections):
            cls_id = int(boxes.cls[i].item())
            confidence = float(boxes.conf[i].item())
            
            # Get mask and resize to original image dimensions
            mask = masks.data[i].cpu().numpy()
            mask_resized = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
            
            # Calculate x-coordinate from mask
            # Use middle portion of image (focus on area ahead of robot)
            roi_start = int(image_height * 0.4)  # Start at 40% down from top
            roi_end = int(image_height * 0.8)    # End at 80% down from top
            
            mask_roi = mask_resized[roi_start:roi_end, :]
            
            # Find x-coordinates where mask is active
            x_coords = np.where(np.any(mask_roi, axis=0))[0]
            
            if len(x_coords) == 0:
                continue
            
            # Calculate mean x position weighted by mask density
            x_positions = []
            weights = []
            for x in x_coords:
                column = mask_roi[:, x]
                weight = np.sum(column)
                if weight > 0:
                    x_positions.append(x)
                    weights.append(weight)
            
            if len(x_positions) == 0:
                continue
            
            mean_x = np.average(x_positions, weights=weights)
            
            # Classify as left or right lane and keep highest confidence detection
            if cls_id == self.left_lane_class_id:
                if confidence > left_confidence:
                    left_x = float(mean_x)
                    left_confidence = confidence
                    self.get_logger().debug(
                        f'Left lane detected: x={left_x:.1f}, conf={confidence:.2f}'
                    )
            elif cls_id == self.right_lane_class_id:
                if confidence > right_confidence:
                    right_x = float(mean_x)
                    right_confidence = confidence
                    self.get_logger().debug(
                        f'Right lane detected: x={right_x:.1f}, conf={confidence:.2f}'
                    )
        
        # Validate lane boundaries (right should be to the right of left)
        if left_x is not None and right_x is not None:
            if right_x <= left_x:
                self.get_logger().warn(
                    f'Invalid lane boundaries: left={left_x:.1f}, right={right_x:.1f}. '
                    'Right lane is not to the right of left lane.'
                )
                return None, None
            
            # Check if lane width is reasonable (not too narrow or too wide)
            lane_width = right_x - left_x
            min_width = image_width * 0.1  # At least 10% of image width
            max_width = image_width * 0.9  # At most 90% of image width
            
            if lane_width < min_width:
                self.get_logger().warn(
                    f'Lane too narrow: {lane_width:.1f}px (min: {min_width:.1f}px)'
                )
                return None, None
            elif lane_width > max_width:
                self.get_logger().warn(
                    f'Lane too wide: {lane_width:.1f}px (max: {max_width:.1f}px)'
                )
                return None, None
        
        return left_x, right_x
    
    def _create_debug_image(self, image, result, left_x, right_x):
        """Create visualization image with detected lanes"""
        debug_img = image.copy()
        h, w = debug_img.shape[:2]
        
        # Draw YOLO detections (masks and boxes)
        if result.masks is not None and len(result.masks) > 0:
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes
            
            for i, mask in enumerate(masks):
                # Resize mask to image size
                mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                
                # Get class and confidence
                cls_id = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())
                
                # Create colored overlay
                color = [0, 255, 0] if cls_id == self.left_lane_class_id else [255, 0, 0]
                colored_mask = np.zeros_like(debug_img)
                colored_mask[mask_resized > 0.5] = color
                debug_img = cv2.addWeighted(debug_img, 1.0, colored_mask, 0.4, 0)
                
                # Draw bounding box
                box = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
                
                # Add label
                label = f'{"Left" if cls_id == self.left_lane_class_id else "Right"} {confidence:.2f}'
                cv2.putText(debug_img, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw ROI boundaries
        roi_start = int(h * 0.4)
        roi_end = int(h * 0.8)
        cv2.line(debug_img, (0, roi_start), (w, roi_start), (255, 255, 0), 1)
        cv2.line(debug_img, (0, roi_end), (w, roi_end), (255, 255, 0), 1)
        
        # Draw lane boundary lines
        if left_x is not None:
            x = int(left_x)
            cv2.line(debug_img, (x, 0), (x, h), (0, 255, 0), 3)
            cv2.putText(debug_img, f'Left: {left_x:.1f}', (x + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if right_x is not None:
            x = int(right_x)
            cv2.line(debug_img, (x, 0), (x, h), (0, 0, 255), 3)
            cv2.putText(debug_img, f'Right: {right_x:.1f}', (x - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw center line and image center
        image_center_x = w // 2
        cv2.line(debug_img, (image_center_x, 0), (image_center_x, h), (128, 128, 128), 1)
        
        if left_x is not None and right_x is not None:
            lane_center_x = int((left_x + right_x) / 2)
            cv2.line(debug_img, (lane_center_x, 0), (lane_center_x, h), (0, 255, 255), 2)
            
            # Draw lateral error
            lateral_error = lane_center_x - image_center_x
            cv2.putText(debug_img, f'Error: {lateral_error:+.1f}px', (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Draw arrow showing error direction
            arrow_start = (image_center_x, h - 60)
            arrow_end = (lane_center_x, h - 60)
            cv2.arrowedLine(debug_img, arrow_start, arrow_end, (0, 255, 255), 3, tipLength=0.3)
        
        # Add frame info
        fps = 1.0 / (self.total_inference_time / max(self.frame_count, 1))
        cv2.putText(debug_img, f'Frame: {self.frame_count} | Avg FPS: {fps:.1f}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return debug_img
    
    def _log_performance_stats(self):
        """Log performance statistics"""
        avg_inference_time = self.total_inference_time / max(self.frame_count, 1)
        avg_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0.0
        detection_rate = (self.successful_detections / max(self.frame_count, 1)) * 100
        
        self.get_logger().info(
            f'\n{"="*60}\n'
            f'Performance Statistics (Last {self.log_interval} frames):\n'
            f'  Total frames processed: {self.frame_count}\n'
            f'  Successful detections: {self.successful_detections} ({detection_rate:.1f}%)\n'
            f'  Failed detections: {self.failed_detections}\n'
            f'  Average inference time: {avg_inference_time*1000:.1f}ms\n'
            f'  Average FPS: {avg_fps:.1f}\n'
            f'{"="*60}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = YOLOv11LaneSegmentationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down YOLOv11 Lane Segmentation Node')
    finally:
        # Log final statistics
        node.get_logger().info(
            f'\nFinal Statistics:\n'
            f'  Total frames: {node.frame_count}\n'
            f'  Successful: {node.successful_detections}\n'
            f'  Failed: {node.failed_detections}\n'
            f'  Success rate: {(node.successful_detections/max(node.frame_count,1)*100):.1f}%'
        )
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()