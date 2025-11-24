#!/usr/bin/env python3
"""
ROS2 Camera Capture Node
Captures images from a camera topic every 5 seconds and saves them to disk.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from datetime import datetime


class CameraCaptureNode(Node):
    def __init__(self):
        super().__init__('camera_capture_node')
        
        # Declare parameters
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('save_directory', '~/camera_captures')
        self.declare_parameter('capture_interval', 5.0)  # seconds
        self.declare_parameter('image_format', 'jpg')
        
        # Get parameters
        self.camera_topic = self.get_parameter('camera_topic').value
        save_dir = self.get_parameter('save_directory').value
        self.capture_interval = self.get_parameter('capture_interval').value
        self.image_format = self.get_parameter('image_format').value
        
        # Expand user path and create directory
        self.save_directory = os.path.expanduser(save_dir)
        os.makedirs(self.save_directory, exist_ok=True)
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        self.latest_image = None
        
        # Subscribe to camera topic
        self.subscription = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            10
        )
        
        # Create timer for periodic capture
        self.timer = self.create_timer(self.capture_interval, self.capture_callback)
        
        self.get_logger().info(f'Camera capture node started')
        self.get_logger().info(f'Subscribing to: {self.camera_topic}')
        self.get_logger().info(f'Saving to: {self.save_directory}')
        self.get_logger().info(f'Capture interval: {self.capture_interval} seconds')
    
    def image_callback(self, msg):
        """Store the latest image from the camera topic"""
        self.latest_image = msg
    
    def capture_callback(self):
        """Capture and save the current image"""
        if self.latest_image is None:
            self.get_logger().warn('No image received yet')
            return
        
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='bgr8')
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'capture_{timestamp}.{self.image_format}'
            filepath = os.path.join(self.save_directory, filename)
            
            # Save image
            cv2.imwrite(filepath, cv_image)
            
            self.get_logger().info(f'Saved image: {filename}')
            
        except Exception as e:
            self.get_logger().error(f'Failed to capture image: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = CameraCaptureNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
