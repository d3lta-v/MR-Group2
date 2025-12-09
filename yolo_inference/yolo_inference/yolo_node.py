#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
from ultralytics import YOLO

class YoloInferenceNode(Node):
    def __init__(self):
        super().__init__('yolo_inference_node')

        # --- CONFIGURATION ---
        self.weights_path = "best.engine" # Make sure this path is absolute or correct relative to execution
        self.conf_threshold = 0.25
        self.device = '0' # 'cpu' or '0'    
            
        # --- ROS 2 INFRASTRUCTURE ---
        # Subscriber: Listens to your camera
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',  # CHANGE THIS to your actual camera topic name
            self.image_callback,
            10)
        
        # Publishers: Outputs results for your PID or debugging
        self.debug_pub = self.create_publisher(Image, '/yolo/debug_image', 10)
        self.data_pub = self.create_publisher(String, '/yolo/detections', 10)
        
        # Utilities
        self.bridge = CvBridge()
        self.model = YOLO(self.weights_path, task='segment')
        self.class_names = self.model.names
        
        self.get_logger().info(f"YOLOv11 Node Started. Loaded: {self.weights_path}")

    def image_callback(self, msg):
        try:
            # 1. Convert ROS Image -> OpenCV Image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 2. Run Inference (Single Frame)
            results = self.model.predict(
                source=cv_image,
                conf=self.conf_threshold,
                device=self.device,
                verbose=False, # Reduce terminal clutter
                retina_masks=False # Set False for speed, True for precision
            )
            
            # 3. Process Results
            result = results[0]
            detection_summary = []
            
            if result.masks is not None:
                # Calculate centroids or masks for your PID controller
                for i, box in enumerate(result.boxes):
                    # Get class and confidence
                    cls_id = int(box.cls[0].item())
                    class_name = self.class_names[cls_id]
                    
                    # Get Bounding Box Center (useful for simple PID tracking)
                    x, y, w, h = box.xywh[0].cpu().numpy()
                    
                    detection_info = {
                        "class": class_name,
                        "center_x": float(x),
                        "center_y": float(y),
                        "width": float(w),
                        "height": float(h)
                    }
                    detection_summary.append(detection_info)

            # 4. Publish Detection Data (JSON string for simplicity)
            # Your PID node can subscribe to '/yolo/detections' and parse this JSON
            msg_str = String()
            msg_str.data = json.dumps(detection_summary)
            self.data_pub.publish(msg_str)

            # 5. Publish Visual Debug Image (with bounding boxes drawn)
            annotated_frame = result.plot()
            ros_image = self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
            self.debug_pub.publish(ros_image)

        except Exception as e:
            self.get_logger().error(f"Inference Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = YoloInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
