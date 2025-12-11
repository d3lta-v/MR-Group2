#!/usr/bin/env python3
import rclpy
import os
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from std_msgs.msg import Float32
from numpy.polynomial import Polynomial
from math import tan, degrees, radians
from ament_index_python.packages import get_package_share_directory

class YoloInferenceNode(Node):
    def __init__(self):
        super().__init__('yolo_inference_node')

        # Load inference engine
        package_share_directory = get_package_share_directory('yolo_inference')
        resource_path = os.path.join(package_share_directory, 'best.pt')

        # --- CONFIGURATION ---
        self.weights_path = resource_path # Make sure this path is absolute or correct relative to execution
        self.conf_threshold = 0.5
        self.iou_threshold = 0.7
        self.imgsz = 320
        self.device = 0 # use Jetson GPU with TensorRT
        self.actual_lane_width = 1.2 # meters, actual lane width for deviation calculation
        
        # --- ROS 2 INFRASTRUCTURE ---
        # Subscriber: Listens to your camera
        self.subscription = self.create_subscription(
            Image,
            '/zed/zed_node/rgb/image_rect_color',
            self.image_callback,
            10)
        
        # Publishers: Outputs results for your PID or debugging
        self.debug_pub = self.create_publisher(Image, '/yolo/debug_image', 10)
        self.track_pub = self.create_publisher(Image, '/yolo/track_image', 10)
        self.cte_pub = self.create_publisher(Float32, '/cte', 10)
        self.angle_e_pub = self.create_publisher(Float32, '/angle_error', 10)

        # Utilities
        self.bridge = CvBridge()
        self.model = YOLO(self.weights_path, task='segment')
        self.class_names = self.model.names
        
        self.get_logger().info(f"YOLOv11 Node Started. Loaded: {self.weights_path}")

    def compute_lane_deviation_and_angle(self, left_polygon, 
                                         right_polygon, 
                                         robot_center_x=0.5, 
                                         num_samples=10,
                                         fit_degree=1):
        """
        Compute lateral deviation and lane heading angle.
        
        Returns:
        - weighted_deviation_meters: Lateral offset from lane center (in meters)
        - heading_angle: Lane heading angle relative to robot (degrees)
        - lane_centers: Array of computed lane center points
        - p: Polynomial object for centre line of the two lanes
        - p_left: Polynomial object for the left lane line of best fit
        - p_right: Polynomial object for the right lane line of best fit
        """
        # Find common y-range
        y_min = max(left_polygon[:, 1].min(), right_polygon[:, 1].min())
        y_max = min(left_polygon[:, 1].max(), right_polygon[:, 1].max())
        y_samples = np.linspace(y_min, y_max, num_samples)
        
        # Compute lane centers at each depth
        lane_centers = []
        # deviations = []
        deviation_meters_list = []

        left_points = []
        right_points = []
        
        for y in y_samples:
            left_idx = np.argmin(np.abs(left_polygon[:, 1] - y))
            right_idx = np.argmin(np.abs(right_polygon[:, 1] - y))
            
            left_x = left_polygon[left_idx, 0]
            right_x = right_polygon[right_idx, 0]

            left_points.append([left_x, y])
            right_points.append([right_x, y])

            center_x = (left_x + right_x) / 2
            lane_centers.append([center_x, y])

            deviation_normalized = robot_center_x - center_x  # positive means lane center is to the right of robot center

            lane_width_normalized = right_x - left_x  # at reference point
            meters_per_unit = self.actual_lane_width / lane_width_normalized
            deviation_meters = deviation_normalized * meters_per_unit
            
            # deviations.append(deviation_normalized)
            deviation_meters_list.append(deviation_meters)
        
        lane_centers = np.array(lane_centers)
        left_points = np.array(left_points)
        right_points = np.array(right_points)
        
        # Weighted deviation (closer points more important)
        weights = y_samples
        # weighted_deviation = np.average(deviations, weights=weights) # Weights are heavier linearly the lower in the image (closer to robot)
        weighted_deviation_meters = np.average(deviation_meters_list, weights=weights)
        
        # Compute line of best fit of the lane centre for angle calculation. This is NOT the same as the deviation calculation above.
        p = Polynomial.fit(lane_centers[:, 1], lane_centers[:, 0], fit_degree) # first degree polynomial is a line
        
        # Compute heading angle at closest point to the robot (at the lowest part of the image, which is the maximal y value)
        y_closest = y_max
        derivative = p.deriv()
        slope_dx_dy = derivative(y_closest)

        # Convert slope to angle
        angle_deg = np.degrees(np.arctan(slope_dx_dy))

        # LANE ORIENTATION (not lane curvature!)
        # Custom lane fit algorithm for angle calculation
        p_left = Polynomial.fit(left_points[:, 1], left_points[:, 0], fit_degree)
        p_right = Polynomial.fit(right_points[:, 1], right_points[:, 0], fit_degree)
        diff_poly = p_right - p_left
        
        # Find vanishing point where left and right lanes intersect
        intersection_x_coords = diff_poly.roots()
        real_intersection_x = intersection_x_coords[np.isreal(intersection_x_coords)].real

        # Compute vehicle-to-lane heading angle via the vanishing point
        vx = p_left(real_intersection_x)[0]
        cx = 0.5
        fx = 1 / (2*tan(radians(110/2)))  # ZED 2 uses a 110 degree horizontal FOV camera

        heading_angle = degrees(np.arctan((vx - cx) / fx))

        # First 2 parameters are for control, the other 3 are for debug messages
        return weighted_deviation_meters, heading_angle, lane_centers, p, p_left, p_right

    def image_callback(self, msg):
        try:
            # 1. Convert ROS Image -> OpenCV Image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 2. Run Inference (Single Frame)
            results = self.model.predict(
                source=cv_image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False, # Reduce terminal clutter
                retina_masks=False # Set False for speed, True for precision
            )
            
            # 3. Process Results
            result = results[0]

            # Extract lane classes for indices
            lane_indices = []
            for i, box in enumerate(result.boxes):
                cls_id = int(box.cls[0])
                cls_name = self.class_names[cls_id] if cls_id in self.class_names else "Unknown"
                conf = box.conf[0]
                if (cls_name == 'lane'):
                    lane_indices.append(i)

            # Masks should be processed as xyn (xy normalized) format for path segmentation training
            # Here we save the masks in JSON format as a list of polygons
            masks_polygons = []
            for i, mask in enumerate(result.masks):
                if i not in lane_indices:
                    continue
                polygons = mask.xyn  # Convert to list for JSON serialization
                masks_polygons.append(polygons)

            if len(masks_polygons) < 2:
                self.get_logger().warning("Less than two lane masks detected, skipping publish")
                return
            
            # Draw a new image directly with normalised polygons for visualization. original image is 320x180
            height = 180*3
            width = 320*3

            # Create a black image (3 channels for BGR, 8-bit unsigned integers for pixel values)
            # np.zeros creates an array filled with zeros, which corresponds to black in BGR
            img = np.zeros((height, width, 3), np.uint8)

            # Detect which is the left lane, by simply checking the x-coordinate
            left_lane_0 = np.mean(masks_polygons[0][0][:, 0]) < np.mean(masks_polygons[1][0][:, 0])
            left_lane_mask = masks_polygons[0] if left_lane_0 else masks_polygons[1]
            right_lane_mask = masks_polygons[1] if left_lane_0 else masks_polygons[0]

            left_lane_pixels = (left_lane_mask * np.array([width, height])).astype(np.int32)
            right_lane_pixels = (right_lane_mask * np.array([width, height])).astype(np.int32)

            cv2.fillPoly(img, [left_lane_pixels], color=(0, 255, 0))   # Green
            cv2.fillPoly(img, [right_lane_pixels], color=(0, 0, 255))  # Red

            deviation_meters, heading_angle, lane_centers, p, p_left, p_right = self.compute_lane_deviation_and_angle(
                left_lane_mask[0], 
                right_lane_mask[0], 
                num_samples=10)
            
            self.get_logger().info(f"Lateral Deviation (meters): {deviation_meters:.4f}, Vehicle-to-Lane Heading Angle: {heading_angle:.2f} degrees")

            cv2.polylines(img, [(lane_centers * np.array([width, height])).astype(np.int32)], isClosed=False, color=(255, 255, 0), thickness=2)

            # Draw the polynomial line of best fit
            x_values = np.arange(0.0, 1.0, 0.01) # normalized x values
            y_values_left = p_left(x_values)
            y_values_right = p_right(x_values)
            points_left = (np.array([y_values_left, x_values], dtype=np.float32).T * np.array([width, height])).astype(np.int32)
            cv2.polylines(img, [points_left], isClosed=False, color=(255, 0, 255), thickness=2)
            points_right = (np.array([y_values_right, x_values], dtype=np.float32).T * np.array([width, height])).astype(np.int32)
            cv2.polylines(img, [points_right], isClosed=False, color=(255, 0, 255), thickness=2)

            # 4. Publish Detection Data
            cte_msg = Float32()
            cte_msg.data = float(deviation_meters)
            self.cte_pub.publish(cte_msg)
            angle_e_msg = Float32()
            angle_e_msg.data = float(heading_angle)
            self.angle_e_pub.publish(angle_e_msg)

            # 5. Publish Debug Images (with bounding boxes drawn, and track lines)
            annotated_frame = result.plot()
            annotated_ros_image = self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
            self.debug_pub.publish(annotated_ros_image)
            # Your PID node can subscribe to '/yolo/track_image' to view these images
            track_pub_msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
            self.track_pub.publish(track_pub_msg)

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
