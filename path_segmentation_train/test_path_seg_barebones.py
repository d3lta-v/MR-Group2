import json
import csv
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from numpy.polynomial import Polynomial
from ultralytics import YOLO
from ultralytics.engine import results
# from cv_bridge import CvBridge

# Configuration
WEIGHTS_PATH = "best.pt"  # use the .engine file for Jetson devices, otherwise use .pt
IMAGE_SOURCE = "../dataset/coco"  # Single file
OUTPUT_DIR = "single_result"
CONF_THRESHOLD = 0.5      # higher confidence for better precision
IOU_THRESHOLD = 0.7       # NMS IoU threshold, higher will result in more accurate segmentation at the cost of object detection
IMGSZ = 320
DEVICE = 'cpu'  # 0 for GPU, 'cpu' for CPU, 'mps' for Apple Silicon

def compute_lane_deviation_and_angle(left_polygon, right_polygon, 
                                     robot_center_x=0.5, 
                                     num_samples=10,
                                     fit_degree=1):
    """
    Compute lateral deviation and lane heading angle.
    
    Returns:
    - deviation: Lateral offset from lane center (normalized units)
    - angle_deg: Lane heading angle relative to robot (degrees)
    - lane_centers: Array of computed lane center points
    """
    # Find common y-range
    y_min = max(left_polygon[:, 1].min(), right_polygon[:, 1].min())
    y_max = min(left_polygon[:, 1].max(), right_polygon[:, 1].max())
    y_samples = np.linspace(y_min, y_max, num_samples)

    print(f"Y range for lane center computation: {y_min} to {y_max}")
    
    # Compute lane centers at each depth
    lane_centers = []
    deviations = []
    
    for y in y_samples:
        left_idx = np.argmin(np.abs(left_polygon[:, 1] - y))
        right_idx = np.argmin(np.abs(right_polygon[:, 1] - y))
        
        left_x = left_polygon[left_idx, 0]
        right_x = right_polygon[right_idx, 0]
        
        center_x = (left_x + right_x) / 2
        lane_centers.append([center_x, y])
        deviations.append(robot_center_x - center_x)
    
    lane_centers = np.array(lane_centers)
    
    # Weighted deviation (closer points more important)
    weights = y_samples
    weighted_deviation = np.average(deviations, weights=weights) # Weights are heavier linearly the lower in the image (closer to robot)
    
    # Compute line of best fit of the lane centre for angle calculation. This is NOT the same as the deviation calculation above.
    p = Polynomial.fit(lane_centers[:, 1], lane_centers[:, 0], fit_degree) # first degree polynomial is a line
    print("Polynomial:", p)
    
    # Compute heading angle at closest point to the robot (at the lowest part of the image, which is the maximal y value)
    y_closest = y_max
    derivative = p.deriv()
    slope_dx_dy = derivative(y_closest)
    
    # Convert slope to angle
    angle_rad = np.arctan(slope_dx_dy)
    angle_deg = np.degrees(angle_rad)
    
    return weighted_deviation, angle_deg, lane_centers, p



# Initialize model
model = YOLO(WEIGHTS_PATH, task='segment')
class_names = model.names

output_path = Path(OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)

# The actual image source is from ROS, the code should look like this from the callback function for Image message:
# In addition, the image is expected to be streamed at 15Hz
# cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
# and then set source=cv_image in the predict function below

# This is a mock version for testing purpose only, by reading a local image file as CV2 image
# cv2img = cv2.imread(IMAGE_SOURCE)

results_w = model.predict(
    source=IMAGE_SOURCE,
    conf=CONF_THRESHOLD,
    iou=IOU_THRESHOLD,
    imgsz=IMGSZ,
    device=DEVICE,
    verbose=True,
    stream=True,
    retina_masks=True  # High-resolution masks
)

# print(f"Inference completed. {len(result)} result(s) obtained.")

for result in results_w:
    # res: results.Results = result[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Print results
    print(f"Detected {len(result.boxes)} objects.")
    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls[0])
        cls_name = class_names[cls_id] if cls_id in class_names else "Unknown"
        conf = box.conf[0]
        print(f"Object {i}: Class ID: {cls_id}, Class Name: {cls_name}, Confidence: {conf:.2f}")

    # Save annotated image
    # annotated_img = res.plot()
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # output_image_path = output_path / f"annotated_{timestamp}.png"
    # cv2.imwrite(str(output_image_path), annotated_img)
    # print(f"Annotated image saved to {output_image_path}")

    # Save masks as separate images
    # for i, mask in enumerate(res.masks.data):
    #     mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
    #     output_mask_path = output_path / f"mask_{i}_{timestamp}.png"
    #     cv2.imwrite(str(output_mask_path), mask_np)
    #     print(f"Mask {i} saved to {output_mask_path}")

    # Masks should be processed as xyn format for path segmentation training
    # Here we save the masks in JSON format as a list of polygons
    masks_polygons = []
    for mask in result.masks:
        polygons = mask.xyn  # Convert to list for JSON serialization
        masks_polygons.append(polygons)

    if len(masks_polygons) < 2:
        print("Less than two lane masks detected, skipping polygon visualization")
        continue

    # Draw a new image directly with normalised polygons for better visualization. original image is 320x180
    height = 180*4
    width = 320*4

    # Create a black image (3 channels for BGR, 8-bit unsigned integers for pixel values)
    # np.zeros creates an array filled with zeros, which corresponds to black in BGR
    img = np.zeros((height, width, 3), np.uint8)

    left_lane_pixels = (masks_polygons[0] * np.array([width, height])).astype(np.int32)
    right_lane_pixels = (masks_polygons[1] * np.array([width, height])).astype(np.int32)

    cv2.fillPoly(img, [left_lane_pixels], color=(0, 255, 0))   # Green
    cv2.fillPoly(img, [right_lane_pixels], color=(0, 0, 255))  # Red

    deviation, angle_deg, lane_centers, p = compute_lane_deviation_and_angle(masks_polygons[0][0], masks_polygons[1][0], num_samples=100)
    print(f"Lateral Deviation: {deviation:.4f}, Lane Heading Angle: {angle_deg:.2f} degrees")

    cv2.polylines(img, [(lane_centers * np.array([width, height])).astype(np.int32)], isClosed=False, color=(255, 255, 0), thickness=2)

    # Draw the polynomial line of best fit
    x_values = np.arange(0.0, 1.0, 0.01) # normalized x values
    y_values = p(x_values)
    # y_values = np.polynomial.polyval(x_values / height, p) * height
    points = (np.array([y_values, x_values], dtype=np.float32).T * np.array([width, height])).astype(np.int32)
    cv2.polylines(img, [points], isClosed=False, color=(255, 0, 255), thickness=2)

    cv2.imshow("Polygons Visualization", img)
    cv2.imwrite(str(output_path / f"polygons_viz_{timestamp}.png"), img)
    cv2.waitKey(3000)  # Display for 1 second

