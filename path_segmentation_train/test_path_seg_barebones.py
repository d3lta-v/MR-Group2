from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from numpy.polynomial import Polynomial
from ultralytics import YOLO
from math import tan, degrees, radians
# from cv_bridge import CvBridge

# Keep in mind that negative angular velocity indicates a turn to the right, and positive indicates a turn to the left for ROS coordinate system.
# In this system, positive angle means you should turn right to align with the lane, negative means turn left.

# Configuration
WEIGHTS_PATH = "best.pt"  # use the .engine file for Jetson devices, otherwise use .pt
IMAGE_SOURCE = "../dataset/coco"  # Single file
OUTPUT_DIR = "single_result"
CONF_THRESHOLD = 0.5      # higher confidence for better precision
IOU_THRESHOLD = 0.7       # NMS IoU threshold, higher will result in more accurate segmentation at the cost of object detection
IMGSZ = 320
DEVICE = 'cpu'  # 0 for GPU, 'cpu' for CPU, 'mps' for Apple Silicon
ACTUAL_LANE_WIDTH = 1.2  # meters, actual lane width for deviation calculation

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
        meters_per_unit = ACTUAL_LANE_WIDTH / lane_width_normalized
        deviation_meters = deviation_normalized * meters_per_unit
        
        deviations.append(deviation_normalized)
        deviation_meters_list.append(deviation_meters)
    
    lane_centers = np.array(lane_centers)
    left_points = np.array(left_points)
    right_points = np.array(right_points)
    
    # Weighted deviation (closer points more important)
    weights = y_samples
    weighted_deviation = np.average(deviations, weights=weights) # Weights are heavier linearly the lower in the image (closer to robot)
    weighted_deviation_meters = np.average(deviation_meters_list, weights=weights)
    
    # Compute line of best fit of the lane centre for angle calculation. This is NOT the same as the deviation calculation above.
    p = Polynomial.fit(lane_centers[:, 1], lane_centers[:, 0], fit_degree) # first degree polynomial is a line
    print("Polynomial:", p)
    
    # Compute heading angle at closest point to the robot (at the lowest part of the image, which is the maximal y value)
    y_closest = y_max
    derivative = p.deriv()
    slope_dx_dy = derivative(y_closest)

    # Convert slope to angle
    angle_deg = np.degrees(np.arctan(slope_dx_dy))

    # LANE ORIENTATION (not lane curvature!)
    # Use only nearest points to avoid perspective distortion
    # near_threshold = 0.7
    # y_cutoff = y_min + (y_max - y_min) * near_threshold

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

    return weighted_deviation, weighted_deviation_meters, angle_deg, lane_centers, p, heading_angle, p_left, p_right


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
    retina_masks=False  # High-resolution masks
)

# print(f"Inference completed. {len(result)} result(s) obtained.")

for result in results_w:
    # res: results.Results = result[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Print results
    lane_indices = []
    print(f"Detected {len(result.boxes)} objects.")
    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls[0])
        cls_name = class_names[cls_id] if cls_id in class_names else "Unknown"
        conf = box.conf[0]
        if (cls_name == 'lane'):
            lane_indices.append(i)
        print(f"Object {i}: Class ID: {cls_id}, Class Name: {cls_name}, Confidence: {conf:.2f}")

    # Masks should be processed as xyn format for path segmentation training
    # Here we save the masks in JSON format as a list of polygons
    masks_polygons = []
    for i, mask in enumerate(result.masks):
        if i not in lane_indices:
            continue
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

    left_lane_0 = np.mean(masks_polygons[0][0][:, 0]) < np.mean(masks_polygons[1][0][:, 0])
    left_lane_mask = masks_polygons[0] if left_lane_0 else masks_polygons[1]
    right_lane_mask = masks_polygons[1] if left_lane_0 else masks_polygons[0]

    left_lane_pixels = (left_lane_mask * np.array([width, height])).astype(np.int32)
    right_lane_pixels = (right_lane_mask * np.array([width, height])).astype(np.int32)

    cv2.fillPoly(img, [left_lane_pixels], color=(0, 255, 0))   # Green
    cv2.fillPoly(img, [right_lane_pixels], color=(0, 0, 255))  # Red

    deviation, deviation_m, angle_deg, lane_centers, p, heading_angle, p_left, p_right = compute_lane_deviation_and_angle(left_lane_mask[0], right_lane_mask[0], num_samples=10)
    print(f"Lateral Deviation (meters): {deviation_m:.4f}, Vehicle-to-Lane Heading Angle: {heading_angle:.2f} degrees")

    cv2.polylines(img, [(lane_centers * np.array([width, height])).astype(np.int32)], isClosed=False, color=(255, 255, 0), thickness=2)

    # Draw the polynomial line of best fit
    x_values = np.arange(0.0, 1.0, 0.01) # normalized x values
    y_values_left = p_left(x_values)
    y_values_right = p_right(x_values)
    points_left = (np.array([y_values_left, x_values], dtype=np.float32).T * np.array([width, height])).astype(np.int32)
    cv2.polylines(img, [points_left], isClosed=False, color=(255, 0, 255), thickness=2)
    points_right = (np.array([y_values_right, x_values], dtype=np.float32).T * np.array([width, height])).astype(np.int32)
    cv2.polylines(img, [points_right], isClosed=False, color=(255, 0, 255), thickness=2)

    cv2.imshow("Polygons Visualization", img)
    cv2.imshow("Original Image", result.orig_img)
    cv2.imwrite(str(output_path / f"polygons_viz_{timestamp}.png"), img)
    cv2.waitKey(3000)  # Display for 3 seconds
