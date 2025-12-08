# Autonomous Racing System - Complete Guide

A comprehensive ROS2 package for autonomous racing with YOLOv11-based perception, PID control, cone-based localization, and autonomous navigation.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Package Structure](#package-structure)
- [Available Nodes](#available-nodes)
- [Launch Files](#launch-files)
- [Configuration Files](#configuration-files)
- [Use Cases](#use-cases)
- [Quick Start Examples](#quick-start-examples)
- [Visualization](#visualization)
- [Parameter Reference](#parameter-reference)
- [Troubleshooting](#troubleshooting)

---

## Overview

This package provides a complete autonomous racing system with three main capabilities:

### 1. **Lane Following**
- Real-time lane boundary detection using YOLOv11 segmentation
- PID-based lateral control for smooth lane following
- Configurable forward speed and control gains
- Safety features (watchdog, error thresholds)

### 2. **Cone Mapping**
- Real-time cone detection using YOLOv11 detection
- 3D localization using depth camera or geometric estimation
- Persistent cone map with duplicate filtering
- Multi-observation averaging for accuracy

### 3. **Autonomous Navigation**
- Navigate to goal position using pre-mapped cones
- Cone-based localization corrects odometry drift
- Autonomous control with obstacle awareness
- Real-time visualization and monitoring

**Key Technologies:**
- YOLOv11 (10-30+ FPS with GPU)
- ZED2 stereo camera (RGB + Depth)
- ROS2 (Humble or newer)
- TF2 for coordinate transformations
- PID control for smooth motion

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AUTONOMOUS RACING SYSTEM                      │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  Lane Following  │      │   Cone Mapping   │      │ Autonomous Nav   │
├──────────────────┤      ├──────────────────┤      ├──────────────────┤
│ • Lane Detection │      │ • Cone Detection │      │ • Goal Navigation│
│ • PID Control    │      │ • 3D Localization│      │ • Cone Matching  │
│ • Velocity Cmd   │      │ • Map Building   │      │ • Pose Correction│
└──────────────────┘      └──────────────────┘      └──────────────────┘
         │                         │                         │
         └─────────────────────────┴─────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │     Shared Components       │
                    ├─────────────────────────────┤
                    │ • YOLOv11 Inference         │
                    │ • ZED2 Camera Interface     │
                    │ • TF2 Transformations       │
                    │ • RViz Visualization        │
                    │ • Parameter Management      │
                    └─────────────────────────────┘
```

---

## Installation

### Prerequisites

**Hardware:**
- Mobile robot with differential drive
- ZED2 stereo camera (RGB + Depth)
- GPU recommended (CUDA) for real-time performance

**Software:**
- Ubuntu 22.04
- ROS2 Humble or newer
- Python 3.8+
- CUDA 11.8+ (optional, for GPU)

### Install Dependencies

```bash
# ROS2 packages
sudo apt update
sudo apt install ros-${ROS_DISTRO}-cv-bridge \
                 ros-${ROS_DISTRO}-image-transport \
                 ros-${ROS_DISTRO}-tf2-geometry-msgs

# Python packages
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip3 install ultralytics opencv-python numpy
```

### Build Package

```bash
# Clone repository
cd ~/ros2_ws/src
git clone <your-repo-url> lane_following_yolov11

# Build
cd ~/ros2_ws
colcon build --packages-select lane_following_yolov11

# Source
source install/setup.bash
```

### Setup Models

```bash
# Place your trained models in the models directory
cp your_lane_model.pt ~/ros2_ws/src/lane_following_yolov11/models/lane_model.pt
cp your_cone_model.pt ~/ros2_ws/src/lane_following_yolov11/models/cone_model.pt
```

---

## Package Structure

```
lane_following_yolov11/
├── lane_following_yolov11/              # Python package
│   ├── __init__.py
│   ├── yolov11_lane_segmentation_node.py
│   ├── lane_following_controller_node.py
│   ├── cone_localization_node.py
│   └── autonomous_cone_navigation_node.py
│
├── launch/                               # Launch files
│   ├── lane_following_yolov11.launch.py          # Lane following
│   ├── lane_following_debug.launch.py            # Debug mode
│   ├── cone_localization.launch.py               # Cone mapping
│   ├── autonomous_navigation.launch.py           # Autonomous nav
│   └── full_system.launch.py                     # All nodes
│
├── config/                               # Configuration files
│   ├── params.yaml                       # Lane following params
│   ├── cone_params.yaml                  # Cone localization params
│   └── cone_map.json                     # Pre-mapped cone positions
│
├── rviz/                                 # RViz configurations
│   ├── cone_visualization.rviz
│   └── autonomous_navigation.rviz
│
├── models/                               # Trained models (user provided)
│   ├── lane_model.pt
│   └── cone_model.pt
│
├── package.xml
├── setup.py
└── README.md
```

---

## Available Nodes

### 1. Lane Segmentation Node
**Executable:** `lane_segmentation_node`

**Purpose:** Detect lane boundaries using YOLOv11 segmentation

**Key Features:**
- Fast inference (10-30+ FPS on GPU)
- Publishes lane boundary x-coordinates
- Debug visualization with detected lanes
- Performance logging

**Topics:**
- Subscribes: `/zed2/zed_node/left/image_rect_color`
- Publishes: `/lane_boundaries`, `/lane_segmentation/debug_image`

---

### 2. Lane Controller Node
**Executable:** `lane_controller_node`

**Purpose:** PID-based lane following control

**Key Features:**
- PID control with runtime tuning
- Watchdog timer for safety
- Detailed logging of PID components
- Error thresholds and clamping

**Topics:**
- Subscribes: `/lane_boundaries`
- Publishes: `/cmd_vel`, `/lane_lateral_error`

---

### 3. Cone Localization Node
**Executable:** `cone_localization_node`

**Purpose:** Detect cones and build persistent map

**Key Features:**
- Real-time cone detection
- 3D localization (depth camera or geometric)
- Duplicate filtering and multi-observation averaging
- RViz visualization with markers

**Topics:**
- Subscribes: `/zed2/zed_node/left/image_rect_color`, `/zed2/zed_node/depth/depth_registered`
- Publishes: `/cone_map/markers`, `/cone_map/poses`, `/cone_detection/debug_image`

---

### 4. Autonomous Navigation Node
**Executable:** `autonomous_cone_navigation_node`

**Purpose:** Navigate to goal using cone-based localization

**Key Features:**
- Loads pre-mapped cone positions
- Cone-based localization (corrects odometry)
- Autonomous navigation to goal
- Correctly labels and determines cone positions

**Topics:**
- Subscribes: `/zed2/zed_node/left/image_rect_color`, `/zed2/zed_node/depth/depth_registered`
- Publishes: `/cmd_vel`, `/estimated_pose`, `/reference_cone_map/markers`, `/goal_marker`

---

## Launch Files

### 1. Lane Following
**File:** `lane_following_yolov11.launch.py`

**Purpose:** Lane following with control enabled

```bash
ros2 launch lane_following_yolov11 lane_following_yolov11.launch.py \
  model_path:=models/lane_model.pt \
  device:=cuda \
  enable_control:=true
```

**Nodes:** Lane Segmentation + Lane Controller

**Use When:** Following lane lines on track

---

### 2. Lane Following Debug
**File:** `lane_following_debug.launch.py`

**Purpose:** Test lane detection without moving robot

```bash
ros2 launch lane_following_yolov11 lane_following_debug.launch.py \
  model_path:=models/lane_model.pt
```

**Features:**
- Control disabled (safe testing)
- Debug-level logging
- Good for PID tuning

**Use When:** Testing new models, tuning parameters, playing rosbags

---

### 3. Cone Localization
**File:** `cone_localization.launch.py`

**Purpose:** Build map of cone positions

```bash
ros2 launch lane_following_yolov11 cone_localization.launch.py \
  cone_model_path:=models/cone_model.pt \
  use_depth_camera:=true
```

**Nodes:** Cone Localization

**Use When:** Mapping track cones, building reference map

---

### 4. Autonomous Navigation
**File:** `autonomous_navigation.launch.py`

**Purpose:** Navigate to goal using pre-mapped cones

```bash
ros2 launch lane_following_yolov11 autonomous_navigation.launch.py \
  cone_model_path:=models/cone_model.pt \
  cone_map_file:=config/cone_map.json \
  goal_x:=2.5 \
  goal_y:=0.0 \
  goal_z:=0.0
```

**Nodes:** Autonomous Navigation

**Use When:** 
- Given pre-mapped area
- Need to reach specific goal position
- Require cone-based localization

---

### 5. Full System
**File:** `full_system.launch.py`

**Purpose:** Run complete system (all nodes)

```bash
ros2 launch lane_following_yolov11 full_system.launch.py \
  lane_model_path:=models/lane_model.pt \
  cone_model_path:=models/cone_model.pt \
  enable_control:=true
```

**Nodes:** Lane Segmentation + Lane Controller + Cone Localization

**Use When:** Racing with simultaneous lane following and cone mapping

---

## Configuration Files

### 1. Lane Following Parameters
**File:** `config/params.yaml`

```yaml
/lane_controller_node:
  ros__parameters:
    forward_speed: 1.0          # m/s
    Kp: 0.005                   # Proportional gain
    Kd: 0.01                    # Derivative gain
    Ki: 0.0001                  # Integral gain
    max_angular_velocity: 0.5   # rad/s
    enable_control: true
    lateral_error_threshold: 0.3
    log_interval: 10

/yolov11_lane_segmentation_node:
  ros__parameters:
    confidence_threshold: 0.5
    iou_threshold: 0.45
    imgsz: 640
    device: 'cuda'
    half_precision: true
    left_lane_class_id: 0
    right_lane_class_id: 1
```

**When to Edit:**
- Tune PID gains for your robot
- Adjust detection thresholds
- Change speeds or error limits

---

### 2. Cone Localization Parameters
**File:** `config/cone_params.yaml`

```yaml
/cone_localization_node:
  ros__parameters:
    confidence_threshold: 0.6
    cone_merge_distance: 0.5    # meters
    max_detection_distance: 20.0
    min_detection_distance: 0.5
    cone_height: 0.3            # meters
    use_depth_camera: true
```

**When to Edit:**
- Adjust cone merge threshold
- Change detection ranges
- Update cone height for your cones

---

### 3. Cone Map
**File:** `config/cone_map.json`

**Format:**
```json
[
  {
    "id": 0,
    "label": "Start_Left",
    "x": 0.0,
    "y": -1.0,
    "z": 0.0
  },
  {
    "id": 1,
    "label": "Start_Right",
    "x": 0.0,
    "y": 1.0,
    "z": 0.0
  },
  {
    "id": 2,
    "label": "Goal_Left",
    "x": 2.5,
    "y": -0.5,
    "z": 0.0
  },
  {
    "id": 3,
    "label": "Goal_Right",
    "x": 2.5,
    "y": 0.5,
    "z": 0.0
  }
]
```

**When to Create:**
- Given pre-mapped area
- Need reference positions for localization
- Want meaningful cone labels

---

## Use Cases

### Use Case 1: Lane Following Race

**Scenario:** Follow lane lines autonomously at speed

**Commands:**
```bash
# 1. Test first (control disabled)
ros2 launch lane_following_yolov11 lane_following_debug.launch.py \
  model_path:=models/lane_model.pt

# 2. View debug image
ros2 run rqt_image_view rqt_image_view /lane_segmentation/debug_image

# 3. Tune PID if needed
ros2 run rqt_reconfigure rqt_reconfigure

# 4. Enable control and race
ros2 launch lane_following_yolov11 lane_following_yolov11.launch.py \
  model_path:=models/lane_model.pt \
  enable_control:=true
```

---

### Use Case 2: Build Cone Map

**Scenario:** Create map of track cone positions

**Commands:**
```bash
# 1. Launch cone localization
ros2 launch lane_following_yolov11 cone_localization.launch.py \
  cone_model_path:=models/cone_model.pt

# 2. Visualize in RViz
rviz2

# 3. Drive around track (manually or with lane following)
# Cones are automatically detected and mapped

# 4. Save map (implement save functionality or use rosbag)
ros2 bag record /cone_map/poses
```

---

### Use Case 3: Navigate to Goal

**Scenario:** Autonomous navigation to (2.5, 0.0, 0.0) using pre-mapped cones

**Requirements:**
- Pre-mapped cone positions in `config/cone_map.json`
- Cone detection model

**Commands:**
```bash
# 1. Verify cone map
cat config/cone_map.json

# 2. Launch autonomous navigation
ros2 launch lane_following_yolov11 autonomous_navigation.launch.py \
  cone_model_path:=models/cone_model.pt \
  cone_map_file:=config/cone_map.json \
  goal_x:=2.5 \
  goal_y:=0.0 \
  goal_z:=0.0

# 3. Visualize in RViz
rviz2

# 4. Monitor progress
# Watch for "GOAL REACHED!" message
```

**What Happens:**
1. System loads reference cone map
2. Detects cones in real-time
3. Matches detections to reference (labels cones correctly)
4. Corrects robot position using cone observations
5. Navigates to goal autonomously
6. Stops when within tolerance

---

### Use Case 4: Complete Race

**Scenario:** Full autonomous race with lane following and cone mapping

**Commands:**
```bash
ros2 launch lane_following_yolov11 full_system.launch.py \
  lane_model_path:=models/lane_model.pt \
  cone_model_path:=models/cone_model.pt \
  enable_control:=true
```

**Features:**
- Follows lane lines
- Maps cone positions
- Real-time visualization
- All data logged

---

## Quick Start Examples

### Example 1: First Time Setup

```bash
# 1. Install dependencies
sudo apt install ros-humble-cv-bridge ros-humble-tf2-geometry-msgs
pip3 install ultralytics torch torchvision opencv-python

# 2. Build package
cd ~/ros2_ws
colcon build --packages-select lane_following_yolov11
source install/setup.bash

# 3. Place models
cp lane_model.pt ~/ros2_ws/src/lane_following_yolov11/models/
cp cone_model.pt ~/ros2_ws/src/lane_following_yolov11/models/

# 4. Test camera
ros2 topic list | grep zed2
ros2 topic hz /zed2/zed_node/left/image_rect_color

# 5. Run first test (debug mode)
ros2 launch lane_following_yolov11 lane_following_debug.launch.py \
  model_path:=models/lane_model.pt
```

---

### Example 2: Quick Lane Following

```bash
# Single command to start lane following
ros2 launch lane_following_yolov11 lane_following_yolov11.launch.py \
  model_path:=models/lane_model.pt

# Monitor in another terminal
ros2 topic echo /lane_lateral_error
```

---

### Example 3: Map Cones While Racing

```bash
# Launch full system
ros2 launch lane_following_yolov11 full_system.launch.py \
  lane_model_path:=models/lane_model.pt \
  cone_model_path:=models/cone_model.pt

# In another terminal - visualize
rviz2

# Add displays:
# - MarkerArray: /cone_map/markers
# - Image: /lane_segmentation/debug_image
```

---

### Example 4: Navigate to Goal

```bash
# Create cone map first
cat > config/cone_map.json << EOF
[
  {"id": 0, "label": "Start_L", "x": 0.0, "y": -1.0, "z": 0.0},
  {"id": 1, "label": "Start_R", "x": 0.0, "y": 1.0, "z": 0.0},
  {"id": 2, "label": "Goal_L", "x": 2.5, "y": -0.5, "z": 0.0},
  {"id": 3, "label": "Goal_R", "x": 2.5, "y": 0.5, "z": 0.0}
]
EOF

# Launch navigation
ros2 launch lane_following_yolov11 autonomous_navigation.launch.py \
  cone_model_path:=models/cone_model.pt \
  cone_map_file:=config/cone_map.json \
  goal_x:=2.5 \
  goal_y:=0.0
```

---

## Visualization

### RViz Setup

```bash
# Launch RViz
rviz2

# Fixed Frame: map

# Add displays:
1. Grid (Reference: map)
2. TF (Show coordinate frames)
3. Image (/lane_segmentation/debug_image)
4. Image (/cone_detection/debug_image)
5. MarkerArray (/cone_map/markers)
6. MarkerArray (/reference_cone_map/markers)
7. Marker (/goal_marker)
8. PoseWithCovariance (/estimated_pose)
```

### Debug Images

**Lane Following:**
```bash
ros2 run rqt_image_view rqt_image_view /lane_segmentation/debug_image
```
- Shows detected lane boundaries
- Lateral error visualization
- Frame rate and detection info

**Cone Detection:**
```bash
ros2 run rqt_image_view rqt_image_view /cone_detection/debug_image
```
- Shows detected cones
- Bounding boxes and confidence
- Depth measurements

**Autonomous Navigation:**
```bash
ros2 run rqt_image_view rqt_image_view /cone_nav/debug_image
```
- Current position and goal
- Distance to goal
- Detected cones count
- Localization status

---

## Parameter Reference

### Runtime Parameter Changes

**View all parameters:**
```bash
ros2 param list
```

**Get parameter value:**
```bash
ros2 param get /lane_controller_node Kp
```

**Set parameter:**
```bash
ros2 param set /lane_controller_node Kp 0.01
```

### Common Adjustments

**PID Tuning:**
```bash
ros2 param set /lane_controller_node Kp 0.008
ros2 param set /lane_controller_node Kd 0.02
ros2 param set /lane_controller_node Ki 0.0002
```

**Speed Control:**
```bash
ros2 param set /lane_controller_node forward_speed 0.5
```

**Detection Thresholds:**
```bash
ros2 param set /yolov11_lane_segmentation_node confidence_threshold 0.6
ros2 param set /cone_localization_node confidence_threshold 0.7
```

**Control Enable/Disable:**
```bash
ros2 param set /lane_controller_node enable_control false
ros2 param set /lane_controller_node enable_control true
```

**Navigation:**
```bash
ros2 param set /autonomous_cone_navigation_node max_linear_velocity 0.3
ros2 param set /autonomous_cone_navigation_node cone_match_threshold 0.5
```

### Using rqt_reconfigure

```bash
ros2 run rqt_reconfigure rqt_reconfigure
```
- Visual interface for parameter tuning
- Real-time slider adjustments
- See changes immediately

---

## Troubleshooting

### No Camera Data

**Check topics:**
```bash
ros2 topic list | grep zed2
ros2 topic hz /zed2/zed_node/left/image_rect_color
```

**Solution:**
- Verify ZED2 is connected
- Launch ZED2 wrapper: `ros2 launch zed_wrapper zed2.launch.py`
- Check USB connection and permissions

---

### No Detections

**Check model:**
```bash
# Test model separately
yolo detect predict model=models/cone_model.pt source=test.jpg
```

**Lower threshold:**
```bash
ros2 param set /node_name confidence_threshold 0.3
```

**Check logs:**
```bash
ros2 node info /yolov11_lane_segmentation_node
```

---

### Robot Not Moving

**Check control enabled:**
```bash
ros2 param get /lane_controller_node enable_control
# Should return: true
```

**Check cmd_vel:**
```bash
ros2 topic echo /cmd_vel
# Should see non-zero values
```

**Enable control:**
```bash
ros2 param set /lane_controller_node enable_control true
```

---

### Poor Lane Following

**View debug image:**
```bash
ros2 run rqt_image_view rqt_image_view /lane_segmentation/debug_image
```

**Tune PID:**
```bash
ros2 run rqt_reconfigure rqt_reconfigure
# Adjust Kp, Ki, Kd while watching
```

**Reduce speed:**
```bash
ros2 param set /lane_controller_node forward_speed 0.5
```

---

### Low FPS

**Check GPU:**
```bash
nvidia-smi
```

**Enable half precision:**
```bash
ros2 param set /node_name half_precision true
```

**Reduce input size:**
```bash
ros2 param set /node_name imgsz 416
```

**Use CPU (slower):**
```bash
# Relaunch with device:=cpu
```

---

### Localization Not Working

**Check cone map loaded:**
Look for startup message:
```
Reference map: 6 cones loaded
  Cone_A: (1.00, -1.50, 0.00)
```

**Check TF tree:**
```bash
ros2 run tf2_tools view_frames
# Verify: map → odom → base_link → camera
```

**Check for matches:**
Should see messages:
```
Matched cone to Cone_A (dist=0.15m)
Localization #5: 3 cone matches
```

**Adjust threshold:**
```bash
ros2 param set /autonomous_cone_navigation_node cone_match_threshold 1.0
```

---

## Topic Reference

### Lane Following

| Topic | Type | Description |
|-------|------|-------------|
| `/lane_boundaries` | Float32MultiArray | [left_x, right_x, width] |
| `/lane_lateral_error` | Float32 | Error in pixels |
| `/cmd_vel` | Twist | Velocity commands |
| `/lane_segmentation/debug_image` | Image | Visualization |

### Cone Mapping

| Topic | Type | Description |
|-------|------|-------------|
| `/cone_map/markers` | MarkerArray | RViz markers |
| `/cone_map/poses` | PoseArray | Cone positions |
| `/cone_detection/debug_image` | Image | Visualization |

### Autonomous Navigation

| Topic | Type | Description |
|-------|------|-------------|
| `/estimated_pose` | PoseWithCovarianceStamped | Corrected pose |
| `/reference_cone_map/markers` | MarkerArray | Reference cones |
| `/goal_marker` | Marker | Goal position |
| `/cone_nav/debug_image` | Image | Visualization |

### Camera Inputs

| Topic | Type | Description |
|-------|------|-------------|
| `/zed2/zed_node/left/image_rect_color` | Image | RGB image |
| `/zed2/zed_node/left/camera_info` | CameraInfo | Intrinsics |
| `/zed2/zed_node/depth/depth_registered` | Image | Depth image |

---

## Best Practices

### 1. Testing New Models
- ✅ Test with static images first
- ✅ Use debug launch files (control disabled)
- ✅ View debug images to verify detections
- ✅ Start with low speeds

### 2. PID Tuning
- ✅ Start with conservative gains
- ✅ Use rqt_reconfigure for real-time tuning
- ✅ Test on straight sections first
- ✅ Save good values to config file

### 3. Cone Mapping
- ✅ Drive slowly for accurate positions
- ✅ View cones in RViz during mapping
- ✅ Make multiple passes for averaging
- ✅ Save map when satisfied

### 4. Autonomous Navigation
- ✅ Verify cone map in RViz first
- ✅ Test localization before navigation
- ✅ Start close to expected position
- ✅ Monitor pose estimate vs odometry

### 5. Safety
- ✅ Always test in safe open area
- ✅ Have emergency stop ready
- ✅ Monitor robot during operation
- ✅ Use debug mode for testing

---

## Performance Benchmarks

| Hardware | Model | Input Size | FPS |
|----------|-------|------------|-----|
| RTX 3080 | YOLOv11n | 640 | 25-30 |
| RTX 3060 | YOLOv11n | 640 | 20-25 |
| GTX 1660 | YOLOv11n | 640 | 15-20 |
| CPU (i7) | YOLOv11n | 640 | 3-5 |

---

## Summary

This package provides a complete autonomous racing system with:

✅ **Lane Following** - PID control with YOLOv11 segmentation
✅ **Cone Mapping** - Build persistent maps with 3D localization
✅ **Autonomous Navigation** - Goal-directed navigation with cone-based localization
✅ **Real-time Visualization** - RViz integration for all components
✅ **Comprehensive Logging** - Detailed performance and status information
✅ **Safety Features** - Watchdogs, error thresholds, graceful stops

**Three Main Use Cases:**
1. **Race with lane following**
2. **Build cone maps**
3. **Navigate to goals autonomously**

**Getting Started:**
1. Install dependencies
2. Build package
3. Add your trained models
4. Choose appropriate launch file
5. Monitor in RViz

For specific use cases, see the [Quick Start Examples](#quick-start-examples) section.

---

## Support

**Common Issues:**
- See [Troubleshooting](#troubleshooting) section
- Check topic rates: `ros2 topic hz <topic>`
- View logs: Terminal output
- Verify TF tree: `ros2 run tf2_tools view_frames`

**For Help:**
1. Check this README
2. Review launch file comments
3. Examine parameter configurations
4. Test components individually
5. Monitor debug visualizations

---

## License

MIT License

## Contributors

[Your Team/Name]

## Version

1.0.0