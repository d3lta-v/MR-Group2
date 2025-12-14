# Motion PID Controller Unit

Take note of the following vehicle characteristics: 

Angular velocity is in radians

1. z = 0.0 (straight)
2. -0.2 <= Angular z < 0.0 (right)
3. 0.0 < Angular z < 0.2 (left)

Linear Velocity is in an arbitrary unit

1. Linear x = 0.0 (stop, throttle: 100)
2. 0.1 < Linear x <= 0.6 (slow, throttle: 104)
3. 0.6 < Linear x <= 1.0 (fast, throttle: 105)

## Installation

Move this folder to your ROS workspace/src, and then run:

```bash
colcon build --packages-select motion_pid_controller --symlink-install
```

## Configuring the parameters

Please open `motion_pid_controller.launch.py` to tune the PID controller unit. The default values are:

```python
"forward_speed": 1.0,       # This is capped by the max speed of the robot which is 1
"Kp" : 0.2, #0.2            # Rotational P gain
"Kd" : 0.09, #0.12975       # Rotational D gain
"Ki" : 0.00001, #0.00001    # Rotational I gain
"Kp_angle" : 2.0,           # Angular P gain
"target_xpos" : 14.0,       # Target x position (in meters) to reach and stop
"LKp" : 0.95, #0.95         # Linear P gain
"LKd" : 0.0,                # Linear D gain
"LKi" : 0.0,                # Linear I gain
"use_pose" : False,         # Use pose instead of odom topic for position feedback
"use_ext_data" : False      # Enable use of external CTE and angular error data, via /cte and /angle_error topics. Should be enabled when using yolo_inference for lane following
```

## Starting this script in ROS

```bash
ros2 launch motion_pid_controller motion_pid_controller.launch.py
```
