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
colcon build --packages-select motion_pid_controller
```

## Starting this script in ROS

```bash
ros2 launch motion_pid_controller motion_pid_controller.launch.py
```
