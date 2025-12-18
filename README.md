# MR-Group2

> Mobile Robotics code by Group 2, class of 2025.

## Training the path segmentation model

Refer to [this README file](path_segmentation_train/README.md) for details on how to train the path and cone segmentation model. 

## Running the full stack (startup order)

> For more detailed information, refer to the READMEs in each of the packages

Start the system components in separate terminals (source your ROS workspace first) in the following order. Each command is a single line you can copy and paste:

0. You must run these commands to prep the Jetson module for high performance (set fixed clock speed for CPU and GPU to ensure consistent performance):

```bash
jetson_clocks
```

1. Start the camera script:

```bash
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2
```

2. Start the remote control interface (manual teleop):

```bash
remote_control
```

3. Start the car control script (vehicle actuator bridge):

```bash
ros2 launch car_control car_control.launch.py
```

4. Start the YOLO inference node:

```bash
ros2 run yolo_inference inference
```

5. Start the object detection visualizer:

```bash
ros2 run obj_det obj_visualizer
```

6. Launch the motion PID controller and control stack:

```bash
ros2 launch motion_pid_controller motion_pid_controller.launch.py
```

> **Notes**:
> - Run each command in its own terminal so you can monitor logs and stop individual components.
> - Make sure your ROS environment is sourced correctly (e.g., `source install/setup.bash` or your workspace setup file) before running `ros2` commands.
