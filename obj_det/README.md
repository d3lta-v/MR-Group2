# How to run

## Installation

Place the folder inside your ROS workspace's src folder, and then build it with 

```
cd ~/ros2_ws
colcon build --packages-select obj_det
```

## Running

1. In the first terminal, run `ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2` to start the camera
2. In the second terminal, run `remote_control` to enable the car's remote control system. Press the reset button on the Arduino once
3. In the third terminal, run the car control script `ros2 launch car_control car_control.launch.py`
4. Finally, launch our custom script `ros2 run obj_det obj_visualizer`
5. Place the car in autonomous mode by pressing the O button on the controller
