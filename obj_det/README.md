# Object Detection Node

## Pre-installation

Confirm that these settings must be set correctly on `common.yaml` (for a full example of the config file, see [here](https://github.com/stereolabs/zed-ros-wrapper/blob/master/zed_wrapper/params/common.yaml)):

```yaml
object_detection:
    od_enabled:                         true                    # True to enable Object Detection
    model:                              'MULTI_CLASS_BOX_FAST'  # Only use the FAST model, anything else is too taxing for the Jetson
    max_range:                          5.                      # Maximum detection range. Lower the detection range as our area of ops is 2.5m
    allow_reduced_precision_inference:  true                    # Allow inference to run at a lower precision to improve runtime and memory usage
    prediction_timeout:                 0.5                     #  During this time [sec], the object will have OK state even if it is not detected. Set this parameter to 0 to disable SDK predictions            
    object_tracking_enabled:            true                    # This MUST be enabled to ensure ZED SDK retains custody of the object 
                                                                # even when it goes out of frame
    mc_people:                          true                    # Enable for our project
    mc_vehicle:                         true                    # Enable for our project
    mc_bag:                             false
    mc_animal:                          false
    mc_electronics:                     false
    mc_fruit_vegetable:                 false
    mc_sport:                           false
```

## Installation

Place the folder inside your ROS workspace's src folder, and then build it with 

```bash
cd ~/ros2_ws
colcon build --packages-select obj_det --symlink-install
```

## Running

1. In the first terminal, run `ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2` to start the camera
2. In the second terminal, run `remote_control` to enable the car's remote control system. Press the reset button on the Arduino once
3. In the third terminal, run the car control script `ros2 launch car_control car_control.launch.py`
4. Finally, launch our custom script `ros2 run obj_det obj_visualizer`
5. Run any autonomous mode code if necessary, at this step
6. Place the car in autonomous mode by pressing the O button on the controller

## Data extraction

The final CSV file containing the list of detected objects is placed in `~/detected_objects.csv`. 
