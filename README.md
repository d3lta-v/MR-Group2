# MR-Group2

> Mobile Robotics code by Group 2, class of 2025.

## Overall architecture

This codebase is designed to autonomously navigate on a track with a pre-trained YOLOv11 segmentation model to identify lanes. It additionally uses the ZED2 camera's depth sensing information to avoid obstacles in its path. 

Architecturally speaking, it needs to have the following key pieces (ordered in NGC, navigation, guidance, control):

- Navigation
    - **Image** to **Lane Segmentation** unit: Processes images from the camera with a YOLOv11n-seg model to identify (segment) lanes, which outputs image masks. This unit then applies a bird's eye transform to these image masks to turn it into a bird's eye view of the track ahead, and outputs this image.
        - Outputs: Image of the bird's eye transformed lane
    - **Point Cloud** to **LaserScan** unit: Ingests the point cloud information from the ZED2 camera, and turns it into a virtual laser scan. Available from [zed_depth_to_laserscan](https://github.com/stereolabs/zed-ros2-examples/tree/master/examples/zed_depth_to_laserscan) so we don't have to develop our own.
        - Outputs: LaserScan message from the pseudo-LIDAR
    - ZED2 **Odometry**: the ZED2 camera provides its own odometry (Cartesian position X, Y, Z, and orientation quaternion), which is within the odometry frame (i.e. no loop closure is done, and is therefore susceptible to drift). This is somewhat acceptable for our use case in mapping the location of the objects.
        - Outputs: ZED2 odometry data
    - ZED2 **Pose** information: the ZED2 camera provides loop-closed pose information (similar to odometry, but with no drift thanks to loop closure). This uses ZED2's own SLAM algorithms without involving an external SLAM software like RTabMap.
        - Outputs: ZED2 pose data
- Guidance
    - NOTE: Guidance data for Racetrack Challenge A and B is trivial - just go forward! Both target destinations only involve an increased x-coordinate with no need to turn to achieve a specific y-coordinate. Utilising only odometry or pose, it is fairly trivial to command the robot to simply "go forward" (increase in x-axis) and adjust the PID controller based on the y-coordinate (as the deviation function). This however needs to be combined with a control function to account for braking characteristics - we can't just shut the motor off and expect it to coast to a specific x-coordinate.
    - NOTE: However, Guidance for AutoLap challenge is more challenging as we need to do obstacle avoidance on the track itself. Hence, guidance data must not only come from the CTE and relative angle to the track (`cte` and `omega_rel`), but also from the pseudo-LaserScan unit from the datasource, providing what is essentially a "repulsion vector" of where the object is. 
    - NOTE: Furthermore, Guidance for AutoLap should also include the distance to target `target_dist` once it passes over a specific odometry milestone. This will give the vehicle sufficient data to compute the braking distance. 
    - **DirectionalGuidancePackage**: Computes the bird's eye view image of the lane to CTE (centre track error) and relative angle unit (`cte` and `omega_rel`), and feeds it to the motion PID controller. This is otherwise known as the lane pose estimator. This node is also responsible for computing the addition or subtraction of the CTE such that the vehicle can avoid obstacles, which involves 2D SLAM and keeping a map of the obstacles so that the obstacles in the map "repulse" the CTE, causing the CTE to change depending on where the obstacle is relative to the vehicle.
        - Inputs: Bird's eye transformed lane image, LaserScan message from pseudo-LIDAR
        - Outputs: `cte` and `omega_rel` CTE and relative angle
    - **LinearGuidancePackage**: Computes the distance to the target once it passes the terminal guidance milestone (which is a hardcoded odometry distance), and feeds it to the motion PID controller as the target distance `target_dist`.
        - Inputs: ZED2 odometry data
        - Outputs: Distance to target `target_dist`, which is set to -1 when target ranging is disabled
- Control
    - **Motion PID controller**: The controller has to take in 3 parameters: CTE, relative angle to lane and distance to target (`cte`, `omega_rel`, `target_dist`) and compute the necessary steering commands, linear velocity and angular velocity. Linear velocity should also be scaled relative to target position (in other words, it has to over damp the input to output transfer function to prevent overshoot). Target distance should be `-1` for most of the trip to let the vehicle maintain its speed until the vehicle approaches terminal guidance mode (i.e. it has completed almost the entire lap and is ready to stop at its starting point as per odometry readings). 
        - Inputs: `cte` and `omega_rel` CTE and relative angle, distance to target `target_dist`
        - Outputs: Directly outputs to the topic `/cmd_vel` the velocity and angular velocity of the vehicle

## Directories

This repo contains ROS packages that can be installed with the generic `colcon build  --packages-select <package name>` once the folder is placed inside the `src` folder of your ROS 2 workspace.

- `zed_image_exporter` exports images captured by a ZED2 camera and then recorded onto a rosbag file.
- `path_segmentation_train/` contains training of the YOLOv11n-seg model for instance segmentation and the trained weights.
- `path_segmentation` contains the ROS package that publishes CTE and relative angle to track, by running path inference. 

## Operations

### Recording real track data using 

We can use the vehicle's manual control mode and record all of the imagery and odometry data into a `rosbag` with the following command:

```bash
ros2 bag record -o project /cmd_vel \
/zed/zed_node/odom \
/zed/zed_node/rgb/image_rect_color
```

### Running the image exporter tool

```
ros2 run zed_image_exporter export_images
```

### Training the path segmentation model

Refer to [this README file](path_segmentation_train/README.md) for details on how to train the path and cone segmentation model. 


