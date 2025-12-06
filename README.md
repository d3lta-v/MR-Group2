# MR-Group2

> Mobile Robotics code by Group 2, class of 2025.

## Directories

This repo contains ROS packages that can be installed with the generic `colcon build  --packages-select <package name>` once the folder is placed inside the `src` folder of your ROS 2 workspace.

- `path_segmentation_train/` contains training of the Detectron 2 based MaskRCNN model for instance segmentation.

## Recording real track data

We can use the vehicle's manual control mode and record all of the imagery and odometry data into a `rosbag` with the following command:

```bash
ros2 bag record -o project /cmd_vel \
/zed/zed_node/odom \
/zed/zed_node/rgb/image_rect_color
```

## Running the image exporter tool

```
ros2 run zed_image_exporter export_images
```

## Training the path segmentation model

