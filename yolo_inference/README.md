# Imagery inference node

## Theory of operation

1. The imagery is streamed from the left camera of the ZED2 camera via the topic `/zed/zed_node/rgb/image_rect_color`
2. The YOLOv11 model segments it into 2 lanes with polygon points as outputs
3. The centre deviation and yaw from track are calculated and filtered with a median filter to remove outlier 
4. The filtered values are then published to `/cte` and `/angle_error`.


## Installing

This package requires you to install Ultralytics, a package for training and deploying YOLO models. As we are using an Nvidia Jetson system that is running JetPack 5, please follow the [instructions provided by Ultralytics](https://docs.ultralytics.com/guides/nvidia-jetson/#run-on-jetpack-512) strictly to install the correct package versions so that the inference code works correctly.

This package uses an optimised copy of our current weights called `best.engine`, which has been compiled from Pytorch format to TensorRT to fully take advantage of the Jetson's GPU. If you have issues with the TensorRT version, feel free to edit the code to revert it back to `best.pt` for the unoptimised Pytorch version.

Place the folder inside your ROS workspace's src folder, and then build it with:

```bash
cd ~/ros2_ws
colcon build --packages-select yolo_inference --symlink-install
```

## Running

```bash
ros2 run yolo_inference inference
```