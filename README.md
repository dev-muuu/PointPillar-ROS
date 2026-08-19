# pointpillar_ws

ROS 2 node for the PointPillars 3D object detector.

- Subscribes: `/pointcloud/vlp16` (`sensor_msgs/PointCloud2`)
- Publishes: `/detections` (`jsk_recognition_msgs/BoundingBoxArray`)

## Build

```bash
cd /home/pointpillar_ws
colcon build --symlink-install
```

## Run

```bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/home/pointpillar_ws/fastdds.xml
export ROS_DOMAIN_ID=0

source install/setup.bash
ros2 launch pointpillars_ros pointpillars.launch.py
```

## Paths

If cloned elsewhere:

```bash
export POINTPILLARS_ROS_ROOT=/path/to/pointpillar_ws/src/pointpillars_ros
```

Rebuild the CUDA extensions if your PyTorch/CUDA version differs:

```bash
cd src/pointpillars_ros/OpenPCDet && python3 setup.py develop
```
