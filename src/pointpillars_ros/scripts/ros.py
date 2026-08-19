#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ROS 2 port of PointPillars (hardened / fail-safe version)

import os
import sys
import time
import threading
from pathlib import Path

import numpy as np
import scipy.linalg as linalg
import torch
from pyquaternion import Quaternion

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray

# Package source root
PKG_ROOT = os.environ.get(
    "POINTPILLARS_ROS_ROOT", "/home/pointpillar_ws/src/pointpillars_ros")
OPENPCDET_ROOT = os.path.join(PKG_ROOT, "OpenPCDet")

# Make OpenPCDet importable
sys.path.append(OPENPCDET_ROOT)
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


# =============================================================
# Settings Parameter
# =============================================================
# Topics
INPUT_TOPIC = '/pointcloud/vlp16'
OUTPUT_TOPIC = '/detections'

# QoS / inference loop
QOS_DEPTH = 10
QOS_RELIABILITY = QoSReliabilityPolicy.RELIABLE   # or BEST_EFFORT
INFERENCE_HZ = 10.0

# Model (overridable via the config_path / ckpt_path ROS parameters)
DEFAULT_CONFIG_PATH = os.path.join(
    OPENPCDET_ROOT, 'tools/cfgs/kitti_models/pointpillar.yaml')
DEFAULT_CKPT_PATH = os.path.join(
    OPENPCDET_ROOT, 'tools/models/pointpillar_7728.pth')

# Point cloud preprocessing filter
MAX_RANGE = 120.0
MIN_POINTS = 800
Z_MIN = -5.0
Z_MAX = 5.0
R_MIN = 1.0
INTENSITY_MAX = 9999.0

# Detection output filter
SCORE_THRESHOLD = 0.6
MAX_POSITION = 300.0
MIN_BOX_SIZE = 0.1
MAX_BOX_SIZE = 30.0
# =============================================================


# -----------------------------
# Dataset wrapper
# -----------------------------
class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names,
            training=training, root_path=root_path, logger=logger
        )


# -----------------------------
# Sanitizing helpers
# -----------------------------
def sanitize_points(np_p, has_intensity,
                    max_range=MAX_RANGE,
                    min_points=MIN_POINTS):
    """
    np_p: shape (N, 3 or 4) float32
    has_intensity: if True, column 3 is used as intensity; otherwise it is zero-filled
    - Applies finite / range / height filtering and returns an (N,4) OpenPCDet input
    - Returns None when too few valid points remain
    """
    if np_p is None or np_p.ndim != 2 or np_p.shape[1] < 3:
        return None

    np_p = np_p.astype(np.float32, copy=False)

    # Finite-value filter
    finite_mask = np.isfinite(np_p[:, 0]) & np.isfinite(np_p[:, 1]) & np.isfinite(np_p[:, 2])
    if has_intensity and np_p.shape[1] >= 4:
        finite_mask &= np.isfinite(np_p[:, 3])
    np_p = np_p[finite_mask]
    if np_p.shape[0] == 0:
        return None

    # Range and height limits
    r = np.linalg.norm(np_p[:, :3], axis=1)
    z_ok = (np_p[:, 2] > Z_MIN) & (np_p[:, 2] < Z_MAX)
    r_ok = (r > R_MIN) & (r < max_range)
    mask = z_ok & r_ok
    np_p = np_p[mask]
    if np_p.shape[0] < min_points:
        return None

    # Intensity clipping and normalization
    if has_intensity and np_p.shape[1] >= 4:
        i = np_p[:, 3].astype(np.float32)
        i = np.nan_to_num(i, nan=0.0, posinf=0.0, neginf=0.0)
        i = np.maximum(i, 0.0)
        i = np.clip(i, 0.0, INTENSITY_MAX)
        i = i / INTENSITY_MAX
    else:
        i = np.zeros((np_p.shape[0],), dtype=np.float32)

    points = np.stack((np_p[:, 0], np_p[:, 1], np_p[:, 2], i), axis=1)

    # Final finite-value check
    points = points[np.isfinite(points).all(axis=1)]
    return points if points.shape[0] >= min_points else None


def sanitize_boxes(boxes_lidar, scores, labels,
                   score_thr=SCORE_THRESHOLD,
                   max_pos=MAX_POSITION,
                   min_size=MIN_BOX_SIZE,
                   max_size=MAX_BOX_SIZE):
    """
    Filter the raw OpenPCDet output.
    """
    if boxes_lidar is None or boxes_lidar.size == 0:
        return None, None, None

    # Basic filter: score and finiteness
    valid = (scores >= score_thr) & np.isfinite(scores)
    valid &= np.isfinite(boxes_lidar).all(axis=1)
    if not np.any(valid):
        return None, None, None

    b = boxes_lidar[valid]
    s = scores[valid]
    l = labels[valid]

    # Position / dimension / heading ranges
    pos_ok = (np.abs(b[:, 0]) < max_pos) & (np.abs(b[:, 1]) < max_pos) & (np.abs(b[:, 2]) < max_pos)
    size_ok = (b[:, 3] > min_size) & (b[:, 4] > min_size) & (b[:, 5] > min_size) & \
              (b[:, 3] < max_size) & (b[:, 4] < max_size) & (b[:, 5] < max_size)
    heading_ok = np.isfinite(b[:, 6])

    keep = pos_ok & size_ok & heading_ok
    if not np.any(keep):
        return None, None, None

    return b[keep], s[keep], l[keep]


# -----------------------------
# SafeRate
# -----------------------------
class SafeRate:
    def __init__(self, node, hz):
        self.node = node
        self.hz = hz
        self.period = 1.0 / hz

    def sleep(self):
        time.sleep(self.period)


# -----------------------------
# Main node
# -----------------------------
class Pointpillars_ROS(Node):
    def __init__(self):
        super().__init__('pointpillars_ros_node')

        self.latest_msg = None
        self.lock = threading.Lock()

        config_path, ckpt_path = self.init_ros()
        self.init_pointpillars(config_path, ckpt_path)

    def init_ros(self):
        self.declare_parameter('config_path', DEFAULT_CONFIG_PATH)
        self.declare_parameter('ckpt_path', DEFAULT_CKPT_PATH)

        config_path = self.get_parameter('config_path').value
        ckpt_path = self.get_parameter('ckpt_path').value

        qos_profile = QoSProfile(
            reliability=QOS_RELIABILITY,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=QOS_DEPTH
        )

        self.sub_velo = self.create_subscription(
            PointCloud2,
            INPUT_TOPIC,
            self.lidar_callback,
            qos_profile
        )
        self.pub_bbox = self.create_publisher(BoundingBoxArray, OUTPUT_TOPIC, QOS_DEPTH)

        self.get_logger().info(f"[topic] sub: {INPUT_TOPIC} -> pub: {OUTPUT_TOPIC}")

        return config_path, ckpt_path

    def init_pointpillars(self, config_path, ckpt_path):
        logger = common_utils.create_logger()
        self.get_logger().info('----------------- ROS2 PointPillars -------------------------')

        # A model config pulls in its dataset config through _BASE_CONFIG_,
        # which OpenPCDet opens verbatim - so it only resolves when the
        # working directory is OpenPCDet/tools, the way upstream runs it.
        # config_path itself is absolute and unaffected by the chdir.
        prev_cwd = os.getcwd()
        os.chdir(os.path.join(OPENPCDET_ROOT, 'tools'))
        try:
            cfg_from_yaml_file(config_path, cfg)
        finally:
            os.chdir(prev_cwd)

        self.demo_dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            training=False,
            ext='.bin',
            logger=logger
        )
        self.model = build_network(
            model_cfg=cfg.MODEL,
            num_class=len(cfg.CLASS_NAMES),
            dataset=self.demo_dataset
        )
        self.model.load_params_from_file(filename=ckpt_path, logger=logger, to_cpu=True)
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

    def start_inference_thread(self):
        self._start_worker()
        threading.Thread(target=self._watchdog, daemon=True).start()

    def _start_worker(self):
        self._worker = threading.Thread(target=self.inference_loop, daemon=True)
        self._worker.start()

    def _watchdog(self):
        while rclpy.ok():
            if not (hasattr(self, "_worker") and self._worker.is_alive()):
                self.get_logger().warn("[watchdog] inference thread died. Restarting...")
                self._start_worker()
            time.sleep(1.0)

    def rotate_mat(self, axis, radian):
        axis = np.asarray(axis, dtype=np.float32)
        n = np.linalg.norm(axis)
        if n == 0.0 or radian == 0.0:
            return np.eye(3, dtype=np.float32)
        axis = axis / n
        return linalg.expm(np.cross(np.eye(3), axis * radian)).astype(np.float32)

    def lidar_callback(self, msg):
        with self.lock:
            self.latest_msg = msg

    def inference_loop(self):
        rate = SafeRate(self, INFERENCE_HZ)
        while rclpy.ok():
            msg = None
            with self.lock:
                if self.latest_msg is not None:
                    msg = self.latest_msg
                    self.latest_msg = None

            if msg is not None:
                try:
                    self.process_lidar(msg)
                except Exception as e:
                    self.get_logger().error(f"[inference_loop] error: {e}")
                    import traceback
                    self.get_logger().error(traceback.format_exc())
            rate.sleep()

    def process_lidar(self, msg):
        t_start = time.time()

        # 1) Read points
        field_names = [f.name for f in msg.fields]
        has_intensity = ('intensity' in field_names)

        try:
            if has_intensity:
                pcl_iter = pc2.read_points(msg, skip_nans=True, field_names=["x", "y", "z", "intensity"])
                num_fields = 4
            else:
                pcl_iter = pc2.read_points(msg, skip_nans=True, field_names=["x", "y", "z"])
                num_fields = 3

            pcl_list = list(pcl_iter)
            if len(pcl_list) > 0:
                # Convert the structured array into a plain array
                if has_intensity:
                    np_raw = np.array([[p[0], p[1], p[2], p[3]] for p in pcl_list], dtype=np.float32)
                else:
                    np_raw = np.array([[p[0], p[1], p[2]] for p in pcl_list], dtype=np.float32)
            else:
                np_raw = np.array([], dtype=np.float32).reshape(0, num_fields)
        except Exception as e:
            self.get_logger().warn(f"[read_points] error: {e}")
            return

        # 2) Preprocess
        points = sanitize_points(np_raw, has_intensity=has_intensity)
        if points is None:
            self.get_logger().warn("[sanitize_points] no valid points")
            # Publish an empty array
            empty = BoundingBoxArray()
            empty.header.frame_id = msg.header.frame_id
            empty.header.stamp = msg.header.stamp
            self.pub_bbox.publish(empty)
            return

        # 3) Apply rotation
        rand_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        yaw = 0.0
        R = self.rotate_mat(rand_axis, yaw)
        pts3 = points[:, :3] @ R.T
        points = np.concatenate([pts3, points[:, 3:4]], axis=1)

        # 4) Build the OpenPCDet input
        input_dict = {'points': points, 'frame_id': 0}
        data_dict = self.demo_dataset.prepare_data(data_dict=input_dict)
        if data_dict is None:
            self.get_logger().warn("[prepare_data] returned None")
            return

        data_dict = self.demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)

        # 5) Inference
        with torch.no_grad():
            pred_dicts, _ = self.model.forward(data_dict)

        boxes_lidar = pred_dicts[0]['pred_boxes'].detach().cpu().numpy()
        scores = pred_dicts[0]['pred_scores'].detach().cpu().numpy()
        labels = pred_dicts[0]['pred_labels'].detach().cpu().numpy()

        # 6) Sanitize the results
        boxes_lidar, scores, labels = sanitize_boxes(boxes_lidar, scores, labels)

        arr_bbox = BoundingBoxArray()
        arr_bbox.header.frame_id = msg.header.frame_id
        arr_bbox.header.stamp = msg.header.stamp

        if boxes_lidar is None:
            self.get_logger().info("[sanitize_boxes] no valid detections")
            self.pub_bbox.publish(arr_bbox)
            return

        # 7) Publish BoundingBox messages
        num_kept = 0
        for i in range(boxes_lidar.shape[0]):
            x, y, z, dx, dy, dz, heading = boxes_lidar[i]

            if not np.isfinite([x, y, z, dx, dy, dz, heading]).all():
                continue
            if dx <= 0 or dy <= 0 or dz <= 0:
                continue

            bbox = BoundingBox()
            bbox.header.frame_id = msg.header.frame_id
            bbox.header.stamp = msg.header.stamp

            bbox.pose.position.x = float(x)
            bbox.pose.position.y = float(y)
            bbox.pose.position.z = float(z)

            bbox.dimensions.x = float(dx)
            bbox.dimensions.y = float(dy)
            bbox.dimensions.z = float(dz)

            q = Quaternion(axis=(0, 0, 1), radians=float(heading)).normalised
            bbox.pose.orientation.x = float(q.x)
            bbox.pose.orientation.y = float(q.y)
            bbox.pose.orientation.z = float(q.z)
            bbox.pose.orientation.w = float(q.w)

            bbox.value = float(scores[i])
            bbox.label = str(int(labels[i]))

            arr_bbox.boxes.append(bbox)
            num_kept += 1

        t_elapsed = time.time() - t_start
        fps = 1.0 / t_elapsed if t_elapsed > 0 else 0
        self.get_logger().info(f"[publish] boxes: {num_kept} | time: {t_elapsed*1000:.1f}ms | FPS: {fps:.1f}")
        self.pub_bbox.publish(arr_bbox)


def main(args=None):
    rclpy.init(args=args)
    node = Pointpillars_ROS()
    node.start_inference_thread()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("Shutting down")


if __name__ == '__main__':
    main()
