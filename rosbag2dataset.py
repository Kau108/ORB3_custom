"""
Extract data from a ROS2 bag file and save it in the cmu_custom dataset format
expected by SplaTAM.

Uses TF transforms from the bag instead of hardcoded values.
Handles odom frame resets and filters erroneous image frames.

Usage:
    python scripts/rosbag2dataset.py <path_to_rosbag> <sequence_name>

Example:
    python scripts/rosbag2dataset.py /path/to/rosbag2_543 conference_room

This will create:
    data/cmu_custom/conference_room/
        rgb/
            frame_000000_<sec>_<nanosec>.png
            ...
        depth/
            frame_000000_<sec>_<nanosec>.png
            ...
        imu/
            imu_data.csv
        rgb.txt
        depth.txt
        groundtruth.txt

ROS2 bag topics used:
    /zed/zed_node/rgb/color/rect/image          -> rgb/ images
    /zed/zed_node/depth/depth_registered         -> depth/ images (16-bit PNG, scale=1000)
    /zed/zed_node/imu/data                       -> imu/imu_data.csv
    /zed/zed_node/odom                           -> groundtruth.txt (TUM format)
    /tf, /tf_static                              -> frame transforms
    /zed/zed_node/rgb/color/rect/camera_info     -> printed to console
    /zed/zed_node/depth/depth_registered/camera_info -> printed to console

TF tree expected:
    map -> odom -> zed_camera_link -> zed_camera_center
                        -> zed_left_camera_frame -> zed_left_camera_frame_optical
                        -> zed_right_camera_frame -> zed_right_camera_frame_optical

Odom processing:
    - Reads map->odom TF and odom->camera_link from odom messages
    - Detects odom frame resets (large jumps) and applies corrections
    - Removes random jump-and-return glitches

Image filtering:
    - Removes frames with duplicate timestamps (all copies)
    - Enforces minimum time interval between consecutive frames
    - Reports large time gaps
"""

import argparse
import bisect
import os
import sys
from collections import Counter, defaultdict, deque

import numpy as np

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, Imu, CameraInfo
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge
import cv2


# ── Configuration ────────────────────────────────────────────────────────

# Topic names
RGB_TOPIC = "/zed/zed_node/rgb/color/rect/image"
DEPTH_TOPIC = "/zed/zed_node/depth/depth_registered"
IMU_TOPIC = "/zed/zed_node/imu/data"
ODOM_TOPIC = "/zed/zed_node/odom"
RGB_INFO_TOPIC = "/zed/zed_node/rgb/color/rect/camera_info"
DEPTH_INFO_TOPIC = "/zed/zed_node/depth/depth_registered/camera_info"
TF_TOPIC = "/tf"
TF_STATIC_TOPIC = "/tf_static"

# Depth scale: depth_meters * DEPTH_SCALE = 16-bit PNG value
DEPTH_SCALE = 1000.0

# Frame names (as they appear in the TF tree)
MAP_FRAME = "map"
ODOM_FRAME = "odom"
CAMERA_LINK_FRAME = "zed_camera_link"
RGB_OPTICAL_FRAME = "zed_left_camera_frame_optical"
IMU_FRAME = "zed_left_camera_frame"

# Filtering defaults (overridable via CLI args)
GAP_WARNING_THRESHOLD = 1.0  # warn if consecutive frames are > this apart (seconds)
ODOM_JUMP_THRESHOLD = 0.5  # meters — detect odom position jumps above this
ODOM_QUEUE_SIZE = 15        # max buffered entries before declaring odom reset
RANDOM_JUMP_RETURN_DIST = 0.3  # meters — consider "returned" if within this distance


# ── Math helpers ─────────────────────────────────────────────────────────

def quaternion_to_matrix(qx, qy, qz, qw):
    """Convert quaternion to 3x3 rotation matrix."""
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n < 1e-20:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (qy * qy + qz * qz), s * (qx * qy - qz * qw), s * (qx * qz + qy * qw)],
        [s * (qx * qy + qz * qw), 1 - s * (qx * qx + qz * qz), s * (qy * qz - qx * qw)],
        [s * (qx * qz - qy * qw), s * (qy * qz + qx * qw), 1 - s * (qx * qx + qy * qy)],
    ])


def matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion (qx, qy, qz, qw)."""
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return qx, qy, qz, qw


def make_T(R, t):
    """Compose a 4x4 homogeneous transform."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inv_T(T):
    """Invert a 4x4 homogeneous transform."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def stamp_to_ns(sec, nanosec):
    return sec * 1_000_000_000 + nanosec


def stamp_to_float(sec, nanosec):
    return sec + nanosec * 1e-9


def make_filename(frame_idx, stamp_sec, stamp_nanosec):
    return f"frame_{frame_idx:06d}_{stamp_sec}_{stamp_nanosec:09d}.png"


def make_timestamp(stamp_sec, stamp_nanosec):
    return f"{stamp_sec}.{stamp_nanosec:09d}"


# ── Simple TF buffer ────────────────────────────────────────────────────

class SimpleTFBuffer:
    """Minimal TF tree built from bag data. Supports static & dynamic transforms
    and chaining through the frame tree via BFS."""

    def __init__(self):
        self.static_tfs = {}                   # (parent, child) -> 4x4
        self.dynamic_tfs = defaultdict(list)   # (parent, child) -> [(time_ns, 4x4), ...]
        self._edges = set()

    def add_static(self, parent, child, T):
        self.static_tfs[(parent, child)] = T
        self._edges.add((parent, child))

    def add_dynamic(self, parent, child, time_ns, T):
        self.dynamic_tfs[(parent, child)].append((time_ns, T))
        self._edges.add((parent, child))

    def finalize(self):
        """Sort dynamic transforms by time for binary-search lookups."""
        for key in self.dynamic_tfs:
            self.dynamic_tfs[key].sort(key=lambda x: x[0])

    # -- direct (single-edge) lookup --

    def _get_direct(self, parent, child, time_ns=None):
        if (parent, child) in self.static_tfs:
            return self.static_tfs[(parent, child)]
        if (child, parent) in self.static_tfs:
            return inv_T(self.static_tfs[(child, parent)])
        if (parent, child) in self.dynamic_tfs:
            return self._nearest_dynamic(parent, child, time_ns)
        if (child, parent) in self.dynamic_tfs:
            return inv_T(self._nearest_dynamic(child, parent, time_ns))
        return None

    def _nearest_dynamic(self, parent, child, time_ns):
        entries = self.dynamic_tfs[(parent, child)]
        if not entries:
            return None
        if time_ns is None:
            return entries[-1][1]
        times = [e[0] for e in entries]
        idx = bisect.bisect_left(times, time_ns)
        if idx == 0:
            return entries[0][1]
        if idx >= len(entries):
            return entries[-1][1]
        if abs(times[idx] - time_ns) < abs(times[idx - 1] - time_ns):
            return entries[idx][1]
        return entries[idx - 1][1]

    # -- chained lookup --

    def lookup(self, parent, child, time_ns=None):
        """Return parent_T_child, chaining through intermediate frames."""
        direct = self._get_direct(parent, child, time_ns)
        if direct is not None:
            return direct

        # BFS to find a path from parent to child
        adj = defaultdict(set)
        for (p, c) in self._edges:
            adj[p].add(c)
            adj[c].add(p)

        visited = {parent}
        queue = deque([(parent, [parent])])
        path = None
        while queue:
            node, p = queue.popleft()
            if node == child:
                path = p
                break
            for nbr in adj[node]:
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, p + [nbr]))
        if path is None:
            raise ValueError(f"No TF path from {parent} to {child}")

        T_acc = np.eye(4)
        for i in range(len(path) - 1):
            T_step = self._get_direct(path[i], path[i + 1], time_ns)
            if T_step is None:
                raise ValueError(f"Missing TF edge {path[i]} -> {path[i + 1]}")
            T_acc = T_acc @ T_step
        return T_acc

    def has_dynamic(self, parent, child):
        return (bool(self.dynamic_tfs.get((parent, child)))
                or bool(self.dynamic_tfs.get((child, parent))))

    def print_tree(self):
        print("TF tree edges:")
        for (p, c) in sorted(self._edges):
            if (p, c) in self.static_tfs or (c, p) in self.static_tfs:
                print(f"  {p} -> {c}  [static]")
            else:
                n = len(self.dynamic_tfs.get((p, c), []))
                if n == 0:
                    n = len(self.dynamic_tfs.get((c, p), []))
                print(f"  {p} -> {c}  [dynamic, {n} entries]")


# ── Odometry processing ─────────────────────────────────────────────────

def process_odom_entries(raw_odom, tf_buffer):
    """Process raw odom using a queue-based approach.

    Normal operation: entries flow straight through to output.
    On jump detection: buffer into a queue (max ODOM_QUEUE_SIZE).
      - If the trajectory returns near the pre-jump position before the
        queue fills → random glitch, discard the buffered entries.
      - If the queue fills without returning → odom frame reset, apply a
        correction transform and flush all buffered entries to output.

    Parameters
    ----------
    raw_odom : list of (timestamp_str, time_ns, odom_T_camera_link 4x4)
    tf_buffer : SimpleTFBuffer (finalised)

    Returns
    -------
    list of (timestamp_str, map_T_optical 4x4)
    """
    if not raw_odom:
        return []

    has_map_odom = tf_buffer.has_dynamic(MAP_FRAME, ODOM_FRAME)
    if has_map_odom:
        print(f"  Using dynamic TF for {MAP_FRAME} -> {ODOM_FRAME}")
    else:
        print(f"  WARNING: No dynamic TF for {MAP_FRAME} -> {ODOM_FRAME}, "
              f"using identity")

    try:
        camera_link_T_optical = tf_buffer.lookup(
            CAMERA_LINK_FRAME, RGB_OPTICAL_FRAME)
        print(f"  TF {CAMERA_LINK_FRAME} -> {RGB_OPTICAL_FRAME} loaded from bag")
    except ValueError:
        print(f"  WARNING: No TF for {CAMERA_LINK_FRAME} -> "
              f"{RGB_OPTICAL_FRAME}, using identity")
        camera_link_T_optical = np.eye(4)

    # Pre-compute raw map_T_camera_link for every odom entry
    raw_map_T_cam = []
    for _tstamp, time_ns, odom_T_cam in raw_odom:
        if has_map_odom:
            map_T_odom = tf_buffer.lookup(MAP_FRAME, ODOM_FRAME, time_ns)
        else:
            map_T_odom = np.eye(4)
        raw_map_T_cam.append(map_T_odom @ odom_T_cam)

    # -- queue-based processing --
    result = []                     # committed (tstamp, map_T_optical)
    queue = deque()                 # buffered (raw_idx, corrected_T)
    accumulated_correction = np.eye(4)
    last_committed_T = None         # last committed map_T_cam
    removed_total = 0
    reset_count = 0

    def _commit(idx, map_T_cam):
        """Append an entry to the output."""
        nonlocal last_committed_T
        last_committed_T = map_T_cam
        result.append((raw_odom[idx][0], map_T_cam @ camera_link_T_optical))

    def _flush_queue_as_reset():
        """Treat the queued entries as an odom reset: recompute correction,
        re-correct every buffered entry, and commit them all."""
        nonlocal accumulated_correction, reset_count
        first_idx = queue[0][0]
        accumulated_correction = (
            last_committed_T @ inv_T(raw_map_T_cam[first_idx]))
        for qi, _ in queue:
            re_corrected = accumulated_correction @ raw_map_T_cam[qi]
            _commit(qi, re_corrected)
        print(f"    idx {first_idx}: odom reset, applied correction "
              f"({len(queue)} buffered frames flushed)")
        reset_count += 1
        queue.clear()

    for i in range(len(raw_map_T_cam)):
        corrected_T = accumulated_correction @ raw_map_T_cam[i]

        # --- normal mode (queue empty) ---
        if not queue:
            if last_committed_T is None:
                _commit(i, corrected_T)
                continue

            jump = np.linalg.norm(
                corrected_T[:3, 3] - last_committed_T[:3, 3])
            if jump <= ODOM_JUMP_THRESHOLD:
                _commit(i, corrected_T)
            else:
                # Jump detected — start buffering
                queue.append((i, corrected_T))
            continue

        # --- buffering mode (queue non-empty) ---
        queue.append((i, corrected_T))

        # Did we return to the pre-jump position?
        dist_to_last = np.linalg.norm(
            corrected_T[:3, 3] - last_committed_T[:3, 3])
        if dist_to_last < RANDOM_JUMP_RETURN_DIST:
            # Random glitch — discard buffered entries, keep the return
            glitch_count = len(queue) - 1
            print(f"    idx {queue[0][0]}: random glitch "
                  f"({glitch_count} frames discarded, "
                  f"returned at idx {i})")
            removed_total += glitch_count
            queue.clear()
            _commit(i, corrected_T)

        elif len(queue) >= ODOM_QUEUE_SIZE:
            # Queue full — treat as permanent odom reset
            _flush_queue_as_reset()

    # Flush anything left in the queue at end of data
    if queue:
        _flush_queue_as_reset()

    if removed_total > 0:
        print(f"  Removed {removed_total} odom entries (random jumps)")
    if reset_count > 0:
        print(f"  Corrected {reset_count} odom frame reset(s)")
    print(f"  Final odom entries: {len(result)}")
    return result


# ── Image timestamp filtering ───────────────────────────────────────────

def filter_image_timestamps(entries, label=""):
    """Filter image timestamp entries.

    Rules
    -----
    1. Remove ALL copies of any timestamp that appears more than once.
    2. Report gaps > GAP_WARNING_THRESHOLD between consecutive frames as
       potential errors (but keep the frames).

    Parameters
    ----------
    entries : list of (ts_float, raw_idx, stamp_sec, stamp_nanosec)

    Returns
    -------
    list of (ts_float, raw_idx, stamp_sec, stamp_nanosec) — kept entries
    """
    if not entries:
        return []

    # Step 1 — remove duplicate timestamps (all copies)
    ts_counts = Counter(e[0] for e in entries)
    duplicates = {ts for ts, c in ts_counts.items() if c > 1}
    if duplicates:
        dup_total = sum(ts_counts[ts] for ts in duplicates)
        print(f"  [{label}] Removing {dup_total} frames across "
              f"{len(duplicates)} duplicate timestamps")
        for ts in sorted(duplicates):
            print(f"    t={ts:.9f}  ×{ts_counts[ts]}")
        entries = [e for e in entries if e[0] not in duplicates]

    if not entries:
        print(f"  [{label}] No frames remaining after duplicate removal")
        return []

    # Step 2 — flag suspicious gaps between consecutive frames
    gap_count = 0
    for i in range(1, len(entries)):
        dt = entries[i][0] - entries[i - 1][0]
        if dt > GAP_WARNING_THRESHOLD:
            print(f"  [{label}] WARNING: gap of {dt:.2f}s between frames "
                  f"{i - 1} and {i} "
                  f"(t={entries[i - 1][0]:.3f} -> {entries[i][0]:.3f})")
            gap_count += 1
    if gap_count:
        print(f"  [{label}] {gap_count} gaps > {GAP_WARNING_THRESHOLD}s detected")

    print(f"  [{label}] Keeping {len(entries)} of "
          f"{len(ts_counts) + dup_total if duplicates else len(ts_counts)} "
          f"raw frames")
    return entries


# ── Bag I/O helpers ──────────────────────────────────────────────────────

def open_bag_reader(bag_path):
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader.open(storage_options, converter_options)
    return reader


def get_topic_type_map(bag_path):
    reader = open_bag_reader(bag_path)
    return {t.name: t.type for t in reader.get_all_topics_and_types()}


_TYPE_MAP = {
    "sensor_msgs/msg/Image": Image,
    "sensor_msgs/msg/Imu": Imu,
    "sensor_msgs/msg/CameraInfo": CameraInfo,
    "nav_msgs/msg/Odometry": Odometry,
    "tf2_msgs/msg/TFMessage": TFMessage,
}


def get_message_type(type_str):
    return _TYPE_MAP.get(type_str)


# ── Main extraction ─────────────────────────────────────────────────────

def extract_bag(bag_path, output_dir):
    bridge = CvBridge()

    rgb_dir = os.path.join(output_dir, "rgb")
    depth_dir = os.path.join(output_dir, "depth")
    imu_dir = os.path.join(output_dir, "imu")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(imu_dir, exist_ok=True)

    type_map = get_topic_type_map(bag_path)
    print("Topics found in bag:")
    for topic, ttype in sorted(type_map.items()):
        print(f"  {topic} -> {ttype}")
    print()

    # ────────────────────────────────────────────────────────────────────
    # Pass 1 — read timestamps, TF, odom, IMU  (skip heavy image decode)
    # ────────────────────────────────────────────────────────────────────
    print("Pass 1: reading metadata & timestamps ...")
    reader = open_bag_reader(bag_path)

    tf_buffer = SimpleTFBuffer()
    rgb_ts_list = []       # (ts_float, raw_idx, stamp_sec, stamp_nanosec)
    depth_ts_list = []
    imu_rows = []
    odom_raw = []          # (timestamp_str, time_ns, odom_T_camera_link 4x4)
    camera_info = None     # will store {width, height, fx, fy, cx, cy}
    rgb_count = 0
    depth_count = 0

    while reader.has_next():
        topic, data, timestamp_ns = reader.read_next()

        # ── TF static ──
        if topic == TF_STATIC_TOPIC:
            msg_type = get_message_type(type_map.get(topic))
            if msg_type is None:
                continue
            msg = deserialize_message(data, msg_type)
            for tf in msg.transforms:
                t = np.array([tf.transform.translation.x,
                              tf.transform.translation.y,
                              tf.transform.translation.z])
                R = quaternion_to_matrix(tf.transform.rotation.x,
                                         tf.transform.rotation.y,
                                         tf.transform.rotation.z,
                                         tf.transform.rotation.w)
                tf_buffer.add_static(tf.header.frame_id, tf.child_frame_id,
                                     make_T(R, t))

        # ── TF dynamic ──
        elif topic == TF_TOPIC:
            msg_type = get_message_type(type_map.get(topic))
            if msg_type is None:
                continue
            msg = deserialize_message(data, msg_type)
            for tf in msg.transforms:
                time_ns = stamp_to_ns(tf.header.stamp.sec,
                                      tf.header.stamp.nanosec)
                t = np.array([tf.transform.translation.x,
                              tf.transform.translation.y,
                              tf.transform.translation.z])
                R = quaternion_to_matrix(tf.transform.rotation.x,
                                         tf.transform.rotation.y,
                                         tf.transform.rotation.z,
                                         tf.transform.rotation.w)
                tf_buffer.add_dynamic(tf.header.frame_id, tf.child_frame_id,
                                      time_ns, make_T(R, t))

        # ── RGB timestamp only ──
        elif topic == RGB_TOPIC:
            msg_type = get_message_type(type_map.get(topic))
            if msg_type is None:
                continue
            msg = deserialize_message(data, msg_type)
            s, ns = msg.header.stamp.sec, msg.header.stamp.nanosec
            rgb_ts_list.append((stamp_to_float(s, ns), rgb_count, s, ns))
            if rgb_count == 0:
                print(f"  RGB frame_id: {msg.header.frame_id}")
            rgb_count += 1

        # ── Depth timestamp only ──
        elif topic == DEPTH_TOPIC:
            msg_type = get_message_type(type_map.get(topic))
            if msg_type is None:
                continue
            msg = deserialize_message(data, msg_type)
            s, ns = msg.header.stamp.sec, msg.header.stamp.nanosec
            depth_ts_list.append((stamp_to_float(s, ns), depth_count, s, ns))
            if depth_count == 0:
                print(f"  Depth frame_id: {msg.header.frame_id}")
            depth_count += 1

        # ── IMU ──
        elif topic == IMU_TOPIC:
            msg_type = get_message_type(type_map.get(topic))
            if msg_type is None:
                continue
            msg = deserialize_message(data, msg_type)
            imu_rows.append({
                "stamp_sec": msg.header.stamp.sec,
                "stamp_nanosec": msg.header.stamp.nanosec,
                "gyro": np.array([msg.angular_velocity.x,
                                  msg.angular_velocity.y,
                                  msg.angular_velocity.z]),
                "accel": np.array([msg.linear_acceleration.x,
                                   msg.linear_acceleration.y,
                                   msg.linear_acceleration.z]),
                "orient_q": (msg.orientation.x, msg.orientation.y,
                             msg.orientation.z, msg.orientation.w),
            })

        # ── Odom ──
        elif topic == ODOM_TOPIC:
            msg_type = get_message_type(type_map.get(topic))
            if msg_type is None:
                continue
            msg = deserialize_message(data, msg_type)
            s, ns = msg.header.stamp.sec, msg.header.stamp.nanosec
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            R = quaternion_to_matrix(ori.x, ori.y, ori.z, ori.w)
            t = np.array([pos.x, pos.y, pos.z])
            odom_raw.append((make_timestamp(s, ns),
                             stamp_to_ns(s, ns),
                             make_T(R, t)))

        # ── Camera info ──
        elif topic in (RGB_INFO_TOPIC, DEPTH_INFO_TOPIC) and camera_info is None:
            msg_type = get_message_type(type_map.get(topic))
            if msg_type is None:
                continue
            msg = deserialize_message(data, msg_type)
            print(f"Camera info from {topic}:")
            print(f"  frame_id: {msg.header.frame_id}")
            print(f"  Resolution: {msg.width}x{msg.height}")
            print(f"  fx={msg.k[0]}, fy={msg.k[4]}, cx={msg.k[2]}, cy={msg.k[5]}")
            print(f"  Distortion: {msg.distortion_model}, D={list(msg.d)}")
            print()
            if topic == RGB_INFO_TOPIC:
                camera_info = {
                    "width": msg.width,
                    "height": msg.height,
                    "fx": msg.k[0],
                    "fy": msg.k[4],
                    "cx": msg.k[2],
                    "cy": msg.k[5],
                }

    print(f"\nRaw counts: RGB={rgb_count}  Depth={depth_count}  "
          f"IMU={len(imu_rows)}  Odom={len(odom_raw)}")

    tf_buffer.finalize()
    tf_buffer.print_tree()
    print()

    # ── Process odometry ──
    print("Processing odometry ...")
    odom_filtered = process_odom_entries(odom_raw, tf_buffer)

    # ── Filter image timestamps ──
    print("\nFiltering RGB timestamps ...")
    rgb_kept = filter_image_timestamps(rgb_ts_list, label="RGB")
    rgb_keep_set = {e[1] for e in rgb_kept}  # set of raw_idx
    rgb_write_map = {}  # raw_idx -> (new_idx, stamp_sec, stamp_nanosec)
    for new_idx, (_, raw_idx, s, ns) in enumerate(rgb_kept):
        rgb_write_map[raw_idx] = (new_idx, s, ns)

    print("\nFiltering depth timestamps ...")
    depth_kept = filter_image_timestamps(depth_ts_list, label="Depth")
    depth_keep_set = {e[1] for e in depth_kept}
    depth_write_map = {}
    for new_idx, (_, raw_idx, s, ns) in enumerate(depth_kept):
        depth_write_map[raw_idx] = (new_idx, s, ns)

    # ────────────────────────────────────────────────────────────────────
    # Pass 2 — re-read bag, write kept images to disk
    # ────────────────────────────────────────────────────────────────────
    # Clear old image files so stale frames from a previous run don't linger
    for d in (rgb_dir, depth_dir):
        for old_file in os.listdir(d):
            os.remove(os.path.join(d, old_file))

    print("\nPass 2: writing kept images ...")
    reader2 = open_bag_reader(bag_path)

    rgb_entries = []   # (tstamp_str, relative_path)
    depth_entries = []
    rgb_counter = 0
    depth_counter = 0

    while reader2.has_next():
        topic, data, timestamp_ns = reader2.read_next()

        if topic == RGB_TOPIC:
            if rgb_counter in rgb_keep_set:
                new_idx, s, ns = rgb_write_map[rgb_counter]
                msg = deserialize_message(data, Image)
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                fname = make_filename(new_idx, s, ns)
                cv2.imwrite(os.path.join(rgb_dir, fname), cv_image)
                rgb_entries.append((make_timestamp(s, ns), f"rgb/{fname}"))
                if (new_idx + 1) % 200 == 0:
                    print(f"  Wrote {new_idx + 1} RGB frames ...")
            rgb_counter += 1

        elif topic == DEPTH_TOPIC:
            if depth_counter in depth_keep_set:
                new_idx, s, ns = depth_write_map[depth_counter]
                msg = deserialize_message(data, Image)
                cv_depth = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                if cv_depth.dtype in (np.float32, np.float64):
                    cv_depth = np.nan_to_num(cv_depth, nan=0.0,
                                             posinf=0.0, neginf=0.0)
                    cv_depth_16u = (cv_depth * DEPTH_SCALE).astype(np.uint16)
                elif cv_depth.dtype == np.uint16:
                    cv_depth_16u = cv_depth
                else:
                    cv_depth_16u = cv_depth.astype(np.uint16)
                fname = make_filename(new_idx, s, ns)
                cv2.imwrite(os.path.join(depth_dir, fname), cv_depth_16u)
                depth_entries.append((make_timestamp(s, ns), f"depth/{fname}"))
                if (new_idx + 1) % 200 == 0:
                    print(f"  Wrote {new_idx + 1} depth frames ...")
            depth_counter += 1

    print(f"  Wrote {len(rgb_entries)} RGB, {len(depth_entries)} depth frames")

    # ────────────────────────────────────────────────────────────────────
    # Write text output files
    # ────────────────────────────────────────────────────────────────────
    print("\nWriting output files ...")

    # rgb.txt
    rgb_txt = os.path.join(output_dir, "rgb.txt")
    with open(rgb_txt, "w") as f:
        f.write("# color images\n# timestamp filename\n")
        for tstamp, path in rgb_entries:
            f.write(f"{tstamp} {path}\n")
    print(f"  {rgb_txt}  ({len(rgb_entries)} entries)")

    # depth.txt
    depth_txt = os.path.join(output_dir, "depth.txt")
    with open(depth_txt, "w") as f:
        f.write("# depth images\n# timestamp filename\n")
        for tstamp, path in depth_entries:
            f.write(f"{tstamp} {path}\n")
    print(f"  {depth_txt}  ({len(depth_entries)} entries)")

    # groundtruth.txt  (TUM format: timestamp tx ty tz qx qy qz qw)
    if odom_filtered:
        gt_txt = os.path.join(output_dir, "groundtruth.txt")
        with open(gt_txt, "w") as f:
            f.write("# ground truth trajectory\n")
            f.write("# file: 'rosbag'\n")
            f.write("# timestamp tx ty tz qx qy qz qw\n")
            for tstamp, T in odom_filtered:
                t = T[:3, 3]
                qx, qy, qz, qw = matrix_to_quaternion(T[:3, :3])
                f.write(f"{tstamp} {t[0]} {t[1]} {t[2]} "
                        f"{qx} {qy} {qz} {qw}\n")
        print(f"  {gt_txt}  ({len(odom_filtered)} entries)")
    else:
        print("  WARNING: no odom data — groundtruth.txt not written")

    # IMU data
    if imu_rows:
        # Get IMU-to-optical rotation from TF
        try:
            imu_T_optical = tf_buffer.lookup(IMU_FRAME, RGB_OPTICAL_FRAME)
            R_imu_to_optical = imu_T_optical[:3, :3]
            print(f"  IMU transform {IMU_FRAME} -> {RGB_OPTICAL_FRAME} from TF")
        except ValueError:
            print(f"  WARNING: no TF for {IMU_FRAME} -> {RGB_OPTICAL_FRAME}, "
                  f"using identity for IMU")
            R_imu_to_optical = np.eye(3)

        # Build timestamp lists for nearest-frame association
        rgb_ts_floats = [e[0] for e in rgb_kept]
        depth_ts_floats = [e[0] for e in depth_kept]

        imu_csv = os.path.join(imu_dir, "imu_data.csv")
        header = ("timestamp_sec,timestamp_nanosec,"
                  "angular_velocity_x,angular_velocity_y,angular_velocity_z,"
                  "linear_acceleration_x,linear_acceleration_y,"
                  "linear_acceleration_z,"
                  "orientation_x,orientation_y,orientation_z,orientation_w,"
                  "color_frame,depth_frame")
        with open(imu_csv, "w") as f:
            f.write(header + "\n")
            for row in imu_rows:
                gyro_cam = R_imu_to_optical @ row["gyro"]
                accel_cam = R_imu_to_optical @ row["accel"]
                R_imu_w = quaternion_to_matrix(*row["orient_q"])
                R_cam_w = R_imu_w @ R_imu_to_optical
                oqx, oqy, oqz, oqw = matrix_to_quaternion(R_cam_w)

                imu_ts = stamp_to_float(row["stamp_sec"],
                                        row["stamp_nanosec"])

                # Nearest RGB frame
                rgb_frame = ""
                if rgb_ts_floats:
                    ri = bisect.bisect_left(rgb_ts_floats, imu_ts)
                    ri = min(ri, len(rgb_ts_floats) - 1)
                    if (ri > 0 and abs(rgb_ts_floats[ri - 1] - imu_ts)
                            < abs(rgb_ts_floats[ri] - imu_ts)):
                        ri -= 1
                    _, _, rs, rns = rgb_kept[ri]
                    rgb_frame = make_filename(ri, rs, rns)

                # Nearest depth frame
                depth_frame = ""
                if depth_ts_floats:
                    di = bisect.bisect_left(depth_ts_floats, imu_ts)
                    di = min(di, len(depth_ts_floats) - 1)
                    if (di > 0 and abs(depth_ts_floats[di - 1] - imu_ts)
                            < abs(depth_ts_floats[di] - imu_ts)):
                        di -= 1
                    _, _, ds, dns = depth_kept[di]
                    depth_frame = make_filename(di, ds, dns)

                f.write(
                    f"{row['stamp_sec']},{row['stamp_nanosec']},"
                    f"{gyro_cam[0]},{gyro_cam[1]},{gyro_cam[2]},"
                    f"{accel_cam[0]},{accel_cam[1]},{accel_cam[2]},"
                    f"{oqx},{oqy},{oqz},{oqw},"
                    f"{rgb_frame},{depth_frame}\n"
                )
        print(f"  {imu_csv}  ({len(imu_rows)} entries)")

        # accelerometer.txt
        accel_txt = os.path.join(output_dir, "accelerometer.txt")
        with open(accel_txt, "w") as f:
            f.write("# accelerometer data\n")
            f.write("# file: 'rosbag'\n")
            f.write("# timestamp ax ay az\n")
            for row in imu_rows:
                accel_cam = R_imu_to_optical @ row["accel"]
                tstamp = make_timestamp(row["stamp_sec"],
                                        row["stamp_nanosec"])
                f.write(f"{tstamp} {accel_cam[0]} "
                        f"{accel_cam[1]} {accel_cam[2]}\n")
        print(f"  {accel_txt}  ({len(imu_rows)} entries)")

    # Write config YAML
    if camera_info is not None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        splatam_root = os.path.dirname(script_dir)
        config_dir = os.path.join(splatam_root, "configs", "data", "cmu_custom")
        os.makedirs(config_dir, exist_ok=True)
        # sequence_name is the last component of output_dir
        seq_name = os.path.basename(output_dir)
        yaml_path = os.path.join(config_dir, f"{seq_name}.yaml")
        with open(yaml_path, "w") as f:
            f.write(f"dataset_name: 'tum'\n")
            f.write(f"camera_params:\n")
            f.write(f"  image_height: {camera_info['height']}\n")
            f.write(f"  image_width: {camera_info['width']}\n")
            f.write(f"  fx: {camera_info['fx']}\n")
            f.write(f"  fy: {camera_info['fy']}\n")
            f.write(f"  cx: {camera_info['cx']}\n")
            f.write(f"  cy: {camera_info['cy']}\n")
            f.write(f"  crop_edge: 0\n")
            f.write(f"  png_depth_scale: {DEPTH_SCALE}\n")
            f.write(f"max_depth: 2.5\n")
        print(f"  {yaml_path}")
    else:
        print("  WARNING: no camera info found — config YAML not written")

    # Write static TFs so dataset2rosbag.py can reconstruct the TF tree
    tf_static_path = os.path.join(output_dir, "tf_static.txt")
    with open(tf_static_path, "w") as f:
        f.write("# parent child tx ty tz qx qy qz qw\n")
        for (parent, child), T in tf_buffer.static_tfs.items():
            t = T[:3, 3]
            qx, qy, qz, qw = matrix_to_quaternion(T[:3, :3])
            f.write(f"{parent} {child} {t[0]} {t[1]} {t[2]} "
                    f"{qx} {qy} {qz} {qw}\n")
    print(f"  {tf_static_path}  ({len(tf_buffer.static_tfs)} transforms)")


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract data from a ROS2 bag into cmu_custom dataset "
                    "format for SplaTAM."
    )
    parser.add_argument("bag_path", help="Path to the ROS2 bag directory")
    parser.add_argument("sequence_name",
                        help="Name for the output sequence folder")
    parser.add_argument("--output_base", default=None,
                        help="Base output directory "
                             "(default: ./data/cmu_custom)")
    parser.add_argument("--gap_threshold", type=float, default=1.0,
                        help="Warn if consecutive frames are more than this "
                             "many seconds apart (default: 1.0)")
    parser.add_argument("--jump_threshold", type=float, default=0.5,
                        help="Odom jump detection threshold in meters "
                             "(default: 0.5)")

    args = parser.parse_args()

    global GAP_WARNING_THRESHOLD, ODOM_JUMP_THRESHOLD
    GAP_WARNING_THRESHOLD = args.gap_threshold
    ODOM_JUMP_THRESHOLD = args.jump_threshold

    if args.output_base is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        splatam_root = os.path.dirname(script_dir)
        args.output_base = os.path.join(splatam_root, "data", "cmu_custom")

    output_dir = os.path.join(args.output_base, args.sequence_name)

    if os.path.exists(output_dir):
        print(f"WARNING: Output directory already exists: {output_dir}")
        response = input("Overwrite? [y/N]: ").strip().lower()
        if response != "y":
            print("Aborted.")
            sys.exit(1)

    if not os.path.exists(args.bag_path):
        print(f"ERROR: Bag path does not exist: {args.bag_path}")
        sys.exit(1)

    print(f"Bag path:        {args.bag_path}")
    print(f"Output dir:      {output_dir}")
    print(f"Gap threshold:   {GAP_WARNING_THRESHOLD}s")
    print(f"Jump threshold:  {ODOM_JUMP_THRESHOLD}m")
    print()

    extract_bag(args.bag_path, output_dir)

    print(f"\nDone! Dataset saved to: {output_dir}")
    print(f"Use '{args.sequence_name}' in your SplaTAM config.")


if __name__ == "__main__":
    main()
