import os
import csv
import cv2
from cv_bridge import CvBridge
import numpy as np  # <-- ADD THIS LINE
# ROS 2 specific imports
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# --- Configuration ---
# Point this to the DIRECTORY containing your .db3 and metadata.yaml files
BAG_PATH = 'zed_data_1' 

# Updated topics based on your ZED metadata
COLOR_TOPIC = '/zed/zed_node/rgb/color/rect/image'
DEPTH_TOPIC = '/zed/zed_node/depth/depth_registered'
ODOM_TOPIC = '/zed/zed_node/odom'
CAM_INFO_TOPIC = '/zed/zed_node/rgb/color/rect/camera_info'

COLOR_DIR = 'color'
DEPTH_DIR = 'depth'
ODOM_CSV = 'odom.csv'

def get_rosbag_options(path, serialization_format='cdr'):
    """Configure the storage and converter options for reading."""
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id='')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format)
    return storage_options, converter_options

def main():
    # 1. Create directories
    os.makedirs(COLOR_DIR, exist_ok=True)
    os.makedirs(DEPTH_DIR, exist_ok=True)
    
    bridge = CvBridge()
    printed_cam_info = False

    print(f"Opening ROS 2 bag directory: {BAG_PATH}")
    
    # 2. Set up the reader
    try:
        storage_options, converter_options = get_rosbag_options(BAG_PATH)
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"Failed to open bag: {e}")
        print("Make sure BAG_PATH is set to the directory containing the metadata.yaml file.")
        return

    # 3. Create a dictionary mapping topic names to their message types
    topic_types = reader.get_all_topics_and_types()
    type_map = {topic_info.name: topic_info.type for topic_info in topic_types}
    
    topics_of_interest = [COLOR_TOPIC, DEPTH_TOPIC, ODOM_TOPIC, CAM_INFO_TOPIC]

    print("Extracting data... This will process ~3300 images and ~3200 odom messages.")

    # 4. Open CSV and iterate through messages
    with open(ODOM_CSV, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Standard odometry headers
        csv_writer.writerow(['timestamp_ns', 'pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w'])

        while reader.has_next():
            (topic, data, timestamp) = reader.read_next()

            # Format the bag timestamp (nanoseconds) to a string like 'sec.nanoseconds'
            ts_sec = timestamp // 1_000_000_000
            ts_nsec = timestamp % 1_000_000_000
            ts_str = f"{ts_sec}.{ts_nsec:09d}"

            if topic in topics_of_interest:
                msg_type = get_message(type_map[topic])
                msg = deserialize_message(data, msg_type)

                # --- Handle Color Images ---
                if topic == COLOR_TOPIC:
                    try:
                        # ZED rect images are standard BGR/RGB
                        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                        filename = os.path.join(COLOR_DIR, f"{ts_str}.png")
                        cv2.imwrite(filename, cv_image)
                    except Exception as e:
                        print(f"Failed to save color image at {ts_str}: {e}")

                # --- Handle Depth Images ---
                elif topic == DEPTH_TOPIC:
                    try:
                        # 1. Get the 32-bit float image (distances in meters)
                        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                        
                        # 2. Convert meters to millimeters
                        cv_image_mm = cv_image * 1000.0
                        
                        # 3. Handle invalid depth points (turn NaNs and Infs into 0)
                        cv_image_mm = np.nan_to_num(cv_image_mm, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        # 4. Cast to 16-bit unsigned integer
                        cv_image_16u = cv_image_mm.astype(np.uint16)
                        
                        # 5. Save as PNG
                        filename = os.path.join(DEPTH_DIR, f"{ts_str}.png")
                        cv2.imwrite(filename, cv_image_16u)
                    except Exception as e:
                        print(f"Failed to save depth image at {ts_str}: {e}")
                # --- Handle Odometry ---
                elif topic == ODOM_TOPIC:
                    pos = msg.pose.pose.position
                    ori = msg.pose.pose.orientation
                    csv_writer.writerow([ts_str, pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w])

                # --- Handle Camera Info ---
                elif topic == CAM_INFO_TOPIC:
                    if not printed_cam_info:
                        print("\n--- RGB Camera Info ---")
                        print(f"Resolution: {msg.width}x{msg.height}")
                        print(f"Distortion Model: {msg.distortion_model}")
                        print(f"D (Distortion coefficients): {msg.d}")
                        print(f"K (Camera matrix): {msg.k}")
                        print(f"R (Rectification matrix): {msg.r}")
                        print(f"P (Projection matrix): {msg.p}")
                        print("-----------------------\n")
                        printed_cam_info = True

    print(f"Extraction complete! Images saved to '{COLOR_DIR}/' and '{DEPTH_DIR}/'. Odometry saved to '{ODOM_CSV}'.")

if __name__ == '__main__':
    main()