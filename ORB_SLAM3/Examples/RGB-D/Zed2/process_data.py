import os
import shutil
import numpy as np

# --- Set your input directories here ---
input_color_dir = "color/"
input_depth_dir = "depth/"

# --- Set your output directories here ---
output_color_dir = "frames/"
output_depth_dir = "depths/"

def process_data():
    # Create output directories if they don't exist
    os.makedirs(output_color_dir, exist_ok=True)
    os.makedirs(output_depth_dir, exist_ok=True)

    if not os.path.exists(input_color_dir) or not os.path.exists(input_depth_dir):
        print(f"Error: Make sure {input_color_dir} and {input_depth_dir} exist in this folder.")
        return

    # Extract timestamps from filenames (assuming format: 1775166586100046058.png, which is nanoseconds)
    def extract_time(filename):
        # returns timestamp in seconds
        name = os.path.splitext(filename)[0]
        return float(name) / 1e9

    color_files = sorted([f for f in os.listdir(input_color_dir) if f.endswith(('.png', '.jpg'))])
    depth_files = sorted([f for f in os.listdir(input_depth_dir) if f.endswith('.png')])

    if not color_files or not depth_files:
        print("No images found in input directories.")
        return

    color_times = np.array([extract_time(f) for f in color_files])
    depth_times = np.array([extract_time(f) for f in depth_files])

    # Threshold for syncing (0.02 seconds = 20 millisec)
    time_threshold = 0.02

    matches = []
    
    # Simple nearest neighbor search for syncing
    for d_idx, d_time in enumerate(depth_times):
        # find closest color image
        c_idx = np.argmin(np.abs(color_times - d_time))
        time_diff = abs(color_times[c_idx] - d_time)
        
        if time_diff < time_threshold:
            matches.append((c_idx, d_idx, color_times[c_idx], d_time))

    num_frames = len(matches)
    
    if num_frames == 0:
        print("Could not synchronize any frames within the time threshold.")
        return

    print(f"Found {num_frames} synchronized pairs.")

    timestamp_file = "timestamps.txt"
    with open(timestamp_file, "w") as f_ts:
        for i, (c_idx, d_idx, c_time, d_time) in enumerate(matches):
            idx_str = f"{i:06d}"
            
            src_color = os.path.join(input_color_dir, color_files[c_idx])
            src_depth = os.path.join(input_depth_dir, depth_files[d_idx])
            
            ext_color = os.path.splitext(color_files[c_idx])[1]
            
            dst_color = os.path.join(output_color_dir, f"frame{idx_str}{ext_color}")
            dst_depth = os.path.join(output_depth_dir, f"depth{idx_str}.png")
            
            shutil.copy(src_color, dst_color)
            shutil.copy(src_depth, dst_depth)
            
            # Write timestamp to file (using original depth image filename as the timestamp)
            original_timestamp_str = os.path.splitext(depth_files[d_idx])[0]
            f_ts.write(f"{original_timestamp_str}\n")

    print(f"Processed and renamed {num_frames} frames successfully. Timestamps saved to {timestamp_file}.")

if __name__ == "__main__":
    process_data()