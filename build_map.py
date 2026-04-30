import os
import sys
import numpy as np
import open3d as o3d

# -------------------------------------------------------------
# 3D DENSE RECONSTRUCTION LOGIC
# -------------------------------------------------------------

# 1. Camera Parameters (Update if different in real2.yaml)
fx, fy = 525.0, 525.0 # Example TUM defaults, update to match real2.yaml
cx, cy = 319.5, 239.5
width, height = 640, 480
intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# 2. Paths
base_dir = "Examples/RGB-D/real2"
association_file = os.path.join(base_dir, "association.txt")
trajectory_file = "CameraTrajectory.txt" # Ensure you've run ORB-SLAM3 to generated this

if not os.path.exists(association_file) or not os.path.exists(trajectory_file):
    print("Files not found. Make sure association.txt and CameraTrajectory.txt exist.")
    sys.exit()

print("Mapping started. Reading trajectory...")

# Associate timestamps to files
assoc_map = {}
with open(association_file, 'r') as f:
    for line in f:
        if line.startswith('#'): continue
        parts = line.strip().split()
        if len(parts) >= 4:
            t = float(parts[0])
            rgb_f = parts[1]
            depth_f = parts[3]
            assoc_map[t] = (rgb_f, depth_f)

# Sort to allow fast searching
assoc_keys = np.array(sorted(list(assoc_map.keys())))

global_pcd = o3d.geometry.PointCloud()

with open(trajectory_file, 'r') as f:
    lines = f.readlines()

count = 0
for line in lines:
    data = line.strip().split()
    if len(data) < 8: continue
    
    t_traj = float(data[0])
    
    # ORB-SLAM3 trajectory timestamps strongly align with association timestamps. Match them:
    idx = np.argmin(np.abs(assoc_keys - t_traj))
    match_time = assoc_keys[idx]
    
    if np.abs(match_time - t_traj) > 0.01:
        continue # If time gap is large, ignore
        
    color_rel, depth_rel = assoc_map[match_time]
    
    depth_path = os.path.join(base_dir, depth_rel)
    color_path = os.path.join(base_dir, color_rel)
    
    if not os.path.exists(depth_path) or not os.path.exists(color_path):
        continue 
        
    # Load images
    depth_img = o3d.io.read_image(depth_path)
    color_img = o3d.io.read_image(color_path)
    
    # Create RGBD Image (TUM scale is typically 5000.0, intel/realsense 1000.0)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_img, depth_img, 
        depth_scale=5000.0,  # Try changing back to 1000.0 if the map looks completely broken
        depth_trunc=4.0,     # Ignore points further than 4 meters to keep it clean
        convert_rgb_to_intensity=False
    )
    
    # Project to 3D
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    
    # Calculate Pose Matrix from Quaternion
    tx, ty, tz = float(data[1]), float(data[2]), float(data[3])
    qx, qy, qz, qw = float(data[4]), float(data[5]), float(data[6]), float(data[7])
    
    pose = np.eye(4)
    rotation = o3d.geometry.get_rotation_matrix_from_quaternion(np.array([qw, qx, qy, qz]))
    pose[:3, :3] = rotation
    pose[:3, 3] = [tx, ty, tz]
    
    # Transform and merge
    pcd.transform(pose)
    global_pcd += pcd
    count += 1
    
    if count % 10 == 0:
        print(f"Fused {count} frames...")

print(f"Total points: {len(global_pcd.points)}. Cleaning up map...")

# VOXEL FUSION: This merges all overlapping points in a 3cm grid and averages colors.
voxel_map = global_pcd.voxel_down_sample(voxel_size=0.03)

# Save result
output_file = "reconstruction_map.ply"
o3d.io.write_point_cloud(output_file, voxel_map)
print(f"SUCCESS! Final 3D map saved as {output_file}")

# Try to visualize the map (works if GUI/X11 is configured correctly)
print("Visualizing map...")
o3d.visualization.draw_geometries([voxel_map])
