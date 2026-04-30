import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

def plot_trajectory(filename, gt_filename=None):
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return

    print(f"Loading trajectory from {filename}...")
    # TUM trajectory format: timestamp tx ty tz qx qy qz qw
    try:
        data = np.loadtxt(filename)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if len(data) == 0:
        print("Error: Empty file or invalid format.")
        return
        
    gt_data = None
    if gt_filename:
        if not os.path.exists(gt_filename):
            print(f"Warning: Ground truth file {gt_filename} not found. Please check the path.")
        else:
            print(f"Loading ground truth from {gt_filename}...")
            try:
                gt_data = np.loadtxt(gt_filename, comments='#')
            except Exception as e:
                print(f"Error loading ground truth: {e}")

    # Extract translations
    timestamps = data[:, 0]
    tx = data[:, 1]
    ty = data[:, 2]
    tz = data[:, 3]

    # Detect jumps or resets to the origin to split into multiple trajectories
    segments = []
    current_segment = [0]
    for i in range(1, len(data)):
        p_prev = np.array([tx[i-1], ty[i-1], tz[i-1]])
        p_curr = np.array([tx[i], ty[i], tz[i]])
        
        # Check if it resets to origin or jumps significantly
        dist_to_origin = np.linalg.norm(p_curr)
        dist_prev_to_origin = np.linalg.norm(p_prev)
        jump_dist = np.linalg.norm(p_curr - p_prev)
        
        # Start a new trajectory if it drops back near the origin
        # or if the change in position is extremely large (> 5.0m instantaneous jump)
        if (dist_to_origin < 0.1 and dist_prev_to_origin > 1.0) or (jump_dist > 5.0):
            segments.append(current_segment)
            current_segment = []
            
        current_segment.append(i)
    if current_segment:
        segments.append(current_segment)

    fig = plt.figure(figsize=(14, 6))

    # 1. 3D Trajectory Plot
    ax1 = fig.add_subplot(121, projection='3d')
    for idx, seg in enumerate(segments):
        ax1.plot(tx[seg], tz[seg], -ty[seg], marker='.', markersize=2, linestyle='-', label=f'Trajectory {idx+1}')

    if gt_data is not None and len(gt_data) > 0:
        from scipy.spatial.transform import Rotation as R
        
        # Raw GT coords
        gt_xyz_raw = gt_data[:, 1:4]
        gt_quat_raw = gt_data[:, 4:8] # qx, qy, qz, qw
        
        # Rotate GT about X-axis by 90 degrees to align with Camera frame
        from scipy.spatial.transform import Rotation as R
        M_base = R.from_euler('x', -90, degrees=True).as_matrix()

        # 1. Convert all GT poses into ORB-SLAM coordinate frame
        p_orb = gt_xyz_raw @ M_base.T
        R_gt = R.from_quat(gt_quat_raw).as_matrix() # shape (N, 3, 3)
        
        # R_orb = M_base @ R_gt @ M_base^T
        R_orb = np.einsum('ij,njk,kl->nil', M_base, R_gt, M_base.T)
        
        # 2. Make the trajectory relative to the FIRST pose
        p0_orb = p_orb[0]
        R0_orb_inv = R_orb[0].T
        
        # p_rel = R0_orb_inv @ (p_orb - p0_orb)
        gt_xyz_rel = np.einsum('ij,nj->ni', R0_orb_inv, p_orb - p0_orb)

        gt_tx = gt_xyz_rel[:, 0]
        gt_ty = -gt_xyz_rel[:, 2]
        gt_tz = gt_xyz_rel[:, 1]

        ax1.plot(gt_tx, gt_tz, -gt_ty, marker='', markersize=1, linestyle='--', color='k', label='Ground Truth')

    # Note: Plotted as tx, tz, -ty to match standard coordinate visualization better
    ax1.set_xlabel('Camera X (Right)')
    ax1.set_ylabel('Camera Z (Forward)')
    ax1.set_zlabel('Camera Y (Up)')
    ax1.set_title('3D Trajectory')
    ax1.legend()

    # 2. 2D Top-Down Plot (X vs Z)
    # In ORB-SLAM, Z is typically forward and X is right
    ax2 = fig.add_subplot(122)
    for idx, seg in enumerate(segments):
        ax2.plot(tx[seg], tz[seg], marker='.', markersize=2, linestyle='-', label=f'Trajectory {idx+1}')

    if gt_data is not None and len(gt_data) > 0:
        ax2.plot(gt_tx, gt_tz, marker='', markersize=1, linestyle='--', color='k', label='Ground Truth')

    ax2.set_xlabel('Camera X (Right)')
    ax2.set_ylabel('Camera Z (Forward)')
    ax2.set_title('Top-Down View (Camera Frame X vs Z)')
    ax2.grid(True)
    ax2.axis('equal') # Keep aspect ratio square
    ax2.legend()

    plt.tight_layout()
    output_filename = "trajectory_plot.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Plot successfully saved to {output_filename}")

    if gt_data is not None and len(gt_data) > 0:
        print("\nComputing Absolute Trajectory Error (ATE) for X and Z only...")
        # Since timestamps are not synced, compute error based on spatial closest point
        # Make sure we use the fully aligned relative gt_tx, gt_tz
        gt_pts_xz = np.vstack((gt_tx, gt_tz)).T
        
        errors = []
        for i in range(len(tx)):
            est_xz = np.array([tx[i], tz[i]])
            # Find the closest ground truth point in X-Z plane
            distances = np.linalg.norm(gt_pts_xz - est_xz, axis=1)
            min_dist = np.min(distances)
            errors.append(min_dist)
                
        if len(errors) > 0:
            rmse = np.sqrt(np.mean(np.array(errors)**2))
            print(f"Computed 2D ATE (RMSE) over {len(errors)} points using spatial nearest neighbor (X, Z only): {rmse:.4f} m")
        else:
            print("No points found to compute ATE.")

    # Display plot if a display manager is available
    try:
        plt.show()
    except Exception as e:
        pass

if __name__ == "__main__":
    # Default file if none provided
    traj_file = "KeyFrameTrajectory.txt" 
    gt_file = None
    
    if len(sys.argv) > 1:
        traj_file = sys.argv[1]
    if len(sys.argv) > 2:
        gt_file = sys.argv[2]
        
    plot_trajectory(traj_file, gt_file)
