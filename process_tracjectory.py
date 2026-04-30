import os
import sys

def process_trajectory():
    timestamps_file = "timestamps.txt"
    trajectory_file = "CameraTrajectory1.txt"
    output_file = "Trajectory.txt"

    if not os.path.exists(timestamps_file):
        print(f"Error: {timestamps_file} not found.")
        return
    if not os.path.exists(trajectory_file):
        print(f"Error: {trajectory_file} not found.")
        return

    # Read timestamps
    with open(timestamps_file, 'r') as f:
        timestamps = [line.strip() for line in f if line.strip()]

    # Read trajectory poses
    with open(trajectory_file, 'r') as f:
        poses = [line.strip().split() for line in f if line.strip() and not line.startswith('#')]

    # Check for length mismatch
    if len(timestamps) != len(poses):
        print(f"Warning: Number of timestamps ({len(timestamps)}) does not match number of poses ({len(poses)}).")
        print("Will process up to the shortest length.")

    # Write out new trajectory
    out_lines = []
    for ts, pose in zip(timestamps, poses):
        # A valid pose line has a timestamp + 7 values (tx, ty, tz, qx, qy, qz, qw)
        if len(pose) >= 8:
            new_line = f"{ts} " + " ".join(pose[1:])
            out_lines.append(new_line)
        else:
            print(f"Warning: Skipping malformed pose line: {pose}")

    with open(output_file, 'w') as f:
        f.write("\n".join(out_lines) + "\n")

    print(f"Successfully processed and wrote {len(out_lines)} lines to {output_file}.")

if __name__ == "__main__":
    process_trajectory()
