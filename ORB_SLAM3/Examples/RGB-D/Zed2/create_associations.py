import os
import cv2

frames_dir = "frames/"
depths_dir = "depths/"
output_file = "associations.txt"

LAPLACIAN_THRESHOLD = 0.0 # Adjust this threshold based on your images

def create_associations():
    if not os.path.exists(frames_dir):
        print(f"Error: Directory '{frames_dir}' not found.")
        return

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg'))])
    
    valid_frames = 0
    skipped_frames = 0
    
    with open(output_file, "w") as f:
        # Use a ts counter since we are dropping some indices
        ts_counter = 0
        for i, fname in enumerate(frame_files):
            img_path = os.path.join(frames_dir, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                variance = cv2.Laplacian(img, cv2.CV_64F).var()
                # print(variancee)
                if variance < LAPLACIAN_THRESHOLD:
                    skipped_frames += 1
                    continue # Skip this blurry frame
            
            # Extract the numerical part from the frame name (e.g. "000000" from "frame000000.png")
            idx_str = "".join(filter(str.isdigit, fname))
            ext = os.path.splitext(fname)[1]
            
            rgb_path = f"{frames_dir}frame{idx_str}{ext}"
            depth_path = f"{depths_dir}depth{idx_str}.png"
            
            ts = str(ts_counter)
            line = f"{ts} {rgb_path} {ts} {depth_path}\n"
            f.write(line)
            ts_counter += 1
            valid_frames += 1

    print(f"Created {output_file} successfully.")
    print(f"Kept {valid_frames} frames, skipped {skipped_frames} blurry frames.")

if __name__ == "__main__":
    create_associations()