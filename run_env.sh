#!/bin/bash

# 1. Define your host workspace directory to be the current directory
WORKSPACE_DIR="$(pwd)"

# 2. Ensure the workspace directory exists to prevent mount errors
if [ ! -d "$WORKSPACE_DIR" ]; then
    echo "Creating workspace directory at $WORKSPACE_DIR..."
    mkdir -p "$WORKSPACE_DIR"
fi

# 3. Check if ORB_SLAM3 exists locally; if not, extract it from the image!
if [ ! -d "$WORKSPACE_DIR/ORB_SLAM3" ]; then
    echo "Local ORB_SLAM3 folder not found. Extracting from container..."
    sudo docker run --rm -v "$WORKSPACE_DIR:/workspace" orb3_env:latest cp -r /root/ORB_SLAM3 /workspace/
    sudo chown -R $USER:$USER "$WORKSPACE_DIR/ORB_SLAM3"
    echo "Extraction complete! You can now edit the files in VS Code."
fi

# 4. Allow local Docker to access your X11 display for the GUI
echo "Unlocking X11 display for Docker..."
xhost +local:docker

# 5. Launch the container
echo "Launching ORB-SLAM3 Environment..."
echo "Mounting $WORKSPACE_DIR to /workspace inside the container."

# Execute Docker
sudo docker run -it --rm \
    --net=host \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="$WORKSPACE_DIR:/workspace:rw" \
    --volume="$WORKSPACE_DIR/ORB_SLAM3:/root/ORB_SLAM3:rw" \
    orb3_env:latest
