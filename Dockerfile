# Use Ubuntu 20.04 as the base for maximum ORB-SLAM3 compatibility
FROM ubuntu:20.04

# Prevent interactive prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive

# Install all necessary system dependencies
RUN apt-get update && apt-get install -y \
    build-essential cmake git pkg-config \
    libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev \
    libglew-dev \
    libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev \
    libeigen3-dev \
    libopencv-dev python3-opencv \
    libboost-serialization-dev \
    libssl-dev \
    wget unzip \
    && rm -rf /var/lib/apt/lists/*

# Build and install Pangolin (Locked to stable v0.6 to prevent -Werror crashes)
WORKDIR /root
RUN git clone https://github.com/stevenlovegrove/Pangolin.git && \
    cd Pangolin && \
    git checkout v0.6 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# Clone, patch, and build ORB-SLAM3
WORKDIR /root
RUN git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git && \
    cd ORB_SLAM3 && \
    # Tell CMake to accept Ubuntu 20.04's default OpenCV 4.2.0
    find . -name "CMakeLists.txt" -exec sed -i 's/OpenCV 4.4/OpenCV 4.2/g' {} + && \
    # Fix the Out-Of-Memory crash by limiting make to 2 jobs (Extremely safe for RAM)
    sed -i 's/make -j/make -j2/g' build.sh && \
    chmod +x build.sh && \
    ./build.sh

# Set the working directory to the ORB-SLAM3 folder when the container starts
WORKDIR /root/ORB_SLAM3
CMD ["/bin/bash"]