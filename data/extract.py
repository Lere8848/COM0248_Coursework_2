import pyrealsense2 as rs
import numpy as np
import cv2
import os

# Set paths
bag_file = "data/realsense/20250328_105126.bag"
output_dir = "data/realsense/20250328_105126/"

os.makedirs(output_dir, exist_ok=True)

rgb_dir = os.path.join(output_dir, "image")
depth_vis_dir = os.path.join(output_dir, "depth_vis")
depth_raw_dir = os.path.join(output_dir, "depthTSDF")

# Create subdirectories
os.makedirs(rgb_dir, exist_ok=True)
os.makedirs(depth_vis_dir, exist_ok=True)
os.makedirs(depth_raw_dir, exist_ok=True)

# Configure pipeline to read from .bag file
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(bag_file, repeat_playback=False)
config.enable_stream(rs.stream.color)
config.enable_stream(rs.stream.depth)

# Start pipeline
pipeline.start(config)
align = rs.align(rs.stream.color)
frame_id = 0

print("Starting frame extraction...")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Save color image
        cv2.imwrite(f"{rgb_dir}/frame_{frame_id:03}.png", color_image)

        # Save depth image (8-bit) for visualization
        depth_vis = cv2.convertScaleAbs(depth_image, alpha=0.03)
        cv2.imwrite(f"{depth_vis_dir}/frame_{frame_id:03}.png", depth_vis)

        # Training MIT set parameters:
        # File: 0002070-000069074347.png, Bit Depth: uint16, Size: 640x480
        # Depth PNG value range: Min = 0, Max = 63992
        # Save raw depth image (16-bit)
        cv2.imwrite(f"{depth_raw_dir}/frame_{frame_id:03}.png", depth_image)

        frame_id += 1
        print(f"frame {frame_id} extracted")

except RuntimeError:
    print("Extraction finished (end of video)")

pipeline.stop()
print(f"Extracted {frame_id} frames, saved to {output_dir}/")

# Read .bag file
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(bag_file, repeat_playback=False)
pipeline.start(config)

# Wait for a frame
frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()

# Get intrinsics
intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
fx = intr.fx
fy = intr.fy
cx = intr.ppx
cy = intr.ppy

# Construct intrinsic matrix K
K = [
    [fx, 0,  cx],
    [0,  fy, cy],
    [0,  0,   1]
]

# Write to file
with open(output_dir+"intrinsics.txt", "w") as f:
    for row in K:
        f.write(" ".join(map(str, row)) + "\n")

print("Intrinsic matrix K saved as intrinsics.txt")
