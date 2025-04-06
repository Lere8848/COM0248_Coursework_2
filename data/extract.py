import pyrealsense2 as rs
import numpy as np
import cv2
import os

# 设置路径
bag_file = "data/realsense/20250328_105126.bag"
output_dir = "data/realsense/20250328_105126/"

os.makedirs(output_dir, exist_ok=True)

rgb_dir = os.path.join(output_dir, "image")
depth_vis_dir = os.path.join(output_dir, "depth_vis")
depth_raw_dir = os.path.join(output_dir, "depthTSDF")

# 创建子目录
os.makedirs(rgb_dir, exist_ok=True)
os.makedirs(depth_vis_dir, exist_ok=True)
os.makedirs(depth_raw_dir, exist_ok=True)

# 配置 pipeline 从 .bag 文件读取
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(bag_file, repeat_playback=False)
config.enable_stream(rs.stream.color)
config.enable_stream(rs.stream.depth)

# 启动
pipeline.start(config)
align = rs.align(rs.stream.color)
frame_id = 0

print("开始提取帧...")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # 转 numpy
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # 保存图像
        cv2.imwrite(f"{rgb_dir}/frame_{frame_id:03}.png", color_image)

        # 保存深度图（8-bit） 仅用于可视化
        depth_vis = cv2.convertScaleAbs(depth_image, alpha=0.03)
        cv2.imwrite(f"{depth_vis_dir}/frame_{frame_id:03}.png", depth_vis)

        # 训练集参数：
        # File: 0002070-000069074347.png, Bit Depth: uint16, Size: 640x480
        # Depth PNG value range: Min = 0, Max = 63992
        
        # 保存原始深度图（16-bit）
        cv2.imwrite(f"{depth_raw_dir}/frame_{frame_id:03}.png", depth_image)

        frame_id += 1
        print(f"frame {frame_id} extracted")

except RuntimeError:
    print("提取结束（视频播放完）")

pipeline.stop()
print(f"共提取 {frame_id} 帧，保存于 {output_dir}/")


# 读取 .bag 文件
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(bag_file, repeat_playback=False)
pipeline.start(config)

# 等待一帧
frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()

# 获取内参
intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
fx = intr.fx
fy = intr.fy
cx = intr.ppx
cy = intr.ppy

# 构造内参矩阵 K
K = [
    [fx, 0,  cx],
    [0,  fy, cy],
    [0,  0,   1]
]

# 写入文件
with open(output_dir+"intrinsics.txt", "w") as f:
    for row in K:
        f.write(" ".join(map(str, row)) + "\n")

print("K 矩阵已保存为 intrinsics.txt")