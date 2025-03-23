import pyrealsense2 as rs
import numpy as np
import cv2
import os

# 设置路径
bag_file = "data/realsense/20250321_105312.bag"
output_dir = "data/realsense/frames_output_20250321_105312"

os.makedirs(output_dir, exist_ok=True)

rgb_dir = os.path.join(output_dir, "rgb")
depth_vis_dir = os.path.join(output_dir, "depth_vis")
depth_raw_dir = os.path.join(output_dir, "depth_raw")

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

        # 保存原始深度图（16-bit）
        cv2.imwrite(f"{depth_raw_dir}/frame_{frame_id:03}.png", depth_image)

        frame_id += 1
        print(f"frame {frame_id} extracted")

except RuntimeError:
    print("提取结束（视频播放完）")

pipeline.stop()
print(f"共提取 {frame_id} 帧，保存于 {output_dir}/")
