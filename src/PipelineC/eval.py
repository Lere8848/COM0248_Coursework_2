#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from src.PipelineC.dgcnn_seg import DGCNN_seg 
from src.utils import get_intrinsics
import cv2


#########################################
# 数据加载与处理
#########################################

DATASET_PATHS = [
    "data/CW2_dataset/harvard_c5/hv_c5_1/",
    "data/CW2_dataset/harvard_c6/hv_c6_1/",
    "data/CW2_dataset/mit_76_studyroom/76-1studyroom2/",
    "data/CW2_dataset/mit_32_d507/d507_2/",
    "data/CW2_dataset/harvard_c11/hv_c11_2/",
    "data/CW2_dataset/mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika/",
    "data/CW2_dataset/mit_76_459/76-459b/"
]

# 选择数据集和数据编号
dataset_path = DATASET_PATHS[5]
data_id = 0


# ========== 参数 ==========
MODEL_PATH = "best_dgcnn_table_seg.pth"
DEPTH_PATH = "data/CW2_Dataset/harvard_c11/hv_c11_2/depthTSDF/0000317-000010578052.png"     # 你的新深度图路径

NUM_POINTS = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================

def depth_to_pointcloud(depth, K):
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u_flat = u.flatten()
    v_flat = v.flatten()
    z = depth.flatten()
    x = (u_flat - K[0, 2]) / K[0, 0]
    y = (v_flat - K[1, 2]) / K[1, 1]
    x *= z
    y *= z
    points = np.vstack((x, y, z)).T
    valid = z > 0
    return points[valid], u_flat[valid], v_flat[valid]

def sample_pointcloud(points, num_points):
    N = points.shape[0]
    if N >= num_points:
        idx = np.random.choice(N, num_points, replace=False)
    else:
        idx = np.concatenate([np.arange(N), np.random.choice(N, num_points - N, replace=True)])
    return points[idx]

def visualize(points, labels):
    colors = np.zeros_like(points)
    colors[:] = [0.5, 0.5, 0.5]  # 默认灰色
    colors[labels == 1] = [1.0, 0.0, 0.0]  # 桌子红色

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

def main():
    # 加载深度图
    depth = cv2.imread(DEPTH_PATH, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError("无法读取深度图")
    if depth.dtype != np.float32:
        depth = depth.astype(np.float32)
    if depth.max() > 100:  # 假设单位是毫米
        depth = depth / 1000.0

    # 加载相机内参
    intrinsics = get_intrinsics(dataset_path)
    K = intrinsics

    # 深度图转点云
    points_full, _, _ = depth_to_pointcloud(depth, K)
    points = sample_pointcloud(points_full, NUM_POINTS)

    # 加载模型
    model = DGCNN_seg(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        input_points = torch.from_numpy(points).float().unsqueeze(0).to(DEVICE)  # (1, N, 3)
        input_points = input_points.permute(0, 2, 1)  # (1, 3, N)
        preds = model(input_points)  # (1, 2, N)
        pred_labels = preds.argmax(dim=1).squeeze().cpu().numpy()  # (N,)

    print(f"检测到桌子点数量：{np.sum(pred_labels == 1)} / {NUM_POINTS}")

    # 可视化
    visualize(points, pred_labels)

if __name__ == "__main__":
    main()
