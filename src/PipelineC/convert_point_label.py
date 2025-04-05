import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from dgcnn.pytorch.model import DGCNN,knn, get_graph_feature
import torch
from torch import nn
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from src.utils import depth_to_point_cloud, get_data, get_intrinsics,get_num_images, visualize_data
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
    
DATASET_PATHS = [
    "data/CW2_dataset/harvard_c5/hv_c5_1/",
    "data/CW2_dataset/harvard_c6/hv_c6_1/",
    "data/CW2_dataset/mit_76_studyroom/76-1studyroom2/",
    "data/CW2_dataset/mit_32_d507/d507_2/",
    "data/CW2_dataset/harvard_c11/hv_c11_2/",
    "data/CW2_dataset/mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika/",
    "data/CW2_dataset/mit_76_459/76-459b/"
]

def random_sampling(points, num_samples):
    if points.shape[0] > num_samples:
        idx = np.random.choice(points.shape[0], num_samples, replace=False)
        return points[idx]
    return points

def depth_to_pointcloud_with_labels(depth, K, polygons, image_size):
    """
    Args:
        depth: (H, W) numpy array, depth image
        K: (3, 3) camera intrinsics
        polygons: list of list of (x, y), e.g. [[(x1, y1), (x2, y2), ...], ...]
        image_size: (H, W), same as depth

    Returns:
        points: (N, 3) point cloud
        labels: (N,) array of 0 or 1
    """
    # Generate full pixel grid
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u_flat = u.flatten()
    v_flat = v.flatten()
    z = depth.flatten()

    # Camera intrinsics
    x = (u_flat - K[0, 2]) / K[0, 0]
    y = (v_flat - K[1, 2]) / K[1, 1]
    x *= z
    y *= z

    points = np.vstack((x, y, z)).T

    # 过滤掉 z <= 0 的点
    valid_mask = z > 0
    points = points[valid_mask]
    u_valid = u_flat[valid_mask]
    v_valid = v_flat[valid_mask]

    # Create polygon mask (in image space)
    mask = np.zeros(image_size, dtype=np.uint8)
    for polygon in polygons:
        polygon_np = np.array([polygon], dtype=np.int32)
        cv2.fillPoly(mask, polygon_np, 1)

    # 查询每个点的 (u, v) 是否在 mask 中
    labels = mask[v_valid, u_valid]  # 注意 v 是 y，u 是 x
    return points, labels

def extract_polygons_from_labels(labels):
    """
    Convert labels like [(x_list, y_list), ...] to OpenCV polygon format.
    Returns:
        List of list of (x, y) integer tuples.
    """
    polygons = []
    if labels is not None:
        for (x_list, y_list) in labels:
            polygon = list(zip(map(int, x_list), map(int, y_list)))
            polygons.append(polygon)
    return polygons

def balanced_sample_pointcloud(points, labels, num_points, table_ratio=0.8):
    """
    从点云中采样，使得桌子点（label=1）占比达到 table_ratio，剩下的为背景。
    不足部分会随机重复采样。
    """
    table_points = points[labels == 1]
    bg_points = points[labels == 0]

    num_table = int(num_points * table_ratio)
    num_bg = num_points - num_table

    # 如果某类数量不足，使用重复采样
    def sample_with_repeat(pts, n):
        if len(pts) == 0:
            return np.zeros((n, 3))  # 如果完全没有点（极端情况），返回空白点
        elif len(pts) >= n:
            idx = np.random.choice(len(pts), n, replace=False)
        else:
            idx = np.concatenate([
                np.arange(len(pts)),
                np.random.choice(len(pts), n - len(pts), replace=True)
            ])
        return pts[idx]

    sampled_table = sample_with_repeat(table_points, num_table)
    sampled_bg = sample_with_repeat(bg_points, num_bg)

    points_sampled = np.concatenate([sampled_table, sampled_bg], axis=0)
    labels_sampled = np.concatenate([
        np.ones(len(sampled_table), dtype=np.int64),
        np.zeros(len(sampled_bg), dtype=np.int64)
    ], axis=0)

    # 打乱
    perm = np.random.permutation(num_points)
    return points_sampled[perm], labels_sampled[perm]

# # 选择数据集和数据编号
# dataset_path = DATASET_PATHS[5]
# data_id = 0

# # 1. load RGB, depth and labels data, and visualize (functions from src/utils.py)
# rgb, depth, label_polygons = get_data(dataset_path, data_id)
# #visualize_data(rgb, depth, labels)
# intrinsics = get_intrinsics(dataset_path)
# img_size = depth.shape[:2]

# # 2. extract polygons from labels
# polygans = extract_polygons_from_labels(label_polygons)

# #3. convert depth to point cloud and labels
# points, point_labels= depth_to_pointcloud_with_labels(depth, intrinsics, polygans, img_size)
# # points = depth_to_point_cloud(depth, intrinsics)
# print(f"点云共 {points.shape[0]} 个点，其中桌子点数量为 {np.sum(point_labels)}")
# #points = random_sampling(points, 16384)
# '''
# visualize point cloud
# '''
# # colors = np.zeros_like(points)
# # colors[:] = [0.5, 0.5, 0.5]  # 背景灰
# # colors[point_labels == 1] = [1.0, 0.0, 0.0]  # 桌子红

# # pcd = o3d.geometry.PointCloud()
# # pcd.points = o3d.utility.Vector3dVector(points)
# # pcd.colors = o3d.utility.Vector3dVector(colors)
# # o3d.visualization.draw_geometries([pcd])
# #4. extract table points from point cloud
# table_points = points[point_labels == 1]
# #5. sample table points to match dgcnn input size
# points_sampled, label_smapled= sample_pointcloud(points, point_labels, 2048)


SAVE_DIR = "data/processed_data"
os.makedirs(SAVE_DIR, exist_ok=True)
NUM_POINTS = 4096

def save_all_processed_data():
    for dataset_path in DATASET_PATHS:
        scene_name = dataset_path.strip("/").split("/")[-2]
        num_images = get_num_images(dataset_path)

        save_dir = os.path.join(SAVE_DIR, scene_name)
        os.makedirs(save_dir, exist_ok=True)  

        for data_id in tqdm(range(num_images), desc=scene_name):
            rgb, depth, label_polygons = get_data(dataset_path, data_id)
            if depth is None or label_polygons is None:
                continue

            intrinsics = get_intrinsics(dataset_path)
            img_size = depth.shape[:2]
            polygons = extract_polygons_from_labels(label_polygons)

            points, point_labels = depth_to_pointcloud_with_labels(depth, intrinsics, polygons, img_size)
            if len(points) < 10 or np.sum(point_labels == 1) == 0:
                continue  # 跳过没有桌子或无效点

            # 👇 使用新的平衡采样函数
            points_sampled, labels_sampled = balanced_sample_pointcloud(points, point_labels, NUM_POINTS, table_ratio=0.8)

            # # 可视化（可选）
            # colors = np.zeros_like(points_sampled)
            # colors[:] = [0.5, 0.5, 0.5]  # 背景灰
            # colors[labels_sampled == 1] = [1.0, 0.0, 0.0]  # 桌子红

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(points_sampled)
            # pcd.colors = o3d.utility.Vector3dVector(colors)
            # o3d.visualization.draw_geometries([pcd])

            print("桌子点占比:", np.sum(labels_sampled) / NUM_POINTS)

            # 保存 .npz 文件
            filename = f"{scene_name}_{data_id}.npz"
            save_path = os.path.join(save_dir, filename)
            np.savez_compressed(save_path,
                                points=points_sampled.astype(np.float32),
                                labels=labels_sampled.astype(np.int64))

if __name__ == "__main__":
    save_all_processed_data()