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

def sample_pointcloud(points, label, num_points):
    """
    统一点云为定长点数
    """
    N = points.shape[0]
    if N >= num_points:
        idx = np.random.choice(N, num_points, replace=False)
    else:
        idx = np.concatenate([
            np.arange(N),
            np.random.choice(N, num_points - N, replace=True)
        ])
    return points[idx], label[idx]

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
            
            if len(points) < 10:
                continue  # 过滤异常样本

            points_sampled, labels_sampled = sample_pointcloud(points, point_labels, NUM_POINTS)

            # colors = np.zeros_like(points_sampled)
            # colors[:] = [0.5, 0.5, 0.5]  # 背景灰
            # colors[labels_sampled == 1] = [1.0, 0.0, 0.0]  # 桌子红

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(points_sampled)
            # pcd.colors = o3d.utility.Vector3dVector(colors)
            # o3d.visualization.draw_geometries([pcd])
            #print("桌子点占比:", np.sum(labels_sampled) / NUM_POINTS)
            filename = f"{scene_name}_{data_id}.npz"
            save_path = os.path.join(save_dir, filename)
            np.savez_compressed(save_path,
                                points=points_sampled.astype(np.float32),
                                labels=labels_sampled.astype(np.int64))

if __name__ == "__main__":
    save_all_processed_data()