#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import numpy as np
import open3d as o3d
import cv2
from pathlib import Path
from src.PipelineC.dgcnn_seg import DGCNN_seg 
from src.utils import get_intrinsics
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from tqdm import tqdm
# ======== 参数 ========
MODEL_PATH = "best_dgcnn_epoch_22.pth"
DEPTH_FOLDER = "data/CW2_Dataset/harvard_c5/hv_c5_1/depthTSDF"  # 深度图目录
DATASET_PATH = "data/CW2_Dataset/harvard_c5/hv_c5_1/"           # 相机内参路径
GT_FOLDER = "data/processed_data/harvard_c5"
NUM_POINTS = 4096
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ======================

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
    return points[valid]

def sample_pointcloud(points, num_points):
    N = points.shape[0]
    if N >= num_points:
        idx = np.random.choice(N, num_points, replace=False)
    else:
        idx = np.concatenate([np.arange(N), np.random.choice(N, num_points - N, replace=True)])
    return points[idx]

def sample_pointcloud_label(points, labels, num_points):
    N = points.shape[0]
    if N >= num_points:
        idx = np.random.choice(N, num_points, replace=False)
    else:
        idx = np.concatenate([
            np.arange(N),
            np.random.choice(N, num_points - N, replace=True)
        ])
    return points[idx], labels[idx]

def visualize(points, labels, title=""):
    colors = np.zeros_like(points)
    colors[:] = [0.5, 0.5, 0.5]  # 默认灰色
    colors[labels == 1] = [1.0, 0.0, 0.0]  # 桌子红色
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    print(f"可视化: {title}")
    o3d.visualization.draw_geometries([pcd])

def evaluate(pred, gt):
    return {
        "f1": f1_score(gt, pred),
        "iou": jaccard_score(gt, pred),
        "precision": precision_score(gt, pred),
        "recall": recall_score(gt, pred)
    }
def main():
    # 加载模型
    model = DGCNN_seg(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 相机内参
    intrinsics = get_intrinsics(DATASET_PATH)
    depth_files = sorted(list(Path(DEPTH_FOLDER).glob("*.png")))
    print(f"共找到 {len(depth_files)} 张深度图")


    all_metrics = []

    for i, depth_path in tqdm(enumerate(depth_files), total=len(depth_files), desc="评估中"):
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            print(f"⚠️ 无法读取：{depth_path.name}")
            continue

        if depth.dtype != np.float32:
            depth = depth.astype(np.float32)
        if depth.max() > 100:
            depth = depth / 1000.0

        # 生成点云
        points_full = depth_to_pointcloud(depth, intrinsics)

        # 加载 ground truth（顺序命名）
        gt_path = os.path.join(GT_FOLDER, f"harvard_c5_{i}.npz")
        if not os.path.exists(gt_path):
            print(f"❌ GT 文件不存在: {gt_path}")
            continue

        npz_data = np.load(gt_path)
        gt_labels = npz_data["labels"]
        gt_points = npz_data["points"]


        # 采样点云和标签
        gt_point_sampled, gt_sampled= sample_pointcloud_label(gt_points, gt_labels, NUM_POINTS)
        points = sample_pointcloud(points_full, NUM_POINTS)
        with torch.no_grad():
            input_points = torch.from_numpy(points).float().unsqueeze(0).to(DEVICE)
            input_points = input_points.permute(0, 2, 1)
            preds = model(input_points)
            pred_labels = preds.argmax(dim=1).squeeze().cpu().numpy()

        metrics = evaluate(pred_labels, gt_sampled)
        all_metrics.append(metrics)

        print(f"[{depth_path.name}] F1: {metrics['f1']:.4f} | IoU: {metrics['iou']:.4f} | "
              f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")

        # 可视化
        visualize(points, pred_labels, title=depth_path.name)



    # 统计平均结果
    if all_metrics:
        avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
        print("\n📊 整体评估结果:")
        print(f"F1 Score   : {avg['f1']:.4f}")
        print(f"IoU        : {avg['iou']:.4f}")
        print(f"Precision  : {avg['precision']:.4f}")
        print(f"Recall     : {avg['recall']:.4f}")
if __name__ == "__main__":
    main()
