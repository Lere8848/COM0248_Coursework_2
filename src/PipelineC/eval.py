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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ======== settings ========
MODEL_PATH = "weights/PipelineC/best.pth"
NUM_POINTS = 4096   # should be same as convert_point_label.py setting
# inference check
depth_path = "data/realsense_testset/20250328_105024/depthTSDF/frame_037.png"
intrinsics_path = "data/realsense_testset/20250328_105024/" 
# test set check
TEST_CONFIGS = [
    {
        "depth_path": "data/CW2_dataset/harvard_c6/hv_c6_1/",
        "gt_path": "data/test_data/harvard_c6",
        "prefix": "harvard_c6"
    },
    {
        "depth_path": "data/CW2_dataset/harvard_c11/hv_c11_2/",
        "gt_path": "data/test_data/harvard_c11",
        "prefix": "harvard_c11"
    },
    {
        "depth_path": "data/CW2_dataset/harvard_c5/hv_c5_1/",
        "gt_path": "data/test_data/harvard_c5",
        "prefix": "harvard_c5"
    },
    {
        "depth_path": "data/CW2_dataset/harvard_tea_2/hv_tea2_2/",
        "gt_path": "data/test_data/harvard_tea2",
        "prefix": "harvard_tea_2"
    },

]
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


def evaluate(pred, gt):
    return {
        "f1": f1_score(gt, pred, pos_label=1, average='binary', zero_division=0),
        "iou": jaccard_score(gt, pred, pos_label=1, average='binary', zero_division=0),
        "precision": precision_score(gt, pred, pos_label=1, average='binary', zero_division=0),
        "recall": recall_score(gt, pred, pos_label=1, average='binary', zero_division=0)
    }

def save_confusion_matrix(gt, pred, save_path, labels=["bg", "table"]):
    cm = confusion_matrix(gt, pred, labels=[0, 1])
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
def save_pointcloud_visualization(points, labels, save_path):
    colors = np.zeros_like(points)
    colors[:] = [0.5, 0.5, 0.5]
    colors[labels == 1] = [1.0, 0.0, 0.0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(save_path)
    vis.destroy_window()

def run_inference_without_gt(depth_path, intrinsics_path):
    model = DGCNN_seg(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # load depth image
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        print(f"Can't load depth: {depth_path}")
        return

    if depth.dtype != np.float32:
        depth = depth.astype(np.float32)
    if depth.max() > 100:
        depth = depth / 1000.0

    # load intrinsics
    intrinsics = get_intrinsics(intrinsics_path)

    # to point cloud
    points_full = depth_to_pointcloud(depth, intrinsics)
    sampled_points = sample_pointcloud(points_full, NUM_POINTS)

    # inference
    with torch.no_grad():
        input_tensor = torch.from_numpy(sampled_points).float().unsqueeze(0).to(DEVICE)
        input_tensor = input_tensor.permute(0, 2, 1)
        preds = model(input_tensor)
        pred_labels = preds.argmax(dim=1).squeeze().cpu().numpy()
    
    # save visualization
    out_dir = "results/inference_only"
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "pred.png")
    save_pointcloud_visualization(sampled_points, pred_labels, save_path)

    print(f"Prediction complete. Saved to {save_path}")

def main():
    model = DGCNN_seg(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_metrics = []
    all_gt_labels = []
    all_pred_labels = []

    for config in TEST_CONFIGS:
        dataset_path = config["depth_path"]
        gt_path = config["gt_path"]
        file_prefix = config["prefix"]

        scene_name = file_prefix
        print(f"\n====== Evaluating scene: {scene_name} ======")

        depth_folder = os.path.join(dataset_path, "depthTSDF")
        if not os.path.exists(depth_folder):
            print(f"Depth folder not found: {depth_folder}")
            continue

        intrinsics = get_intrinsics(dataset_path)
        depth_files = sorted(list(Path(depth_folder).glob("*.png")))
        print(f"Found {len(depth_files)} depth images in {scene_name}")

        scene_metrics = []
        scene_gts = []
        scene_preds = []

        for i, depth_path in tqdm(enumerate(depth_files), total=len(depth_files), desc=f"Scene: {scene_name}"):
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth is None:
                print(f"Can't load depth: {depth_path.name}")
                continue

            if depth.dtype != np.float32:
                depth = depth.astype(np.float32)
            if depth.max() > 100:
                depth = depth / 1000.0

            points_full = depth_to_pointcloud(depth, intrinsics)

            gt_file = os.path.join(gt_path, f"{file_prefix}_{i}.npz")
            if not os.path.exists(gt_file):
                print(f"GT missing: {gt_file} — defaulting to all background (no table)")

                # Generate pseudo ground truth: all background (label 0)
                sampled_points = sample_pointcloud(points_full, NUM_POINTS)
                gt_sampled_labels = np.zeros(NUM_POINTS, dtype=np.int64)
            else:
                npz_data = np.load(gt_file)
                gt_labels = npz_data["labels"]
                gt_points = npz_data["points"]
                gt_sampled_points, gt_sampled_labels = sample_pointcloud_label(gt_points, gt_labels, NUM_POINTS)
                sampled_points = sample_pointcloud(points_full, NUM_POINTS)

            with torch.no_grad():
                input_tensor = torch.from_numpy(sampled_points).float().unsqueeze(0).to(DEVICE)
                input_tensor = input_tensor.permute(0, 2, 1)
                preds = model(input_tensor)
                pred_labels = preds.argmax(dim=1).squeeze().cpu().numpy()

            # clloect teh metrics
            scene_gts.append(gt_sampled_labels)
            scene_preds.append(pred_labels)
            all_gt_labels.append(gt_sampled_labels)
            all_pred_labels.append(pred_labels)

            # calculate metrics
            metrics = evaluate(pred_labels, gt_sampled_labels)
            scene_metrics.append(metrics)
            all_metrics.append(metrics)

            print(f"[{depth_path.name}] F1: {metrics['f1']:.4f} | IoU: {metrics['iou']:.4f} | "
                  f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")

        # save the confusion matrix for the scene
        if scene_metrics:
            avg_scene = {k: np.mean([m[k] for m in scene_metrics]) for k in scene_metrics[0]}
            print(f"\nScene [{scene_name}] Average Metrics:")
            print(f"F1 Score   : {avg_scene['f1']:.4f}")
            print(f"IoU        : {avg_scene['iou']:.4f}")
            print(f"Precision  : {avg_scene['precision']:.4f}")
            print(f"Recall     : {avg_scene['recall']:.4f}")

            scene_all_gt = np.concatenate(scene_gts)
            scene_all_pred = np.concatenate(scene_preds)

            vis_folder = os.path.join("output_vis", scene_name)
            os.makedirs(vis_folder, exist_ok=True)
            scene_cm_path = os.path.join(vis_folder, f"{scene_name}_avg_cm.png")
            save_confusion_matrix(scene_all_gt, scene_all_pred, scene_cm_path)

    # calculate overall metrics
    if all_metrics:
        avg_total = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
        print(f"\n====== Overall Metrics Across All Scenes ======")
        print(f"F1 Score   : {avg_total['f1']:.4f}")
        print(f"IoU        : {avg_total['iou']:.4f}")
        print(f"Precision  : {avg_total['precision']:.4f}")
        print(f"Recall     : {avg_total['recall']:.4f}")

        final_gt = np.concatenate(all_gt_labels)
        final_pred = np.concatenate(all_pred_labels)

        os.makedirs("output_vis", exist_ok=True)
        final_cm_path = os.path.join("output_vis", "all_scenes_avg_cm.png")
        save_confusion_matrix(final_gt, final_pred, final_cm_path)


if __name__ == "__main__":
    main()

    run_inference_without_gt(depth_path, intrinsics_path)