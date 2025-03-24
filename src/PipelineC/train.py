#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from torch import nn
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from src.utils import depth_to_point_cloud, get_data, get_intrinsics, visualize_data
def post_process_labels(preds, table_indices=[17]):
    """
    将预训练模型的预测标签后处理为二分类标签。
    如果预测标签在 table_indices 中，则判定为“桌子”（1），否则为“非桌子”（0）。
    
    参数:
      preds: numpy 数组，形状为 (N,) ，每个元素为 0~39 的预测类别
      table_indices: list，表示对应桌子的类别编号（可根据实际情况调整）
    
    返回:
      binary_preds: numpy 数组，形状为 (N,) ，值为 1（桌子）或 0（非桌子）
    """
    binary_preds = np.isin(preds, table_indices).astype(np.int32)
    return binary_preds

def random_sampling(points, num_samples):
    if points.shape[0] > num_samples:
        idx = np.random.choice(points.shape[0], num_samples, replace=False)
        return points[idx]
    return points
# 从 DGCNN 源码中导入模型（此处假设 DGCNN 源码中定义了 dgcnn/pytorch/model.py，并提供了 get_model 接口）
from dgcnn.pytorch.model import DGCNN
class Args:
    def __init__(self, k=20, emb_dims=1024, dropout=0.5):
        self.k = k              # 邻域点数
        self.emb_dims = emb_dims  # 嵌入维度
        self.dropout = dropout    # Dropout 概率

# 使用自定义的参数创建实例
custom_args = Args(k=20, emb_dims=1024, dropout=0.3)
# 设置分割类别数（例如这里假设为 2 类：背景与目标）
num_classes = 40
model = DGCNN(custom_args, num_classes)
model = nn.DataParallel(model)

# 预训练权重文件位于当前 train 目录下，文件名为 model.1024.t7
pretrained_path = "D:\comp0248\COM0248_Coursework_2\src\PiplineC\dgcnn\pytorch\pretrained\model.1024.t7"
checkpoint = torch.load(pretrained_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
model.eval()
print("预训练 DGCNN 模型加载成功！")

#########################################
# 数据加载与处理
#########################################

DATASET_PATHS = [
    "data/CW2-Dataset/harvard_c5/hv_c5_1/",
    "data/CW2-Dataset/harvard_c6/hv_c6_1/",
    "data/CW2-Dataset/mit_76_studyroom/76-1studyroom2/",
    "data/CW2-Dataset/mit_32_d507/d507_2/",
    "data/CW2-Dataset/harvard_c11/hv_c11_2/",
    "data/CW2-Dataset/mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika/",
    "data/CW2-Dataset/mit_76_459/76-459b/"
]

# 选择数据集和数据编号
dataset_path = DATASET_PATHS[5]
data_id = 0

# 1. 加载 RGB、深度和标签数据，并可视化（工具函数来自 src/utils.py）
rgb, depth, labels = get_data(dataset_path, data_id)
#visualize_data(rgb, depth, labels)

# 2. 加载相机内参，并将深度图转换为点云
intrinsics = get_intrinsics(dataset_path)
print("相机内参：/n", intrinsics)
points = depth_to_point_cloud(depth, intrinsics)
print(f"加载到点云数据，共 {points.shape[0]} 个点。")
points = random_sampling(points, 16384)

#########################################
# 模型推理与分割
#########################################

# 将点云数据转换为 Tensor，并增加 batch 维度，形状为 (1, N, 3)
points_tensor = torch.from_numpy(points.astype(np.float32)).unsqueeze(0).transpose(2, 1)
with torch.no_grad():
    logits = model(points_tensor)  # 输出形状为 (1, N, num_classes)
    preds = logits.argmax(dim=-1).squeeze(0).cpu().numpy()  # (N,)

print("点云分割完成！")

#########################################
# 分割结果可视化
#########################################
max_label = preds.max()
# 确保 normed 是一个数组
normed = np.atleast_1d(preds / (max_label if max_label > 0 else 1))
cmap = plt.get_cmap("tab20")
mapped = cmap(normed)
colors = np.array(mapped)
if colors.ndim == 1:  # 如果只有一个颜色，则扩展为二维
    colors = colors[np.newaxis, :]
colors = colors[:, :3]

# 后处理预测标签
binary_preds = post_process_labels(preds)
print("二分类标签后处理完成！")
# 可视化二分类结果
colors = np.zeros((points.shape[0], 3))
colors[binary_preds == 1] = [0, 1, 0]  # 桌子（绿色）
colors[binary_preds == 0] = [1, 0, 0]  # 非桌子（红色）

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd])