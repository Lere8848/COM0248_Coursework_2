import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from src.utils import depth_to_point_cloud, get_data, get_intrinsics, visualize_data

DATASET_PATHS = ["data/CW2_dataset/harvard_c5/hv_c5_1/",
         "data/CW2_dataset/harvard_c6/hv_c6_1/",
         "data/CW2_dataset/mit_76_studyroom/76-1studyroom2/",
         "data/CW2_dataset/mit_32_d507/d507_2/",
         "data/CW2_dataset/harvard_c11/hv_c11_2/",
         "data/CW2_dataset/mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika/",
         "data/CW2_dataset/mit_76_459/76-459b/"]

NUM_IMAGES = [23, 35, 48, 108, 13, 13, 43]
# ----------------------------
# 1. 加载预训练的 Point Transformer 分割模型
# ----------------------------
# 通过 torch.hub.load 直接加载官方预训练模型
# 请确保网络环境正常，模型会自动下载并缓存
model = torch.hub.load('POSTECH-CVLab/PointTransformer', 
                       'point_transformer_segmentation', 
                       pretrained=True)
model.eval()
print("预训练 Point Transformer 分割模型加载成功！")

# ----------------------------
# 2. 加载点云数据
# ----------------------------
# Load data
dataset_path = DATASET_PATHS[0]
data_id = 0
rgb, depth, labels = get_data(dataset_path, data_id)
visualize_data(rgb, depth, labels)

# Load intrinsics
intrinsics = get_intrinsics(dataset_path)
print(intrinsics)
points = depth_to_point_cloud(depth, intrinsics)  # 点云数据形状应为 (N, 3)
print(f"加载到点云数据，共 {points.shape[0]} 个点。")

# ----------------------------
# 3. 对点云进行分割预测
# ----------------------------
# 假设模型输入形状为 (B, N, 3)
points_tensor = torch.from_numpy(points.astype(np.float32)).unsqueeze(0)  # (1, N, 3)
with torch.no_grad():
    logits = model(points_tensor)  # 输出形状为 (1, N, num_classes)
    preds = logits.argmax(dim=-1).squeeze(0).cpu().numpy()  # (N,)
print("点云分割完成！")

# ----------------------------
# 4. 可视化分割结果
# ----------------------------
max_label = preds.max()
colors = plt.get_cmap("tab20")(preds / (max_label if max_label > 0 else 1))[:, :3]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd])
